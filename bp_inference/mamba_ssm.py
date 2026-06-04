"""Seed 4: lightweight selective-state-space (Mamba-style) U-Net (PPG-only).

Modern frontier seed. Lineage: Mamba-UNet (2025) and BM-BPW bidirectional Mamba
(2025, EMBC) for cuffless BP. The true Mamba selective scan needs the mamba-ssm
CUDA kernel, which is banned here for portability/hardware reasons; this is a
dependency-free, stable approximation: a diagonal linear recurrence (time-
invariant decay per channel) with an input-dependent gate and SiLU output
gating, the structural skeleton of a selective SSM. The loop can later swap in
the real kernel where available. Single PPG channel, two regression heads.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bp_inference import train


def ssm_scan(a: torch.Tensor, b: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Exact parallel form of the recurrence h_t = a * h_{t-1} + b_t (h_{-1}=0).

    `a` is a per-channel decay of shape (D,), constant over time, so the
    recurrence is linear and *time-invariant* and has a closed parallel form.
    We use a chunked scan: an intra-chunk decay-weighted prefix sum (one einsum,
    parallel over all chunks) plus a short sequential carry across the L/chunk
    chunk boundaries. This replaces the O(L) Python loop with ~L/chunk
    sequential steps, exact to floating point. With chunk M[i,j,d]=a_d^(i-j)
    (i>=j) the within-chunk sum is local[c,i]=sum_{j<=i} a^(i-j) b[c,j], and the
    boundary state obeys S_{c+1}=a^C S_c + local[c,C-1], giving the global state
    h[c,i] = local[c,i] + a^(i+1) S_c.

    Args:
        a: (D,) decay coefficients, expected in (0, 1).
        b: (B, L, D) input term (already g_t * x_t in the SSM block).
        chunk: chunk length; capped at L.
    Returns:
        (B, L, D) scanned states h_t for t in [0, L).
    """
    B, L, D = b.shape
    C = min(chunk, L)
    n_chunks = (L + C - 1) // C
    pad = n_chunks * C - L
    if pad:
        b = F.pad(b, (0, 0, 0, pad))                 # zero-pad time axis at the end
    bc = b.view(B, n_chunks, C, D)

    idx = torch.arange(C, device=b.device)
    expo = idx[:, None] - idx[None, :]               # (C, C): i - j
    lower = expo >= 0
    M = torch.where(                                 # M[i,j,d] = a_d^(i-j) if i>=j else 0
        lower[..., None],
        a.pow(expo.clamp(min=0).to(a.dtype)[..., None]),
        torch.zeros((), dtype=a.dtype, device=b.device),
    )
    local = torch.einsum("ijd,bcjd->bcid", M, bc)    # (B, n_chunks, C, D)

    a_C = a.pow(C)                                    # (D,)
    last = local[:, :, C - 1, :]                      # (B, n_chunks, D)
    S = torch.zeros(B, D, dtype=b.dtype, device=b.device)
    carries = []
    for c in range(n_chunks):                         # ~L/chunk sequential steps
        carries.append(S)
        S = a_C * S + last[:, c]
    carries = torch.stack(carries, dim=1)             # (B, n_chunks, D): state before each chunk

    a_pow = a.pow(torch.arange(1, C + 1, device=b.device).to(a.dtype)[:, None])  # (C, D): a^(i+1)
    h = local + a_pow[None, None] * carries[:, :, None, :]   # (B, n_chunks, C, D)
    return h.reshape(B, n_chunks * C, D)[:, :L]


class SSMBlock(nn.Module):
    """Diagonal linear recurrence h_t = a * h_{t-1} + g_t * x_t with SiLU gating.

    `a = sigmoid(a_logit)` is a per-channel decay in (0, 1) (time-invariant,
    S4-lite). `g_t = sigmoid(W x_t)` is the input-dependent selective gate. The
    scan over time is computed in parallel by `ssm_scan` (chunked) -- exact to
    the sequential recurrence but ~L/chunk sequential steps instead of L, which
    is what makes a 1250-sample PPG window tractable on the GPU.
    """

    def __init__(self, dim: int, conv_k: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.out_gate = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dwconv = nn.Conv1d(dim, dim, conv_k, padding=conv_k - 1, groups=dim)
        self.a_logit = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                        # (B, L, D)
        res = x
        x = self.norm(x)
        u = self.in_proj(x)
        # short causal depthwise conv for local context
        uc = self.dwconv(u.transpose(1, 2))[..., :u.shape[1]].transpose(1, 2)
        uc = F.silu(uc)
        g = torch.sigmoid(self.gate(x))          # selective input gate
        a = torch.sigmoid(self.a_logit)          # (D,) decay
        y = ssm_scan(a, g * uc)                  # exact parallel linear-recurrence scan
        y = y * F.silu(self.out_gate(x))         # output gating
        return res + self.out_proj(y)


class MambaSSMNet(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        dim = int(model_cfg.get("dim", 48))
        n_layers = int(model_cfg.get("n_layers", 2))
        self.stem = nn.Conv1d(in_channels, dim, 7, padding=3)
        self.blocks = nn.ModuleList([SSMBlock(dim) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Dropout(float(model_cfg.get("dropout", 0.2))),
            nn.Linear(dim, n_targets))

    def forward(self, x):                        # (B, T, C)
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)   # (B, T, dim)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.mean(dim=1))          # global temporal pool


def _factory(in_channels, T, model_cfg, n_targets):
    return MambaSSMNet(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "mamba_ssm")
