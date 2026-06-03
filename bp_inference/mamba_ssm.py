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


class SSMBlock(nn.Module):
    """Diagonal linear recurrence h_t = a * h_{t-1} + g_t * x_t with SiLU gating.

    `a = sigmoid(a_logit)` is a per-channel decay in (0, 1) (time-invariant,
    S4-lite). `g_t = sigmoid(W x_t)` is the input-dependent selective gate. The
    scan is sequential over time (O(T)); fine for PPG-length sequences.
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
        h = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        outs = []
        for t in range(x.shape[1]):
            h = a * h + g[:, t] * uc[:, t]
            outs.append(h)
        y = torch.stack(outs, dim=1)             # (B, L, D)
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
