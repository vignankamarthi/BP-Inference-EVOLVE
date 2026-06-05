"""New family: structured-state-space (S4-lite), PPG-only regression.

A learnable long depthwise causal convolution kernel per channel realizes the
"S4 as a global convolution" view of a state-space model, a long receptive
field without a sequential scan (and distinct from the mamba input-gated
recurrence). Single PPG channel in, or +VPG/+APG, two heads. Lineage: Gu,
Goel & Re 2022 (S4); kernel truncated to `kernel_len` for portability.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bp_inference import train


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class S4Block(nn.Module):
    def __init__(self, ch: int, kernel_len: int, dropout: float):
        super().__init__()
        self.L = kernel_len
        self.kernel = nn.Parameter(torch.randn(ch, kernel_len) / (kernel_len ** 0.5))
        self.mix = nn.Conv1d(ch, ch, 1)
        self.norm = _norm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                            # (B, ch, T)
        xp = F.pad(x, (self.L - 1, 0))               # causal left pad
        y = F.conv1d(xp, self.kernel.unsqueeze(1), groups=x.shape[1])   # depthwise long conv
        y = self.drop(F.relu(self.norm(self.mix(y))))
        return x + y


class S4Net(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        ch = int(model_cfg.get("channels", 48))
        n_blocks = int(model_cfg.get("n_blocks", 3))
        kernel_len = min(int(model_cfg.get("kernel_len", 128)), T)
        dropout = float(model_cfg.get("dropout", 0.2))
        self.stem = nn.Conv1d(in_channels, ch, 1)
        self.blocks = nn.ModuleList([S4Block(ch, kernel_len, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(ch, n_targets))

    def forward(self, x):                            # (B, T, C)
        h = self.stem(x.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.mean(dim=-1))


def _factory(in_channels, T, model_cfg, n_targets):
    return S4Net(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "s4")
