"""New family: deep 1D residual network (XResNet1d-style), PPG-only regression.

The published PPG-only ceiling (Moulaeifard, Charlton & Strodthoff 2025) used
XResNet1d101 on PulseDB, so this backbone is the literal reference the project is
measured against, it belongs in the architecture screen. A strided-downsample
residual stack (basic blocks, GroupNorm for batch safety, ReLU), global average
pool, two regression heads (SBP, DBP). Single PPG channel in, or +VPG/+APG.
"""
from pathlib import Path

import torch.nn.functional as F
from torch import nn

from bp_inference import train


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class ResBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1)
        self.norm1 = _norm(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.norm2 = _norm(cout)
        self.drop = nn.Dropout(dropout)
        self.proj = (nn.Conv1d(cin, cout, 1, stride=stride)
                     if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):                            # (B, C, T)
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.drop(self.norm2(self.conv2(h)))
        return F.relu(h + self.proj(x))


class XResNet1d(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        c = int(model_cfg.get("base_channels", 32))
        depth = int(model_cfg.get("blocks_per_stage", 2))
        n_stages = int(model_cfg.get("n_stages", 4))
        dropout = float(model_cfg.get("dropout", 0.2))
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, 7, stride=2, padding=3), _norm(c), nn.ReLU())
        blocks, cin = [], c
        for s in range(n_stages):
            cout = c * (2 ** s)
            for b in range(depth):
                stride = 2 if (b == 0 and s > 0) else 1
                blocks.append(ResBlock1D(cin, cout, stride, dropout))
                cin = cout
        self.stages = nn.Sequential(*blocks)
        self.head = nn.Linear(cin, n_targets)

    def forward(self, x):                            # (B, T, C)
        h = self.stem(x.transpose(1, 2))
        h = self.stages(h)
        return self.head(h.mean(dim=-1))             # global average pool over time


def _factory(in_channels, T, model_cfg, n_targets):
    return XResNet1d(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "xresnet1d")
