"""New family: Inception-1D, PPG-only regression.

Each block runs parallel conv branches at several kernel sizes (multi-scale
features at one resolution) plus a pooling branch, concatenated; blocks stack
with max-pool downsampling. Single PPG channel in, or +VPG/+APG, two heads.
Lineage: Szegedy et al. 2015 (Inception), 1D adaptation.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bp_inference import train


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class InceptionBlock(nn.Module):
    def __init__(self, cin: int, c: int, kernels=(3, 7, 15, 31)):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, c, 1)
        self.branches = nn.ModuleList([nn.Conv1d(c, c, k, padding=k // 2) for k in kernels])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                         nn.Conv1d(c, c, 1))
        self.out_ch = c * (len(kernels) + 1)
        self.norm = _norm(self.out_ch)

    def forward(self, x):
        z = self.bottleneck(x)
        outs = [b(z) for b in self.branches] + [self.pool_branch(z)]
        return F.relu(self.norm(torch.cat(outs, dim=1)))


class Inception1d(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        c = int(model_cfg.get("channels", 16))
        n_blocks = int(model_cfg.get("n_blocks", 3))
        dropout = float(model_cfg.get("dropout", 0.2))
        blocks, cin = [], in_channels
        for _ in range(n_blocks):
            blk = InceptionBlock(cin, c)
            blocks += [blk, nn.MaxPool1d(2)]
            cin = blk.out_ch
        self.net = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(cin, n_targets)

    def forward(self, x):                            # (B, T, C)
        h = self.net(x.transpose(1, 2))              # (B, out_ch, T')
        return self.head(self.drop(h.mean(dim=-1)))


def _factory(in_channels, T, model_cfg, n_targets):
    return Inception1d(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "inception1d")
