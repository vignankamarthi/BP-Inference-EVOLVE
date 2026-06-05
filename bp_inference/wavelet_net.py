"""New family: multi-resolution (wavelet-style) cascade, PPG-only regression.

A conv stem then a cascade of strided convolutions producing coarser-and-coarser
subbands; every level is global-pooled and concatenated, so the head sees the
pulse at multiple scales at once (the frequency/scale lever as an architecture).
Single PPG channel in, or +VPG/+APG (multi-channel ablation), two heads.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bp_inference import train


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class WaveletNet(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        c = int(model_cfg.get("channels", 32))
        levels = int(model_cfg.get("levels", 4))
        dropout = float(model_cfg.get("dropout", 0.2))
        self.stem = nn.Conv1d(in_channels, c, 7, padding=3)
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(c, c, 4, stride=2, padding=1), _norm(c), nn.ReLU())
            for _ in range(levels)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c * (levels + 1), n_targets)

    def forward(self, x):                            # (B, T, C)
        h = F.relu(self.stem(x.transpose(1, 2)))     # (B, c, T)
        feats = [h.mean(dim=-1)]
        for down in self.downs:
            h = down(h)
            feats.append(h.mean(dim=-1))
        return self.head(self.drop(torch.cat(feats, dim=-1)))


def _factory(in_channels, T, model_cfg, n_targets):
    return WaveletNet(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "wavelet_net")
