"""New family: Transformer encoder over the PPG window, PPG-only regression.

A conv stem downsamples the 1250-sample window (keeping attention O(T'^2)
tractable), then a stack of self-attention encoder layers with fixed sinusoidal
positions, global-pooled to two regression heads. Single PPG channel in, or
+VPG/+APG (multi-channel ablation).
"""
import math
from pathlib import Path

import torch
from torch import nn

from bp_inference import train


def _sinusoidal_pos(length: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(length, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Transformer1d(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        d = int(model_cfg.get("d_model", 64))
        nhead = int(model_cfg.get("n_heads", 4))
        nlayers = int(model_cfg.get("n_layers", 3))
        ds = int(model_cfg.get("downsample", 8))
        dropout = float(model_cfg.get("dropout", 0.2))
        self.stem = nn.Conv1d(in_channels, d, kernel_size=2 * ds + 1, stride=ds, padding=ds)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=2 * d,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_targets))

    def forward(self, x):                            # (B, T, C)
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)         # (B, T', d)
        h = h + _sinusoidal_pos(h.shape[1], h.shape[2], h.device).unsqueeze(0)
        h = self.enc(h)
        return self.head(h.mean(dim=1))


def _factory(in_channels, T, model_cfg, n_targets):
    return Transformer1d(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "transformer1d")
