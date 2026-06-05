"""New family: dilated Temporal Convolutional Network (TCN), PPG-only regression.

Stacked residual blocks of dilated 1D convolutions (dilation 1, 2, 4, ...) give
an exponentially-growing receptive field over the 1250-sample window without the
cost of attention -- the "extended RF" lever expressed as an architecture.
Lineage: Bai, Kolter & Koltun 2018 (generic conv vs. recurrent nets). Single PPG
channel in (or +VPG/+APG multi-channel ablation), two regression heads (SBP,
DBP). GroupNorm (batch-size independent) keeps a last partial train batch safe.
"""
from pathlib import Path

import torch.nn.functional as F
from torch import nn

from bp_inference import train


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


class TCNBlock(nn.Module):
    """Two dilated convs + residual; causal via a right-side chomp of `pad`."""

    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=self.pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=self.pad, dilation=dilation)
        self.norm1, self.norm2 = _norm(ch), _norm(ch)
        self.drop = nn.Dropout(dropout)

    def _causal(self, conv, x):
        y = conv(x)
        return y[..., :-self.pad] if self.pad else y

    def forward(self, x):                            # (B, C, T)
        h = self.drop(F.relu(self.norm1(self._causal(self.conv1, x))))
        h = self.drop(F.relu(self.norm2(self._causal(self.conv2, h))))
        return x + h


class TCNNet(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        ch = int(model_cfg.get("channels", 48))
        n_blocks = int(model_cfg.get("n_blocks", 5))
        kernel = int(model_cfg.get("kernel_size", 7))
        dropout = float(model_cfg.get("dropout", 0.2))
        self.stem = nn.Conv1d(in_channels, ch, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(ch, kernel, 2 ** i, dropout) for i in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(ch, ch), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ch, n_targets))

    def forward(self, x):                            # (B, T, C)
        h = self.stem(x.transpose(1, 2))             # (B, ch, T)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.mean(dim=-1))             # global average pool over time


def _factory(in_channels, T, model_cfg, n_targets):
    return TCNNet(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "tcn")
