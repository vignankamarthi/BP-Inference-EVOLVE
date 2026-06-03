"""Seed 3: self-attention ResUNet (PPG-only).

PPG-only adaptation of the PulseDB self-attention ResUNet family (2025, IEEE
Sensors Journal), which reached calibration-free AAMI/IEEE-1708/BHS on PulseDB
with PPG+ECG. Here the input is the single PPG channel. Distinct from seed 2: a
3-level residual encoder/decoder with squeeze-excite residual blocks and a
self-attention block at the bottleneck, no STFT front-end (operates on the raw
waveform). Two regression heads (SBP, DBP).
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bp_inference import train


class SEResBlock1D(nn.Module):
    """Residual block with a squeeze-excite channel-attention gate."""

    def __init__(self, cin: int, cout: int, k: int = 7, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(cout)
        self.proj = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        hidden = max(1, cout // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(cout, hidden, 1), nn.ReLU(),
            nn.Conv1d(hidden, cout, 1), nn.Sigmoid())

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h * self.se(h)                       # channel recalibration
        return F.relu(h + self.proj(x))


class SelfAttention1D(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):                        # (B, C, L)
        t = x.transpose(1, 2)
        a, _ = self.attn(t, t, t)
        return self.norm(t + a).transpose(1, 2)


class ResUNetSA(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        c = int(model_cfg.get("base_channels", 24))
        heads = int(model_cfg.get("num_heads", 4))
        self.pool = nn.MaxPool1d(2)
        self.e1 = SEResBlock1D(in_channels, c)
        self.e2 = SEResBlock1D(c, 2 * c)
        self.e3 = SEResBlock1D(2 * c, 4 * c)
        self.bottleneck = SEResBlock1D(4 * c, 4 * c)
        self.sa = SelfAttention1D(4 * c, heads)
        self.d3 = SEResBlock1D(8 * c, 2 * c)
        self.d2 = SEResBlock1D(4 * c, c)
        self.d1 = SEResBlock1D(2 * c, c)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(c, c), nn.ReLU(),
            nn.Dropout(float(model_cfg.get("dropout", 0.2))),
            nn.Linear(c, n_targets))

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)

    def forward(self, x):                        # (B, T, C)
        x = x.transpose(1, 2)
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.sa(self.bottleneck(self.pool(e3)))
        d3 = self.d3(torch.cat([self._up(b, e3), e3], dim=1))
        d2 = self.d2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.d1(torch.cat([self._up(d2, e1), e1], dim=1))
        return self.head(d1)


def _factory(in_channels, T, model_cfg, n_targets):
    return ResUNetSA(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "resunet_sa")
