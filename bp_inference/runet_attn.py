"""Seed 2: rU-Net + (optional) STFT front-end + multi-head attention bottleneck.

PPG-only single-channel adaptation of the PulseDB SOTA family (Chen et al. 2025,
IEEE JBHI, DOI 10.1109/JBHI.2024.3483301), which fused U-Net + ResNet with STFT
time-frequency features and multi-head attention. Here: one channel (PPG), two
regression heads (SBP, DBP). When `model.use_stft` is set, a torch.stft magnitude
front-end turns the waveform into a (freq-as-channels, frames) tensor for the 1D
U-Net; otherwise the raw waveform is used. The loop can toggle/scale every knob.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from bp_inference import train


class ResConvBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(cout)
        self.proj = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.proj(x))


class STFTFrontEnd(nn.Module):
    def __init__(self, n_fft: int = 64, hop: int = 16):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))

    @property
    def out_channels(self) -> int:
        return self.n_fft // 2 + 1

    def forward(self, x):                       # x: (B, 1, T)
        spec = torch.stft(x[:, 0], n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, return_complex=True,
                          center=True)
        return spec.abs()                        # (B, F, frames)


class RUNetAttn(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        c = int(model_cfg.get("base_channels", 32))
        heads = int(model_cfg.get("num_heads", 4))
        self.use_stft = bool(model_cfg.get("use_stft", True))
        self.stft = STFTFrontEnd(int(model_cfg.get("n_fft", 64)),
                                 int(model_cfg.get("hop", 16))) if self.use_stft else None
        cin = self.stft.out_channels if self.use_stft else in_channels

        self.enc1 = ResConvBlock1D(cin, c)
        self.enc2 = ResConvBlock1D(c, 2 * c)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ResConvBlock1D(2 * c, 2 * c)
        self.attn = nn.MultiheadAttention(2 * c, heads, batch_first=True)
        self.dec2 = ResConvBlock1D(4 * c, c)     # concat skip (2c) + up (2c)
        self.dec1 = ResConvBlock1D(2 * c, c)     # concat skip (c) + up (c)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(c, c), nn.ReLU(),
            nn.Dropout(float(model_cfg.get("dropout", 0.2))),
            nn.Linear(c, n_targets))

    def forward(self, x):                        # x: (B, T, C)
        x = x.transpose(1, 2)                    # (B, C, T)
        if self.stft is not None:
            x = self.stft(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        tokens = b.transpose(1, 2)               # (B, L, C)
        a, _ = self.attn(tokens, tokens, tokens)
        b = (tokens + a).transpose(1, 2)
        u2 = F.interpolate(b, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.head(d1)


def _factory(in_channels, T, model_cfg, n_targets):
    return RUNetAttn(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "runet_attn")
