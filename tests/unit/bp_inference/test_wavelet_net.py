"""Tests for the multi-resolution (wavelet-style) family, bp_inference.wavelet_net.

A cascade of strided convolutions produces coarser-and-coarser subbands; each
level is global-pooled and concatenated, the frequency/scale lever expressed as
an architecture. Single PPG channel or +VPG/+APG, two heads. Smoke-level only."""
import torch

from framework import render


def test_wavelet_net_registered():
    assert "wavelet_net" in render.FAMILY_ENTRY_POINTS
    assert render.FAMILY_COMPUTE["wavelet_net"] == "gpu"


def test_wavelet_net_forward():
    from bp_inference.wavelet_net import _factory
    m = _factory(in_channels=1, T=1250, model_cfg={"channels": 16, "levels": 3}, n_targets=2)
    out = m(torch.randn(3, 1250, 1))
    assert out.shape == (3, 2) and torch.isfinite(out).all()


def test_wavelet_net_multichannel():
    from bp_inference.wavelet_net import _factory
    m = _factory(in_channels=3, T=1250, model_cfg={}, n_targets=2)
    out = m(torch.randn(2, 1250, 3))
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def test_wavelet_net_run_from_dir():
    import bp_inference.wavelet_net as mod
    assert callable(mod.run_from_dir)
