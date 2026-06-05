"""Tests for the deep 1D ResNet family (new architecture), bp_inference.xresnet1d.

The published PPG-only ceiling (Moulaeifard 2025) used XResNet1d101, so this
backbone is the reference the whole project is measured against, it belongs in
the architecture screen. Deep residual 1D conv stages with strided downsampling,
GroupNorm (batch-safe), global pool, two regression heads. Smoke-level synthetic
forward only (ANTIPATTERNS 12).
"""
import torch

from framework import render


def test_xresnet1d_registered_in_engine():
    assert "xresnet1d" in render.FAMILY_ENTRY_POINTS
    assert render.FAMILY_COMPUTE["xresnet1d"] == "gpu"


def test_xresnet1d_forward_two_heads():
    from bp_inference.xresnet1d import _factory
    model = _factory(in_channels=1, T=1250,
                     model_cfg={"base_channels": 16, "n_stages": 3}, n_targets=2)
    out = model(torch.randn(3, 1250, 1))
    assert out.shape == (3, 2) and torch.isfinite(out).all()


def test_xresnet1d_accepts_multichannel():
    from bp_inference.xresnet1d import _factory
    model = _factory(in_channels=3, T=1250, model_cfg={}, n_targets=2)
    out = model(torch.randn(2, 1250, 3))
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def test_xresnet1d_run_from_dir_is_callable():
    import bp_inference.xresnet1d as x
    assert callable(x.run_from_dir)
