"""Tests for the Transformer encoder family, bp_inference.transformer1d.

Conv stem downsamples the 1250-sample window, then a stack of self-attention
encoder layers with sinusoidal positions. Single PPG channel or +VPG/+APG, two
heads. Smoke-level synthetic forward only (ANTIPATTERNS 12)."""
import torch

from framework import render


def test_transformer1d_registered():
    assert "transformer1d" in render.FAMILY_ENTRY_POINTS
    assert render.FAMILY_COMPUTE["transformer1d"] == "gpu"


def test_transformer1d_forward():
    from bp_inference.transformer1d import _factory
    m = _factory(in_channels=1, T=1250,
                 model_cfg={"d_model": 32, "n_heads": 4, "n_layers": 2, "downsample": 8},
                 n_targets=2)
    out = m(torch.randn(3, 1250, 1))
    assert out.shape == (3, 2) and torch.isfinite(out).all()


def test_transformer1d_multichannel():
    from bp_inference.transformer1d import _factory
    m = _factory(in_channels=3, T=1250, model_cfg={}, n_targets=2)
    out = m(torch.randn(2, 1250, 3))
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def test_transformer1d_run_from_dir():
    import bp_inference.transformer1d as mod
    assert callable(mod.run_from_dir)
