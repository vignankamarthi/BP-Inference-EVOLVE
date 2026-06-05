"""Tests for the Inception-1D family, bp_inference.inception1d.

Parallel conv branches at multiple kernel sizes (multi-scale features at one
resolution) concatenated per block, stacked with downsampling. Single PPG
channel or +VPG/+APG, two heads. Smoke-level synthetic forward only."""
import torch

from framework import render


def test_inception1d_registered():
    assert "inception1d" in render.FAMILY_ENTRY_POINTS
    assert render.FAMILY_COMPUTE["inception1d"] == "gpu"


def test_inception1d_forward():
    from bp_inference.inception1d import _factory
    m = _factory(in_channels=1, T=1250, model_cfg={"channels": 12, "n_blocks": 2}, n_targets=2)
    out = m(torch.randn(3, 1250, 1))
    assert out.shape == (3, 2) and torch.isfinite(out).all()


def test_inception1d_multichannel():
    from bp_inference.inception1d import _factory
    m = _factory(in_channels=3, T=1250, model_cfg={}, n_targets=2)
    out = m(torch.randn(2, 1250, 3))
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def test_inception1d_run_from_dir():
    import bp_inference.inception1d as mod
    assert callable(mod.run_from_dir)
