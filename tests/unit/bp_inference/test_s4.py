"""Tests for the structured-state-space (S4-lite) family, bp_inference.s4.

A learnable long depthwise causal convolution kernel per channel (the "S4 as a
global convolution" view), distinct from the mamba input-gated recurrence.
Single PPG channel or +VPG/+APG, two heads. Smoke-level synthetic forward only."""
import torch

from framework import render


def test_s4_registered():
    assert "s4" in render.FAMILY_ENTRY_POINTS
    assert render.FAMILY_COMPUTE["s4"] == "gpu"


def test_s4_forward():
    from bp_inference.s4 import _factory
    m = _factory(in_channels=1, T=1250, model_cfg={"channels": 32, "kernel_len": 64}, n_targets=2)
    out = m(torch.randn(3, 1250, 1))
    assert out.shape == (3, 2) and torch.isfinite(out).all()


def test_s4_multichannel():
    from bp_inference.s4 import _factory
    m = _factory(in_channels=3, T=1250, model_cfg={}, n_targets=2)
    out = m(torch.randn(2, 1250, 3))
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def test_s4_run_from_dir():
    import bp_inference.s4 as mod
    assert callable(mod.run_from_dir)
