"""Tests for the dilated TCN family (new architecture), bp_inference.tcn.

Stacked dilated-conv blocks with an exponentially-growing receptive field over
the 1250-sample PPG window (the "extended RF" lever as an architecture). Single
channel, or +VPG/+APG (multi-channel ablation), two regression heads.
Smoke-level synthetic forward only (ANTIPATTERNS 12).
"""
import torch

from framework import render


def test_tcn_registered_in_engine():
    assert "tcn" in render.FAMILY_ENTRY_POINTS
    assert render.FAMILY_COMPUTE["tcn"] == "gpu"


def test_tcn_forward_two_heads():
    from bp_inference.tcn import _factory
    model = _factory(in_channels=1, T=1250,
                     model_cfg={"channels": 32, "n_blocks": 4}, n_targets=2)
    out = model(torch.randn(3, 1250, 1))
    assert out.shape == (3, 2) and torch.isfinite(out).all()


def test_tcn_accepts_multichannel_for_vpg_apg():
    from bp_inference.tcn import _factory
    model = _factory(in_channels=3, T=1250, model_cfg={}, n_targets=2)  # ppg+vpg+apg
    out = model(torch.randn(2, 1250, 3))
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def test_tcn_run_from_dir_is_callable():
    import bp_inference.tcn as tcn
    assert callable(tcn.run_from_dir)
