"""Tests for the bidirectional-GRU family, bp_inference.bigru.

Conv stem downsamples, then a bidirectional GRU over the reduced sequence
(recurrent baseline, the prior DS4400 project's family). Single PPG channel or
+VPG/+APG, two heads. Smoke-level synthetic forward only (ANTIPATTERNS 12)."""
import torch

from framework import render


def test_bigru_registered():
    assert "bigru" in render.FAMILY_ENTRY_POINTS
    assert render.FAMILY_COMPUTE["bigru"] == "gpu"


def test_bigru_forward():
    from bp_inference.bigru import _factory
    m = _factory(in_channels=1, T=1250,
                 model_cfg={"channels": 16, "hidden": 32, "n_layers": 1}, n_targets=2)
    out = m(torch.randn(3, 1250, 1))
    assert out.shape == (3, 2) and torch.isfinite(out).all()


def test_bigru_multichannel():
    from bp_inference.bigru import _factory
    m = _factory(in_channels=3, T=1250, model_cfg={}, n_targets=2)
    out = m(torch.randn(2, 1250, 3))
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def test_bigru_run_from_dir():
    import bp_inference.bigru as mod
    assert callable(mod.run_from_dir)
