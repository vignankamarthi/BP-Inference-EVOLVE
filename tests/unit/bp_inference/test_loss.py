"""Tests for the SBP-weighted regression loss, bp_inference.train._loss_fn.

SBP is the binding AAMI constraint, so a spec may weight the SBP head harder via
`training.loss_weights = [w_sbp, w_dbp]`. Equal weights reproduce the default mean.
"""
import torch

from bp_inference.train import _loss_fn


def test_equal_weights_match_default_mean():
    pred, tgt = torch.randn(8, 2), torch.randn(8, 2)
    default = _loss_fn("mse")(pred, tgt)
    weighted = _loss_fn("mse", weights=[1.0, 1.0])(pred, tgt)
    assert torch.isclose(default, weighted, atol=1e-6)


def test_sbp_weight_emphasizes_sbp_error():
    tgt = torch.zeros(1, 2)
    sbp_err = torch.tensor([[1.0, 0.0]])      # error only in SBP
    dbp_err = torch.tensor([[0.0, 1.0]])      # error only in DBP
    eq = _loss_fn("mse", weights=[1.0, 1.0])
    wt = _loss_fn("mse", weights=[2.0, 1.0])
    assert torch.isclose(eq(sbp_err, tgt), eq(dbp_err, tgt))          # equal: symmetric
    assert wt(sbp_err, tgt) > wt(dbp_err, tgt)                        # weighted: SBP heavier
    assert torch.isclose(wt(sbp_err, tgt) / wt(dbp_err, tgt), torch.tensor(2.0))


def test_weighted_loss_is_differentiable():
    pred = torch.randn(4, 2, requires_grad=True)
    tgt = torch.randn(4, 2)
    _loss_fn("smooth_l1", weights=[3.0, 1.0])(pred, tgt).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
