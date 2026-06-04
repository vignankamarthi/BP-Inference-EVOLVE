"""Tests for the selective-SSM (Mamba-style) seed, bp_inference/mamba_ssm.py.

The block scan h_t = a * h_{t-1} + b_t is a linear *time-invariant* recurrence
(`a` is a per-channel constant, input-independent), so it has an exact parallel
form. `ssm_scan` must reproduce a naive sequential reference within float
tolerance, for any (B, L, D) including L not divisible by the chunk size.

This pins the refactor that replaced the O(L) Python loop over 1250 timesteps
(which wall-killed on the cluster at ~2300 s/epoch, ~20x the conv seeds) with a
chunked scan. Correctness is checked in float64 so the assertion isolates the
algorithm from float32 rounding; gradient/shape checks run in float32.
"""
import pytest
import torch

from bp_inference.mamba_ssm import MambaSSMNet, SSMBlock, ssm_scan


def _reference_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Ground truth: the explicit sequential recurrence (the original loop)."""
    h = torch.zeros(b.shape[0], b.shape[2], dtype=b.dtype)
    outs = []
    for t in range(b.shape[1]):
        h = a * h + b[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


@pytest.mark.parametrize(
    "B,L,D",
    [(2, 1, 4), (3, 5, 8), (2, 64, 6), (2, 65, 6), (4, 130, 16), (2, 1250, 12)],
)
def test_ssm_scan_matches_reference(B, L, D):
    """Parallel scan equals the sequential reference, incl. L not divisible by chunk."""
    torch.manual_seed(0)
    a = torch.sigmoid(torch.randn(D, dtype=torch.float64))   # decay in (0, 1)
    b = torch.randn(B, L, D, dtype=torch.float64)
    got = ssm_scan(a, b)
    ref = _reference_scan(a, b)
    assert got.shape == (B, L, D)
    assert torch.allclose(got, ref, atol=1e-9, rtol=1e-7)


def test_ssm_scan_decay_extremes():
    """Exact for a near 0 (no memory) and a near 1 (long memory)."""
    torch.manual_seed(1)
    L, D = 300, 5
    b = torch.randn(2, L, D, dtype=torch.float64)
    for a_val in (1e-3, 0.999):
        a = torch.full((D,), a_val, dtype=torch.float64)
        assert torch.allclose(ssm_scan(a, b), _reference_scan(a, b), atol=1e-8, rtol=1e-6)


def test_ssm_scan_gradients_flow():
    """Scan is differentiable w.r.t. both the decay and the input term."""
    a = torch.rand(4, requires_grad=True)        # leaf decay in (0, 1)
    b = torch.randn(2, 16, 4, requires_grad=True)
    ssm_scan(a, b).sum().backward()
    assert b.grad is not None and torch.isfinite(b.grad).all()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_block_forward_shape_and_finite():
    torch.manual_seed(0)
    blk = SSMBlock(dim=8)
    x = torch.randn(3, 50, 8)
    y = blk(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_net_forward_two_heads():
    torch.manual_seed(0)
    net = MambaSSMNet(in_channels=1, T=1250, model_cfg={"dim": 16, "n_layers": 2}, n_targets=2)
    out = net(torch.randn(2, 1250, 1))
    assert out.shape == (2, 2) and torch.isfinite(out).all()
