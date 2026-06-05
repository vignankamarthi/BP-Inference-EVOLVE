"""Tests for PPG-derivative input channels (VPG/APG), bp_inference.train.

VPG (velocity plethysmogram) = d/dt PPG; APG (acceleration plethysmogram) =
d2/dt2 PPG. Both are computed from the single PPG channel (still PPG-only source,
ANTIPATTERNS rule 2); the multi-channel result is a LABELED ABLATION arm, never
the hero claim. Diffs are length-preserved so T is unchanged.
"""
import numpy as np

from bp_inference.train import add_derivative_channels


def test_raw_passthrough_when_no_derivatives():
    X = np.random.RandomState(0).randn(4, 10, 1).astype(np.float32)
    out = add_derivative_channels(X, [])
    assert out.shape == (4, 10, 1)
    np.testing.assert_array_equal(out, X)


def test_vpg_is_length_preserved_first_diff():
    X = np.arange(10, dtype=np.float32).reshape(1, 10, 1)   # ppg = 0..9 (slope 1)
    out = add_derivative_channels(X, ["vpg"])
    assert out.shape == (1, 10, 2)                          # ppg + vpg
    np.testing.assert_array_equal(out[0, :, 0], X[0, :, 0])  # ppg channel preserved
    vpg = out[0, :, 1]
    assert vpg[0] == 0.0                                    # prepend edge
    assert np.allclose(vpg[1:], 1.0)                        # constant slope


def test_apg_is_second_diff():
    X = (np.arange(10, dtype=np.float32) ** 2).reshape(1, 10, 1)  # ppg = t^2
    out = add_derivative_channels(X, ["vpg", "apg"])
    assert out.shape == (1, 10, 3)                          # ppg + vpg + apg
    apg = out[0, :, 2]
    assert np.allclose(apg[2:], 2.0, atol=1e-4)             # 2nd diff of t^2 = 2


def test_apg_without_vpg_appends_only_apg():
    X = np.random.RandomState(1).randn(2, 8, 1).astype(np.float32)
    out = add_derivative_channels(X, ["apg"])
    assert out.shape == (2, 8, 2)                           # ppg + apg (vpg internal)


def test_dtype_preserved():
    X = np.random.RandomState(2).randn(3, 6, 1).astype(np.float32)
    assert add_derivative_channels(X, ["vpg", "apg"]).dtype == np.float32
