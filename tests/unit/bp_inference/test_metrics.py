"""TDD: regression + AAMI/BHS compliance metrics. Hand-computed expectations."""
import numpy as np
import pytest

from bp_inference import metrics


def test_regression_metrics_known_values():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 1.0, -1.0])
    m = metrics.regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["me"] == pytest.approx(0.0)
    # sample SD (ddof=1): sqrt(4/3)
    assert m["sd"] == pytest.approx(np.sqrt(4.0 / 3.0))
    assert m["n"] == 4
    # constant y_true -> pearson and r2 fall back to 0
    assert m["pearson_r"] == pytest.approx(0.0)


def test_regression_metrics_shape_guard():
    with pytest.raises(ValueError):
        metrics.regression_metrics(np.zeros(3), np.zeros(4))


def test_aami_pass_boundaries():
    assert metrics.aami_pass(0.0, np.sqrt(4.0 / 3.0)) is True
    assert metrics.aami_pass(5.0, 8.0) is True          # inclusive boundary
    assert metrics.aami_pass(-5.0, 8.0) is True          # bias is absolute
    assert metrics.aami_pass(6.0, 1.0) is False          # ME fails
    assert metrics.aami_pass(0.0, 9.0) is False          # SD fails


def test_aami_margin_sign():
    # deep inside the box
    assert metrics.aami_margin(0.0, 1.0) == pytest.approx(5.0)
    # SD binds and is violated -> negative margin
    assert metrics.aami_margin(3.0, 9.0) == pytest.approx(-1.0)


def test_bhs_grade_A_and_D():
    a = metrics.bhs_grade(np.full(10, 3.0))               # all within 5 mmHg
    assert a["grade"] == "A"
    assert a["pct5"] == pytest.approx(100.0)
    # 60% within 5/10, 100% within 15 -> fails A/B/C on the 10 mmHg gate
    d = metrics.bhs_grade(np.array([3, 3, 3, 3, 3, 3, 12, 12, 12, 12.0]))
    assert d["grade"] == "D"
    assert d["pct10"] == pytest.approx(60.0)


def test_full_metric_suite_payload():
    y_true = np.array([[120, 80], [110, 70], [130, 90], [100, 60.0]])
    y_pred = np.array([[122, 82], [108, 68], [134, 86], [96, 64.0]])
    out = metrics.full_metric_suite(y_true, y_pred)

    assert out["sbp"]["mae"] == pytest.approx(3.0)
    assert out["dbp"]["mae"] == pytest.approx(3.0)
    assert out["sbp"]["me"] == pytest.approx(0.0)
    assert out["aami_pass_both"] is True
    assert out["sbp"]["bhs"]["grade"] == "A"
    # flat fitness fields present
    for key in ("sbp_sd", "dbp_sd", "sbp_me_abs", "dbp_me_abs",
                "sbp_mae", "dbp_mae", "aami_margin"):
        assert key in out
    # worst-target margin = overall margin
    assert out["aami_margin"] == pytest.approx(
        min(out["sbp"]["aami_margin"], out["dbp"]["aami_margin"]))


def test_error_signature_shape_and_normalization():
    rng = np.random.default_rng(0)
    y_true = rng.normal(120, 15, size=(200, 2))
    y_pred = y_true + rng.normal(0, 5, size=(200, 2))
    sig = metrics.error_signature(y_true, y_pred)
    n_bins = len(metrics.ERROR_SIGNATURE_EDGES) - 1
    assert len(sig) == 2 * n_bins
    # each target's histogram normalizes to ~1 (all errors land inside edges)
    assert sum(sig[:n_bins]) == pytest.approx(1.0, abs=1e-9)
    assert sum(sig[n_bins:]) == pytest.approx(1.0, abs=1e-9)


def test_full_metric_suite_shape_guard():
    with pytest.raises(ValueError):
        metrics.full_metric_suite(np.zeros((4, 3)), np.zeros((4, 3)))
