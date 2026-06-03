"""TDD: calibration regimes."""
import numpy as np
import pytest

from bp_inference import calibration


def test_calibration_free_is_identity():
    y = np.array([[120.0, 80.0], [110.0, 70.0]])
    out = calibration.calibration_free(y)
    assert np.allclose(out, y)
    assert out is not y                       # returns a copy


def test_fit_and_apply_subject_offsets_corrects_bias():
    # Subject A predictions are 5 mmHg below truth on both targets.
    subjects = np.array(["A", "A", "B", "B"])
    y_true = np.array([[120, 80], [124, 84], [100, 60], [108, 66.0]])
    y_pred = y_true - 5.0                       # uniform -5 bias
    offsets = calibration.fit_subject_offsets(y_true, y_pred, subjects)
    assert np.allclose(offsets["A"], [5.0, 5.0])
    assert np.allclose(offsets["B"], [5.0, 5.0])

    corrected = calibration.apply_subject_offsets(y_pred, subjects, offsets)
    assert np.allclose(corrected, y_true)       # bias removed


def test_apply_offsets_leaves_uncalibrated_subject_untouched():
    subjects = np.array(["A", "C"])
    y_pred = np.array([[100.0, 60.0], [130.0, 90.0]])
    offsets = {"A": np.array([2.0, 1.0])}        # no entry for C
    out = calibration.apply_subject_offsets(y_pred, subjects, offsets)
    assert np.allclose(out[0], [102.0, 61.0])
    assert np.allclose(out[1], [130.0, 90.0])


def test_calibrate_dispatch():
    y = np.array([[120.0, 80.0]])
    assert np.allclose(calibration.calibrate("free", y, ["A"]), y)
    with pytest.raises(ValueError):
        calibration.calibrate("bogus", y, ["A"])
    with pytest.raises(ValueError):
        calibration.calibrate("per_subject", y, ["A"])  # missing cal data
