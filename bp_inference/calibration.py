"""Calibration regimes (ANTIPATTERNS rule 5).

Two regimes, never blended:

  - calibration-free: predictions used as-is. No subject-specific adaptation.
  - per-subject (calibration-based): a per-subject additive offset is estimated
    from that subject's designated CALIBRATION segments (true - pred), then added
    to the subject's evaluation predictions. The offset never sees evaluation
    labels. This is the classic mean/one-point cuffless-BP calibration.
"""
import numpy as np


def calibration_free(y_pred: np.ndarray) -> np.ndarray:
    """Identity. No adaptation. Kept explicit so call sites name the regime."""
    return np.asarray(y_pred, dtype=np.float64).copy()


def fit_subject_offsets(y_true_cal: np.ndarray, y_pred_cal: np.ndarray,
                        subjects_cal) -> dict:
    """Per-subject additive offset = mean(true - pred) over calibration segments.

    Returns {subject_id: np.ndarray shape (2,)} for (SBP, DBP).
    """
    y_true_cal = np.asarray(y_true_cal, dtype=np.float64)
    y_pred_cal = np.asarray(y_pred_cal, dtype=np.float64)
    if y_true_cal.shape != y_pred_cal.shape:
        raise ValueError("calibration true/pred shape mismatch")
    subjects_cal = np.asarray(subjects_cal).ravel()

    offsets: dict = {}
    for subj in np.unique(subjects_cal):
        m = subjects_cal == subj
        residual = y_true_cal[m] - y_pred_cal[m]   # what to add to predictions
        offsets[subj.item() if hasattr(subj, "item") else subj] = \
            residual.mean(axis=0)
    return offsets


def apply_subject_offsets(y_pred: np.ndarray, subjects,
                          offsets: dict) -> np.ndarray:
    """Add each subject's offset to its predictions. Subjects with no calibration
    offset are left uncorrected (offset 0), which is logged by the caller.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).copy()
    subjects = np.asarray(subjects).ravel()
    n_targets = y_pred.shape[1]
    for i, subj in enumerate(subjects):
        key = subj.item() if hasattr(subj, "item") else subj
        off = offsets.get(key)
        if off is not None:
            y_pred[i] += np.asarray(off, dtype=np.float64)[:n_targets]
    return y_pred


def calibrate(regime: str, y_pred_eval: np.ndarray, subjects_eval,
              y_true_cal: np.ndarray | None = None,
              y_pred_cal: np.ndarray | None = None,
              subjects_cal=None) -> np.ndarray:
    """Dispatch on the `calibration` gene. `regime` in {'free', 'per_subject'}."""
    if regime == "free":
        return calibration_free(y_pred_eval)
    if regime == "per_subject":
        if y_true_cal is None or y_pred_cal is None or subjects_cal is None:
            raise ValueError("per_subject calibration needs calibration data")
        offsets = fit_subject_offsets(y_true_cal, y_pred_cal, subjects_cal)
        return apply_subject_offsets(y_pred_eval, subjects_eval, offsets)
    raise ValueError(f"unknown calibration regime {regime!r}")
