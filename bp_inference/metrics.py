"""Regression + clinical-compliance metric suite for cuffless BP.

This is the BP analog of the AI4Pain classification metric suite. It produces
the `best_val_metrics` payload that the cluster writes into result.json and
that `framework.eval.evaluate_program` reads back as the fitness vector.

Clinical standards (ANTIPATTERNS rule 6 -- do not confuse MAE with ME):
  - AAMI: mean error (bias) ME <= 5 mmHg AND standard deviation of error
    SD <= 8 mmHg. Bounds ME and SD, NOT MAE.
  - BHS: cumulative percentage of |error| within 5 / 10 / 15 mmHg, graded
    A / B / C / D.

Error convention: err = y_pred - y_true (so ME > 0 means over-prediction).
SD uses the sample standard deviation (ddof=1), the convention in the cuffless
BP literature for the AAMI SD.
"""
import numpy as np

from bp_inference import TARGETS

AAMI_ME_THRESHOLD = 5.0
AAMI_SD_THRESHOLD = 8.0

# Fixed signed-error bin edges (mmHg) for the novelty signature. Wide enough to
# cover the realistic cuffless error range; fixed so signatures are comparable
# across the whole population.
ERROR_SIGNATURE_EDGES = np.array(
    [-40, -25, -15, -10, -5, -2, 0, 2, 5, 10, 15, 25, 40], dtype=np.float64)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-target scalar metrics. 1-D arrays of equal length."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n = y_true.size
    if n == 0:
        raise ValueError("empty arrays")

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    me = float(np.mean(err))
    sd = float(np.std(err, ddof=1)) if n > 1 else 0.0

    # Pearson r, guarded against constant inputs.
    if n > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = 0.0

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "mae": mae, "rmse": rmse, "me": me, "sd": sd,
        "pearson_r": pearson_r, "r2": r2, "n": int(n),
    }


def aami_pass(me: float, sd: float,
              me_threshold: float = AAMI_ME_THRESHOLD,
              sd_threshold: float = AAMI_SD_THRESHOLD) -> bool:
    """AAMI: |ME| <= 5 AND SD <= 8 mmHg. Native bool (JSON-safe for result.json)."""
    return bool(abs(me) <= me_threshold and sd <= sd_threshold)


def aami_margin(me: float, sd: float,
                me_threshold: float = AAMI_ME_THRESHOLD,
                sd_threshold: float = AAMI_SD_THRESHOLD) -> float:
    """Signed distance inside the AAMI box. Positive => compliant, and larger
    means deeper inside. Negative => non-compliant by that many mmHg on the
    binding constraint. This is the primary fitness scalar (higher is better).
    """
    return float(min(me_threshold - abs(me), sd_threshold - sd))


def bhs_grade(abs_err: np.ndarray) -> dict:
    """British Hypertension Society grade from cumulative |error| percentages.

    Grade A: >=60% within 5, >=85% within 10, >=95% within 15 mmHg.
    Grade B: >=50% / 75% / 90%.  Grade C: >=40% / 65% / 85%.  else D.
    """
    abs_err = np.asarray(abs_err, dtype=np.float64).ravel()
    pct5 = float(np.mean(abs_err <= 5.0) * 100.0)
    pct10 = float(np.mean(abs_err <= 10.0) * 100.0)
    pct15 = float(np.mean(abs_err <= 15.0) * 100.0)

    if pct5 >= 60 and pct10 >= 85 and pct15 >= 95:
        grade = "A"
    elif pct5 >= 50 and pct10 >= 75 and pct15 >= 90:
        grade = "B"
    elif pct5 >= 40 and pct10 >= 65 and pct15 >= 85:
        grade = "C"
    else:
        grade = "D"
    return {"grade": grade, "pct5": pct5, "pct10": pct10, "pct15": pct15}


def error_signature(y_true: np.ndarray, y_pred: np.ndarray,
                    edges: np.ndarray = ERROR_SIGNATURE_EDGES) -> list:
    """Concatenated normalized signed-error histograms (SBP then DBP).

    Replaces the AI4Pain confusion-matrix vector as the novelty descriptor:
    two programs that make qualitatively different error distributions are
    novel relative to each other even at similar headline MAE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sig: list[float] = []
    for i in range(y_true.shape[1]):
        err = y_pred[:, i] - y_true[:, i]
        hist, _ = np.histogram(err, bins=edges)
        total = hist.sum()
        norm = (hist / total) if total > 0 else hist.astype(np.float64)
        sig.extend(norm.tolist())
    return sig


def full_metric_suite(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Full payload for result.json `best_val_metrics`.

    `y_true`, `y_pred`: shape (N, 2), columns = (SBP, DBP) in mmHg.

    Returns per-target sub-dicts plus flat fitness-vector fields consumed by
    the framework Pareto axes and the compliance scalar.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError(f"expected (N, 2) targets, got {y_true.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    out: dict = {}
    for i, target in enumerate(TARGETS):
        m = regression_metrics(y_true[:, i], y_pred[:, i])
        m["aami_pass"] = aami_pass(m["me"], m["sd"])
        m["aami_margin"] = aami_margin(m["me"], m["sd"])
        m["bhs"] = bhs_grade(np.abs(y_pred[:, i] - y_true[:, i]))
        out[target] = m

    # Flat fitness-vector fields (axes are "min" unless noted).
    out["sbp_sd"] = out["sbp"]["sd"]
    out["dbp_sd"] = out["dbp"]["sd"]
    out["sbp_me_abs"] = abs(out["sbp"]["me"])
    out["dbp_me_abs"] = abs(out["dbp"]["me"])
    out["sbp_mae"] = out["sbp"]["mae"]
    out["dbp_mae"] = out["dbp"]["mae"]

    # Primary compliance scalar (max): the worst target's AAMI margin. Driving
    # this above 0 makes BOTH SBP and DBP compliant.
    out["aami_margin"] = float(min(out["sbp"]["aami_margin"],
                                   out["dbp"]["aami_margin"]))
    out["aami_pass_both"] = bool(out["sbp"]["aami_pass"]
                                 and out["dbp"]["aami_pass"])

    # Novelty descriptor.
    out["error_signature"] = error_signature(y_true, y_pred)
    return out
