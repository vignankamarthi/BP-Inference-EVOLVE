"""Seed 1: ROCKET-family random convolutional kernels + RidgeCV regressor.

The given seed (Vignan). Lineage: Dempster, Petitjean & Webb 2020 (ROCKET) and
Dempster, Schmidt & Webb 2021 (MINIROCKET). This is a compact, dependency-free
ROCKET-family transform (random dilated kernels -> PPV + global-max pooling)
feeding a multi-output RidgeCV regressor. No SGD, no torch: the fast,
neural-adjacent counter-baseline for the otherwise-deep seed pool. The loop
mutates kernel count, pooling, and the regressor from here.
"""
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from bp_inference import train
from bp_inference.metrics import full_metric_suite


class RocketTransform:
    """Random dilated convolutional kernels with PPV + max pooling.

    Two features per kernel: PPV (proportion of positive values above a random
    bias) and the global max of the dilated convolution. PPG is single-channel.
    """

    def __init__(self, num_kernels: int = 1000, kernel_length: int = 9,
                 seed: int = 42):
        self.num_kernels = int(num_kernels)
        self.kernel_length = int(kernel_length)
        self.seed = int(seed)

    def fit(self, X: np.ndarray) -> "RocketTransform":
        rng = np.random.default_rng(self.seed)
        T = X.shape[1]
        L = self.kernel_length
        w = rng.standard_normal((self.num_kernels, L))
        self.weights = (w - w.mean(axis=1, keepdims=True)).astype(np.float32)
        # dilations as powers of two whose receptive field fits in T
        max_exp = max(0, int(np.floor(np.log2((T - 1) / (L - 1))))) if T > L else 0
        self.dilations = (2 ** rng.integers(0, max_exp + 1,
                                            size=self.num_kernels)).astype(int)
        self.biases = rng.uniform(-1.0, 1.0, size=self.num_kernels).astype(np.float32)
        self._T = T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float32)[:, :, 0]      # (N, T) PPG only
        N, T = x.shape
        feats = np.empty((N, 2 * self.num_kernels), dtype=np.float32)
        L = self.kernel_length
        for k in range(self.num_kernels):
            d = int(self.dilations[k])
            span = (L - 1) * d + 1
            if span > T:                                  # dilation too large
                feats[:, 2 * k] = 0.0
                feats[:, 2 * k + 1] = 0.0
                continue
            windows = sliding_window_view(x, span, axis=1)[:, :, ::d]   # (N, T-span+1, L)
            conv = windows @ self.weights[k]              # (N, n_out)
            feats[:, 2 * k] = (conv > self.biases[k]).mean(axis=1)
            feats[:, 2 * k + 1] = conv.max(axis=1)
        return feats


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    """Fit the ROCKET transform + RidgeCV regressor, write result.json."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    fe = spec.get("feature_extraction", {}) or {}
    mdl = spec.get("model", {}) or {}
    seed = int(spec.get("training", {}).get("seed", 42))
    calib_cfg = spec.get("calibration", {"mode": "free"})

    (Xtr, ytr, _), (Xv, yv, sv) = train._load_train_val(data_root, spec)

    rocket = RocketTransform(
        num_kernels=int(fe.get("num_kernels", 1000)),
        kernel_length=int(fe.get("kernel_length", 9)),
        seed=seed,
    ).fit(Xtr)
    Ftr, Fv = rocket.transform(Xtr), rocket.transform(Xv)

    scaler = StandardScaler().fit(Ftr)
    Ftr_s, Fv_s = scaler.transform(Ftr), scaler.transform(Fv)
    t_mean, t_std = train.standardize_targets(ytr)
    ytr_std = (ytr - t_mean) / t_std

    alphas = mdl.get("alphas", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    reg = RidgeCV(alphas=alphas).fit(Ftr_s, ytr_std)
    pred = reg.predict(Fv_s) * t_std + t_mean

    yv_eval, vp_eval = train.evaluate_regime(yv, pred, sv, calib_cfg, seed=seed)
    metrics = full_metric_suite(yv_eval, vp_eval)

    result = {
        "name": spec.get("name", "minirocket"),
        "best_val_metrics": metrics,
        "final_val_metrics": metrics,
        "history": [{"epoch": 0, "val_aami_margin": metrics["aami_margin"],
                     "val_sbp_mae": metrics["sbp_mae"],
                     "val_dbp_mae": metrics["dbp_mae"]}],
        "param_count": int(Ftr.shape[1]),       # transform feature dimension
        "train_seconds": 0.0,
        "inference_seconds": 0.0,
        "generalization_gap": 0.0,
        "device": "cpu",
        "spec": spec,
    }
    train._atomic_write_json(run_dir / "result.json", result)
    return result
