"""TDD: seed model families + the regression train harness.

MiniRocket is torch-free. The neural seeds and the full training path require
torch (skipped if absent). All use synthetic data (TDD rule 12).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bp_inference import data, minirocket, train

TORCH = importlib.util.find_spec("torch") is not None


def _write_synth_train(tmp_path: Path, n=96, T=256, seed=1):
    X, y, subj = data.make_synthetic_split(n=n, T=T, seed=seed)
    np.savez(tmp_path / "train.npz", X=X, sbp=y[:, 0], dbp=y[:, 1], subjects=subj)


def _spec(family: str, **over) -> dict:
    spec = {
        "name": f"test_{family}",
        "data": {"signals": ["ppg"], "val_fraction": 0.25},
        "feature_extraction": None,
        "model": {"family": family, "base_channels": 8, "dim": 16, "num_heads": 2},
        "training": {"loss": "smooth_l1", "optimizer": "adam", "lr": 1e-3,
                     "epochs": 1, "batch_size": 16, "seed": 3},
        "calibration": {"mode": "free"},
        "decode": {"strategy": "identity"},
    }
    spec.update(over)
    return spec


def test_rocket_transform_shapes():
    X, _, _ = data.make_synthetic_split(n=10, T=128, seed=0)
    rk = minirocket.RocketTransform(num_kernels=50, seed=0).fit(X)
    feats = rk.transform(X)
    assert feats.shape == (10, 100)             # 2 features per kernel
    assert np.isfinite(feats).all()


def test_minirocket_run_from_dir(tmp_path):
    _write_synth_train(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _spec("ridge_regressor_cv")
    spec["feature_extraction"] = {"family": "minirocket", "num_kernels": 80}
    spec["model"] = {"family": "ridge_regressor_cv", "alphas": [0.1, 1.0, 10.0]}
    (run_dir / "spec.json").write_text(json.dumps(spec))

    res = minirocket.run_from_dir(run_dir, tmp_path)
    assert (run_dir / "result.json").exists()
    bvm = res["best_val_metrics"]
    assert "aami_margin" in bvm
    assert bvm["sbp"]["n"] > 0 and bvm["dbp"]["n"] > 0
    assert isinstance(bvm["aami_pass_both"], bool)


def test_evaluate_regime_per_subject_holds_out_eval():
    # 2 subjects, biased predictions; per-subject calibration should shrink bias.
    subjects = np.array(["A"] * 10 + ["B"] * 10)
    y_true = np.random.default_rng(0).uniform(80, 160, size=(20, 2))
    y_pred = y_true + np.array([7.0, -4.0])     # constant per-target bias
    yt, yp = train.evaluate_regime(y_true, y_pred, subjects,
                                   {"mode": "per_subject", "cal_fraction": 0.4},
                                   seed=0)
    # eval set is a strict subset (calibration segments held out)
    assert len(yt) < len(y_true)
    # bias on eval predictions is near zero after offset correction
    assert abs((yp - yt).mean(axis=0)).max() < 1.0


@pytest.mark.skipif(not TORCH, reason="torch not installed")
@pytest.mark.parametrize("family", ["runet_attn", "resunet_sa", "mamba_ssm"])
def test_nn_forward_shapes(family):
    import torch

    from bp_inference import mamba_ssm, resunet_sa, runet_attn
    factories = {"runet_attn": runet_attn._factory,
                 "resunet_sa": resunet_sa._factory,
                 "mamba_ssm": mamba_ssm._factory}
    x = torch.randn(4, 256, 1)
    model = factories[family](in_channels=1, T=256,
                              model_cfg={"base_channels": 8, "dim": 16,
                                         "num_heads": 2}, n_targets=2)
    out = model(x)
    assert out.shape == (4, 2)


@pytest.mark.skipif(not TORCH, reason="torch not installed")
def test_runet_train_end_to_end(tmp_path):
    """Full harness: load -> zscore -> train 1 epoch -> result.json contract."""
    from bp_inference import runet_attn
    _write_synth_train(tmp_path, n=64, T=256, seed=2)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _spec("runet_attn")
    spec["model"]["use_stft"] = False           # raw-waveform path for speed
    (run_dir / "spec.json").write_text(json.dumps(spec))

    res = runet_attn.run_from_dir(run_dir, tmp_path)
    assert (run_dir / "result.json").exists()
    for key in ("best_val_metrics", "history", "param_count",
                "train_seconds", "generalization_gap", "spec"):
        assert key in res
    assert res["param_count"] > 0
    assert len(res["history"]) == 1
