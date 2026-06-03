"""TDD: PPG-only data access."""
import numpy as np
import pytest

from bp_inference import data


def test_enforce_ppg_only_accepts_aliases():
    assert data.enforce_ppg_only(["ppg"]) == ("ppg",)
    assert data.enforce_ppg_only(["BVP"]) == ("bvp",)


def test_enforce_ppg_only_rejects_ecg():
    with pytest.raises(ValueError):
        data.enforce_ppg_only(["ppg", "ecg"])
    with pytest.raises(ValueError):
        data.enforce_ppg_only([])


def test_make_synthetic_split_shapes():
    X, y, subjects = data.make_synthetic_split(n=32, T=128, seed=3)
    assert X.shape == (32, 128, 1)
    assert y.shape == (32, 2)
    assert subjects.shape == (32,)
    assert X.dtype == np.float32
    # diastolic below systolic on average (sanity of the synthetic generator)
    assert y[:, 1].mean() < y[:, 0].mean()


def test_load_split_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_split(tmp_path, "train")


def test_load_split_round_trip(tmp_path):
    X = np.random.randn(10, 64, 1).astype(np.float32)
    sbp = np.full(10, 120.0, dtype=np.float32)
    dbp = np.full(10, 80.0, dtype=np.float32)
    subjects = np.arange(10)
    np.savez(tmp_path / "train.npz", X=X, sbp=sbp, dbp=dbp, subjects=subjects)

    Xo, yo, so = data.load_split(tmp_path, "train")
    assert Xo.shape == (10, 64, 1)
    assert yo.shape == (10, 2)
    assert np.allclose(yo[:, 0], 120.0) and np.allclose(yo[:, 1], 80.0)
    assert len(so) == 10


def test_load_split_rejects_bad_split(tmp_path):
    with pytest.raises(ValueError):
        data.load_split(tmp_path, "not_a_split")
