"""Shared pytest fixtures.

Per ANTIPATTERNS rule 21 (TDD): every public function gets a test in
tests/test_<module>.py BEFORE implementation. Xfail with strict=True and
raises=NotImplementedError is the red-state pattern: passing today (because
NotImplementedError is raised), failing automatically once you implement
the function and forget to remove the xfail marker.
"""
import pytest
from pathlib import Path


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Temporary SQLite path for ledger tests."""
    return tmp_path / "test_experiments.db"


@pytest.fixture
def sample_program_spec() -> dict:
    """Minimal placeholder program spec for render and constraints tests."""
    return {
        "preprocessing": {"normalize": "per_channel_zscore", "window_seconds": 10.0},
        "feature_extraction": None,
        "model": {"family": "runet_attn", "base_channels": 32, "num_heads": 4},
        "training": {"loss": "smooth_l1", "optimizer": "adamw", "lr": 5e-4,
                     "epochs": 10, "seed": 42},
        "calibration": {"mode": "free"},
        "data": {"signals": ["ppg"], "val_fraction": 0.2},
        "decode": {"strategy": "identity"},
    }


@pytest.fixture
def sample_fitness_vector() -> dict:
    """Minimal placeholder fitness vector for fitness and ledger tests."""
    return {
        "balanced_acc": 0.55,
        "macro_f1": 0.52,
        "per_class_pr": {"NP": (0.6, 0.5), "HP": (0.5, 0.55), "AP": (0.5, 0.55)},
        "confusion_3x3": [[10, 3, 2], [4, 9, 2], [3, 3, 9]],
        "auc_ovr": 0.61,
        "ece": 0.08,
        "param_count": 50_000,
        "train_seconds": 180.0,
        "inference_seconds": 1.5,
        "generalization_gap": 0.05,
    }
