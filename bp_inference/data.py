"""PulseDB data access -- PPG / BVP only.

PulseDB segments are fixed-length (10 s at 125 Hz = 1250 samples), so unlike the
AI4Pain variable-length trials there is no padding step. The on-disk cache is one
`.npz` per split, written on the cluster from the PulseDB pipeline (reuse the
prior project's export under `../Blood-Pressure-Inference-with-BVP/`):

    <data_root>/<split>.npz
        X         float32 (N, T, 1)   PPG channel only
        sbp       float32 (N,)        systolic, mmHg
        dbp       float32 (N,)        diastolic, mmHg
        subjects  (N,)                subject id per segment

ANTIPATTERNS rule 2: PPG only. `enforce_ppg_only` rejects any non-PPG signal so a
spec can never smuggle ECG into the model input.
"""
from pathlib import Path

import numpy as np

# Accepted aliases for the single allowed channel.
PPG_ALIASES = frozenset({"ppg", "bvp", "pleth", "ppg_record"})

VALID_SPLITS = frozenset({
    "train", "validation", "calfree", "calbased", "aami_cal", "aami_test",
})


def enforce_ppg_only(signals) -> tuple[str, ...]:
    """Return the normalized signal tuple or raise if anything non-PPG appears."""
    sigs = tuple(str(s).lower() for s in signals)
    if not sigs:
        raise ValueError("no signals requested; PPG is required")
    bad = [s for s in sigs if s not in PPG_ALIASES]
    if bad:
        raise ValueError(
            f"ANTIPATTERNS rule 2: PPG/BVP only. Rejected signals: {bad}. "
            f"Allowed: {sorted(PPG_ALIASES)}")
    return sigs


def load_split(data_root: Path, split: str,
               signals=("ppg",)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one cached split. Returns (X, y, subjects).

    X:        float32 (N, T, 1)
    y:        float32 (N, 2)  columns (SBP, DBP)
    subjects: (N,)
    """
    enforce_ppg_only(signals)
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}; valid: {sorted(VALID_SPLITS)}")

    path = Path(data_root) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"split cache not found: {path}. Build it on the cluster from the "
            f"PulseDB pipeline (see PLAN.md Phase 1).")

    with np.load(path, allow_pickle=True) as npz:
        X = np.asarray(npz["X"], dtype=np.float32)
        sbp = np.asarray(npz["sbp"], dtype=np.float32).ravel()
        dbp = np.asarray(npz["dbp"], dtype=np.float32).ravel()
        subjects = np.asarray(npz["subjects"]).ravel()

    if X.ndim == 2:                       # (N, T) -> (N, T, 1)
        X = X[:, :, None]
    if X.shape[-1] != 1:
        raise ValueError(
            f"PPG-only expects 1 channel, got {X.shape[-1]} in {path}")
    y = np.stack([sbp, dbp], axis=1).astype(np.float32)
    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(f"length mismatch in {path}")
    return X, y, subjects


def make_synthetic_split(n: int = 64, T: int = 256, seed: int = 0
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic PPG-like data for tests (TDD rule 12).

    A noisy sinusoid whose amplitude and a phase feature weakly encode SBP/DBP,
    so a model can learn a non-trivial mapping in a fast unit test.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi * 8, T)
    n_subjects = max(2, n // 8)
    subjects = rng.integers(0, n_subjects, size=n)

    sbp = rng.uniform(95, 175, size=n).astype(np.float32)
    dbp = (sbp * 0.55 + rng.uniform(-8, 8, size=n)).astype(np.float32)
    amp = (sbp - 95) / 80.0
    X = np.zeros((n, T, 1), dtype=np.float32)
    for i in range(n):
        wave = amp[i] * np.sin(t + dbp[i] / 30.0) + 0.1 * rng.standard_normal(T)
        X[i, :, 0] = wave.astype(np.float32)
    y = np.stack([sbp, dbp], axis=1).astype(np.float32)
    return X, y, subjects
