"""Subject-disjoint split utilities (ANTIPATTERNS rule 3).

Pure functions over subject-id arrays. No data access, fully unit-testable.
"""
import numpy as np


def subject_disjoint_check(split_subjects: dict[str, set]) -> None:
    """Raise if any subject appears in more than one split.

    `split_subjects`: {split_name: set_of_subject_ids}.
    """
    names = list(split_subjects)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = set(split_subjects[names[i]]) & set(split_subjects[names[j]])
            if overlap:
                raise ValueError(
                    f"subject leakage between {names[i]!r} and {names[j]!r}: "
                    f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}")


def k_subject_subset(subjects, k: int, seed: int = 0) -> list:
    """Deterministically pick `k` distinct subjects. Stable for (subjects, seed).

    Mirrors the AI4Pain subset-transfer selector: the fast-fitness inner loop
    trains on K subjects rather than the full cohort.
    """
    unique = sorted(set(subjects))
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(unique):
        raise ValueError(f"k={k} exceeds available subjects={len(unique)}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(unique))[:k]
    return [unique[i] for i in sorted(idx)]


def train_val_split_by_subject(subjects, val_fraction: float = 0.2,
                               seed: int = 0) -> tuple[list, list]:
    """Partition subjects into (train, val) by subject id. No subject in both.

    The validation split for in-loop fitness is carved from TRAIN subjects only;
    CalFree / AAMI_Test stay blind (ANTIPATTERNS rule 4).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    unique = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_idx = set(perm[:n_val].tolist())
    val = [unique[i] for i in range(len(unique)) if i in val_idx]
    train = [unique[i] for i in range(len(unique)) if i not in val_idx]
    return train, val


def mask_for_subjects(subjects, chosen) -> np.ndarray:
    """Boolean mask selecting segments whose subject is in `chosen`."""
    chosen_set = set(chosen)
    return np.array([s in chosen_set for s in subjects], dtype=bool)
