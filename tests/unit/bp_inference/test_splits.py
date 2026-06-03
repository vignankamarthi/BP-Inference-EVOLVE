"""TDD: subject-disjoint split utilities."""
import numpy as np
import pytest

from bp_inference import splits


def test_k_subject_subset_deterministic_and_bounded():
    subs = [f"s{i}" for i in range(10)]
    a = splits.k_subject_subset(subs, k=3, seed=0)
    b = splits.k_subject_subset(subs, k=3, seed=0)
    assert a == b                      # stable
    assert len(a) == 3
    assert set(a).issubset(set(subs))  # valid members
    with pytest.raises(ValueError):
        splits.k_subject_subset(subs, k=99, seed=0)


def test_subject_disjoint_check_detects_leak():
    splits.subject_disjoint_check({"train": {"a", "b"}, "test": {"c"}})  # ok
    with pytest.raises(ValueError):
        splits.subject_disjoint_check({"train": {"a", "b"}, "test": {"b"}})


def test_train_val_split_by_subject_is_disjoint():
    subs = np.array([f"s{i // 4}" for i in range(40)])   # 10 subjects
    train, val = splits.train_val_split_by_subject(subs, val_fraction=0.2, seed=1)
    assert set(train).isdisjoint(set(val))
    assert len(val) == 2                                  # round(10 * 0.2)
    assert set(train) | set(val) == set(subs.tolist())


def test_mask_for_subjects():
    subs = np.array(["a", "b", "a", "c"])
    mask = splits.mask_for_subjects(subs, ["a"])
    assert mask.tolist() == [True, False, True, False]
