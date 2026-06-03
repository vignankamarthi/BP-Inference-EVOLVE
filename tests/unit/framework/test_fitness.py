"""Tests for framework.fitness. Spec: FRAMEWORK.md Section 3."""
import pytest
import numpy as np
from framework import fitness


def test_module_imports():
    assert callable(fitness.pareto_rank)
    assert callable(fitness.novelty_score)
    assert callable(fitness.confidence_weighted)
    assert callable(fitness.failure_aware_boost)
    assert callable(fitness.scalar_score)


# ---------- confidence_weighted ----------

def test_confidence_weighted_basic_form():
    s = fitness.confidence_weighted(accuracy=0.8, ece=0.1, lam=1.0)
    assert s == pytest.approx(0.8 * 0.9)


def test_confidence_weighted_zero_ece_unchanged():
    s = fitness.confidence_weighted(accuracy=0.7, ece=0.0, lam=1.0)
    assert s == pytest.approx(0.7)


# ---------- pareto_rank ----------

def test_pareto_rank_single_individual_is_front_zero():
    fvs = [{"balanced_acc": 0.5, "ece": 0.1}]
    ranks = fitness.pareto_rank(fvs, axes=["balanced_acc", "ece"])
    assert ranks == [0]


def test_pareto_rank_clear_dominator_gets_rank_zero():
    # fv0 dominates fv1: higher bal_acc, lower ece.
    fvs = [
        {"balanced_acc": 0.9, "ece": 0.05},
        {"balanced_acc": 0.5, "ece": 0.20},
    ]
    ranks = fitness.pareto_rank(fvs, axes=["balanced_acc", "ece"])
    assert ranks[0] == 0
    assert ranks[1] == 1


def test_pareto_rank_non_dominated_share_front_zero():
    # Trade-off: high acc but bad ece vs low acc but good ece. Neither dominates.
    fvs = [
        {"balanced_acc": 0.9, "ece": 0.30},
        {"balanced_acc": 0.5, "ece": 0.02},
    ]
    ranks = fitness.pareto_rank(fvs, axes=["balanced_acc", "ece"])
    assert ranks == [0, 0]


def test_pareto_rank_three_fronts():
    fvs = [
        {"balanced_acc": 0.9, "ece": 0.05},  # front 0 (best both)
        {"balanced_acc": 0.7, "ece": 0.10},  # front 1 (dominated by 0)
        {"balanced_acc": 0.5, "ece": 0.20},  # front 2 (dominated by 0 and 1)
    ]
    ranks = fitness.pareto_rank(fvs, axes=["balanced_acc", "ece"])
    assert ranks == [0, 1, 2]


def test_pareto_rank_rejects_unknown_axis():
    fvs = [{"foo": 1.0}]
    with pytest.raises(ValueError):
        fitness.pareto_rank(fvs, axes=["foo"])


def test_pareto_rank_full_axis_set_test_from_scaffold():
    fvs = [
        {"balanced_acc": 0.7, "ece": 0.1, "param_count": 100_000, "generalization_gap": 0.05},
        {"balanced_acc": 0.8, "ece": 0.2, "param_count": 200_000, "generalization_gap": 0.10},
    ]
    ranks = fitness.pareto_rank(fvs, axes=["balanced_acc", "ece", "param_count", "generalization_gap"])
    assert len(ranks) == 2


# ---------- novelty_score ----------

def test_novelty_zero_when_population_empty():
    child = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    assert fitness.novelty_score(child, [], k=5) == 0.0


def test_novelty_zero_when_child_matches_population():
    child = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    pop = [child, child, child]
    assert fitness.novelty_score(child, pop, k=2) == pytest.approx(0.0)


def test_novelty_grows_with_distance():
    child = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    near = [np.array([[9, 1, 0], [1, 9, 0], [0, 0, 10]])]
    far = [np.array([[0, 5, 5], [5, 0, 5], [5, 5, 0]])]
    assert fitness.novelty_score(child, far, k=1) > fitness.novelty_score(child, near, k=1)


# ---------- failure_aware_boost ----------

def test_failure_aware_boost_activates_on_negative_deltas():
    out = fitness.failure_aware_boost(
        recent_deltas=[-0.05, -0.04, -0.03], threshold=-0.02)
    assert out["activate_boost"] is True
    assert out["temperature_multiplier"] > 1.0
    assert out["novelty_alpha_delta"] > 0.0


def test_failure_aware_boost_quiet_on_improving():
    out = fitness.failure_aware_boost(
        recent_deltas=[0.01, 0.02, 0.005], threshold=-0.02)
    assert out["activate_boost"] is False
    assert out["temperature_multiplier"] == pytest.approx(1.0)


def test_failure_aware_boost_handles_empty():
    out = fitness.failure_aware_boost(recent_deltas=[], threshold=-0.02)
    assert out["activate_boost"] is False


# ---------- scalar_score ----------

def test_scalar_score_returns_float():
    s = fitness.scalar_score(pareto_rank_value=0, novelty=0.3, accuracy=0.7,
                              ece=0.05, alpha=0.7, lam=1.0)
    assert isinstance(s, float)


def test_scalar_score_higher_for_lower_pareto_rank():
    same = dict(novelty=0.3, accuracy=0.7, ece=0.05, alpha=0.7, lam=1.0)
    s_front0 = fitness.scalar_score(pareto_rank_value=0, **same)
    s_front2 = fitness.scalar_score(pareto_rank_value=2, **same)
    assert s_front0 > s_front2


def test_scalar_score_higher_for_higher_accuracy():
    same = dict(pareto_rank_value=0, novelty=0.3, ece=0.05, alpha=0.7, lam=1.0)
    s_low = fitness.scalar_score(accuracy=0.5, **same)
    s_high = fitness.scalar_score(accuracy=0.9, **same)
    assert s_high > s_low


def test_scalar_score_higher_for_higher_novelty_when_alpha_low():
    """When alpha=0, score is dominated by novelty."""
    same = dict(pareto_rank_value=0, accuracy=0.7, ece=0.05, alpha=0.0, lam=1.0)
    s_low = fitness.scalar_score(novelty=0.1, **same)
    s_high = fitness.scalar_score(novelty=0.9, **same)
    assert s_high > s_low
