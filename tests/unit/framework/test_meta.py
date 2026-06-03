"""Tests for framework.meta. Spec: FRAMEWORK.md Section 6."""
import random
import pytest
from framework import meta


def test_module_imports():
    assert callable(meta.drift_mix_ratio)
    assert callable(meta.update_failure_boost)
    assert callable(meta.step_meta_state)


# ---------- drift_mix_ratio ----------

def test_drift_mix_ratio_stays_in_bounds():
    rng = random.Random(0)
    p = 0.5
    for _ in range(500):
        p = meta.drift_mix_ratio(p, sigma=0.2, bounds=(0.2, 0.8), rng=rng)
        assert 0.2 <= p <= 0.8


def test_drift_mix_ratio_clamps_low():
    rng = random.Random(0)
    # With large negative steps and a low starting point, must clamp to lo.
    p = 0.21
    for _ in range(50):
        p = meta.drift_mix_ratio(p, sigma=1.0, bounds=(0.2, 0.8), rng=rng)
        assert p >= 0.2


def test_drift_mix_ratio_clamps_high():
    rng = random.Random(0)
    p = 0.79
    for _ in range(50):
        p = meta.drift_mix_ratio(p, sigma=1.0, bounds=(0.2, 0.8), rng=rng)
        assert p <= 0.8


def test_drift_mix_ratio_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        meta.drift_mix_ratio(0.5, sigma=0.05, bounds=(0.8, 0.2))


# ---------- update_failure_boost ----------

def _baseline_state():
    return {"temperature": 0.5, "novelty_alpha": 0.3, "tabu_k": 50,
            "lineage_cap": 5}


def test_failure_boost_activates_on_negative_deltas():
    out = meta.update_failure_boost(
        recent_deltas=[-0.05, -0.04, -0.06],
        window=3, threshold=-0.02, current_state=_baseline_state())
    assert out["failure_boost_active"] is True
    assert out["temperature"] > 0.5
    assert out["novelty_alpha"] > 0.3
    assert out["tabu_k"] > 50


def test_failure_boost_tightens_lineage_cap_when_active():
    out = meta.update_failure_boost(
        recent_deltas=[-0.1, -0.05, -0.07],
        window=3, threshold=-0.02, current_state=_baseline_state())
    assert out["lineage_cap"] < 5


def test_failure_boost_quiet_on_improving():
    out = meta.update_failure_boost(
        recent_deltas=[0.01, 0.02, 0.005],
        window=3, threshold=-0.02, current_state=_baseline_state())
    assert out["failure_boost_active"] is False
    assert out["temperature"] == pytest.approx(0.5)


def test_failure_boost_novelty_alpha_clamped_to_one():
    state = _baseline_state()
    state["novelty_alpha"] = 0.95
    out = meta.update_failure_boost(
        recent_deltas=[-0.1, -0.1, -0.1],
        window=3, threshold=-0.02, current_state=state)
    assert out["novelty_alpha"] <= 1.0


# ---------- step_meta_state ----------

def test_step_meta_state_returns_complete_state():
    rng = random.Random(0)
    out = meta.step_meta_state(
        current_meta={"p_lit": 0.5, "novelty_alpha": 0.3, "temperature": 0.5,
                      "failure_boost_active": False, "tabu_k": 50, "lineage_cap": 5},
        recent_deltas=[0.01, 0.02, -0.01], rng=rng)
    for key in ("p_lit", "novelty_alpha", "temperature", "failure_boost_active",
                "tabu_k", "lineage_cap"):
        assert key in out


def test_step_meta_state_widens_under_negative_deltas():
    rng = random.Random(0)
    base = {"p_lit": 0.5, "novelty_alpha": 0.3, "temperature": 0.5,
            "failure_boost_active": False, "tabu_k": 50, "lineage_cap": 5}
    out = meta.step_meta_state(base, recent_deltas=[-0.05, -0.04, -0.05],
                                rng=rng)
    assert out["temperature"] > base["temperature"]
    assert out["novelty_alpha"] > base["novelty_alpha"]


def test_step_meta_state_relaxes_toward_defaults_when_improving():
    rng = random.Random(0)
    # Inflated state should relax back down when deltas are positive.
    base = {"p_lit": 0.5, "novelty_alpha": 0.7, "temperature": 1.5,
            "failure_boost_active": False, "tabu_k": 80, "lineage_cap": 3}
    out = meta.step_meta_state(base, recent_deltas=[0.05, 0.04, 0.05],
                                rng=rng)
    assert out["temperature"] < base["temperature"]
    assert out["novelty_alpha"] < base["novelty_alpha"]
