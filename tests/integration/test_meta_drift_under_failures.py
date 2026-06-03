"""Integration: Level 1 meta-stochastic responds to failure regimes.

FRAMEWORK.md Section 6.2 + 3.4: when recent fitness deltas turn negative,
the failure-aware boost should raise temperature and novelty_alpha. When
deltas turn positive, state relaxes back toward defaults.
"""
import random
from framework import meta


def test_meta_drift_widens_under_negative_deltas():
    rng = random.Random(0)
    base = {"p_lit": 0.5, "novelty_alpha": 0.3, "temperature": 0.5,
            "failure_boost_active": False, "tabu_k": 50, "lineage_cap": 5}
    out = meta.step_meta_state(current_meta=base,
                               recent_deltas=[-0.05, -0.04, -0.03, -0.05],
                               rng=rng)
    assert out["temperature"] > base["temperature"]
    assert out["novelty_alpha"] > base["novelty_alpha"]
    assert out["failure_boost_active"] is True


def test_meta_drift_relaxes_when_improving():
    rng = random.Random(0)
    # Inflated state (above the meta defaults of temp 1.0, alpha 0.3) so that
    # relaxation moves the values back DOWN toward defaults under positive deltas.
    base = {"p_lit": 0.5, "novelty_alpha": 0.7, "temperature": 1.5,
            "failure_boost_active": True, "tabu_k": 80, "lineage_cap": 3}
    out = meta.step_meta_state(current_meta=base,
                               recent_deltas=[0.04, 0.03, 0.05],
                               rng=rng)
    assert out["temperature"] < base["temperature"]
    assert out["novelty_alpha"] < base["novelty_alpha"]
