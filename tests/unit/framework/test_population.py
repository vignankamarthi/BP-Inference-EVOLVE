"""Tests for framework.population. Spec: FRAMEWORK.md Section 2."""
import pytest
from framework import population


def test_module_imports():
    assert population.IslandState is not None
    assert population.Islands is not None


# ---------- construction ----------

def test_islands_construction():
    isl = population.Islands(m=8, k=10, reset_cadence=100)
    assert len(isl) == 0
    assert isl.island_size(0) == 0


def test_islands_construction_rejects_invalid_args():
    with pytest.raises(ValueError):
        population.Islands(m=0, k=10, reset_cadence=100)
    with pytest.raises(ValueError):
        population.Islands(m=8, k=0, reset_cadence=100)
    with pytest.raises(ValueError):
        population.Islands(m=8, k=10, reset_cadence=0)


# ---------- seeding ----------

def test_seed_adds_member_without_eviction():
    isl = population.Islands(m=4, k=5, reset_cadence=100)
    for i in range(3):
        isl.seed(0, f"r{i}", {"balanced_acc": 0.5 + 0.1 * i})
    assert isl.island_size(0) == 3
    assert len(isl) == 3


def test_seed_updates_best_fitness():
    isl = population.Islands(m=4, k=5, reset_cadence=100)
    isl.seed(0, "r0", {"balanced_acc": 0.4})
    isl.seed(0, "r1", {"balanced_acc": 0.7})
    state = isl.island_state(0)
    assert state.best_fitness == pytest.approx(0.7)


# ---------- sample_parent ----------

def test_sample_parent_returns_island_member():
    isl = population.Islands(m=4, k=5, reset_cadence=100, rng_seed=0)
    for i in range(3):
        isl.seed(0, f"r{i}", {"balanced_acc": 0.5 + 0.1 * i})
    rid = isl.sample_parent(island_id=0, tournament_size=2)
    assert rid in {"r0", "r1", "r2"}


def test_sample_parent_empty_island_raises():
    isl = population.Islands(m=4, k=5, reset_cadence=100)
    with pytest.raises(ValueError):
        isl.sample_parent(island_id=0, tournament_size=2)


def test_sample_parent_tournament_picks_higher_fitness_on_average():
    """With tournament_size = island size, we always pick the best."""
    isl = population.Islands(m=1, k=10, reset_cadence=100, rng_seed=0)
    for i in range(5):
        isl.seed(0, f"r{i}", {"balanced_acc": 0.1 * i})
    rid = isl.sample_parent(island_id=0, tournament_size=5)
    assert rid == "r4"  # highest fitness


# ---------- GENITOR replacement ----------

def test_genitor_inserts_below_capacity_without_eviction():
    isl = population.Islands(m=4, k=5, reset_cadence=100)
    isl.seed(0, "r0", {"balanced_acc": 0.5})
    isl.insert_child(island_id=0, child_run_id="r_new", child_fitness={"balanced_acc": 0.6})
    assert isl.island_size(0) == 2


def test_genitor_evicts_lowest_when_at_capacity():
    isl = population.Islands(m=1, k=3, reset_cadence=100)
    isl.seed(0, "r0", {"balanced_acc": 0.3})
    isl.seed(0, "r1", {"balanced_acc": 0.5})
    isl.seed(0, "r2", {"balanced_acc": 0.7})
    isl.insert_child(island_id=0, child_run_id="r_new", child_fitness={"balanced_acc": 0.8})
    state = isl.island_state(0)
    assert "r0" not in state.member_run_ids  # lowest evicted
    assert "r_new" in state.member_run_ids
    assert isl.island_size(0) == 3


def test_genitor_updates_best_when_child_is_better():
    isl = population.Islands(m=1, k=3, reset_cadence=100)
    isl.seed(0, "r0", {"balanced_acc": 0.5})
    isl.insert_child(0, "r1", {"balanced_acc": 0.9})
    assert isl.island_state(0).best_fitness == pytest.approx(0.9)


def test_genitor_does_not_update_best_when_child_is_worse():
    isl = population.Islands(m=1, k=3, reset_cadence=100)
    isl.seed(0, "r0", {"balanced_acc": 0.9})
    isl.insert_child(0, "r1", {"balanced_acc": 0.4})
    assert isl.island_state(0).best_fitness == pytest.approx(0.9)


# ---------- maybe_reset_islands ----------

def test_reset_skipped_off_cadence():
    isl = population.Islands(m=4, k=3, reset_cadence=10)
    for i in range(4):
        isl.seed(i, f"r_{i}", {"balanced_acc": 0.5})
    reset = isl.maybe_reset_islands(current_iter=5, global_champion_run_id="champ",
                                    global_champion_fitness={"balanced_acc": 0.9})
    assert reset == []


def test_reset_fires_on_cadence_and_targets_bottom_islands():
    isl = population.Islands(m=4, k=3, reset_cadence=10)
    isl.seed(0, "r0", {"balanced_acc": 0.9})
    isl.seed(1, "r1", {"balanced_acc": 0.7})
    isl.seed(2, "r2", {"balanced_acc": 0.5})
    isl.seed(3, "r3", {"balanced_acc": 0.3})
    reset = isl.maybe_reset_islands(current_iter=10, global_champion_run_id="champ",
                                    global_champion_fitness={"balanced_acc": 0.95},
                                    fraction_to_reset=0.5)
    # Bottom 2 islands by best_fitness are 2 and 3.
    assert sorted(reset) == [2, 3]


# ---------- stagnant_islands ----------

def test_stagnant_islands_detected_at_patience():
    isl = population.Islands(m=2, k=3, reset_cadence=100)
    isl.seed(0, "r0", {"balanced_acc": 0.5})
    isl.seed(1, "r1", {"balanced_acc": 0.5})
    # Both seeded at iter 0. After patience iterations, both are stagnant.
    stale = isl.stagnant_islands(current_iter=10, patience=5)
    assert set(stale) == {0, 1}


def test_stagnant_islands_excludes_recently_improved():
    isl = population.Islands(m=2, k=3, reset_cadence=100)
    isl.seed(0, "r0", {"balanced_acc": 0.5})
    isl.seed(1, "r1", {"balanced_acc": 0.5})
    # Improve island 1 at iter 1.
    isl.insert_child(1, "r1b", {"balanced_acc": 0.8})
    stale = isl.stagnant_islands(current_iter=3, patience=5)
    assert 1 not in stale  # last improved at iter 1, patience 5 -> not stagnant


# ---------- global_champion ----------

def test_global_champion_returns_overall_best():
    isl = population.Islands(m=2, k=3, reset_cadence=100)
    isl.seed(0, "r0", {"balanced_acc": 0.5})
    isl.seed(1, "r1", {"balanced_acc": 0.9})
    isl.seed(0, "r2", {"balanced_acc": 0.7})
    champ = isl.global_champion()
    assert champ.run_id == "r1"


def test_global_champion_none_when_empty():
    isl = population.Islands(m=2, k=3, reset_cadence=100)
    assert isl.global_champion() is None
