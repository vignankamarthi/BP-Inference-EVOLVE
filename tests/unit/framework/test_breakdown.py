"""Tests for framework.breakdown. Spec: FRAMEWORK.md Section 5."""
import pytest
from framework import breakdown


def test_module_imports():
    assert breakdown.CriticIndividual is not None
    assert callable(breakdown.trigger_migration)
    assert breakdown.CriticPopulation is not None
    assert callable(breakdown.stagnation_escalation)


# ---------- trigger_migration ----------

def test_trigger_migration_returns_dict():
    child = breakdown.trigger_migration(stagnant_island_id=2,
                                         champion_run_id="run_a",
                                         foreign_champion_run_id="run_b")
    assert isinstance(child, dict)


def test_trigger_migration_records_parent_ids():
    child = breakdown.trigger_migration(stagnant_island_id=2,
                                         champion_run_id="run_a",
                                         foreign_champion_run_id="run_b")
    assert child["local_parent_run_id"] == "run_a"
    assert child["foreign_parent_run_id"] == "run_b"
    assert child["mutation_type"] == "migration_crossover"


# ---------- CriticPopulation construction ----------

def test_critic_population_construction_rejects_size_zero():
    with pytest.raises(ValueError):
        breakdown.CriticPopulation(size=0)


def test_critic_population_seeds_to_size():
    pop = breakdown.CriticPopulation(size=15, rng_seed=0)
    assert len(pop) == 15


def test_critic_genome_shape():
    """Compact 3-tuple per FRAMEWORK.md Section 9 decision 1."""
    pop = breakdown.CriticPopulation(size=5, rng_seed=0)
    crit = pop.hardest_critics(1)[0]
    assert isinstance(crit.subject_subset, list)
    assert isinstance(crit.signal_perturbation, dict)
    assert isinstance(crit.channel_permutation, list)
    assert len(crit.channel_permutation) == 4
    assert sorted(crit.channel_permutation) == [0, 1, 2, 3]


# ---------- evolve_one ----------

def test_evolve_one_returns_critic_individual():
    pop = breakdown.CriticPopulation(size=10, rng_seed=0)
    crit = pop.evolve_one(program_population_failures={})
    assert isinstance(crit, breakdown.CriticIndividual)


def test_evolve_one_keeps_population_size_constant():
    pop = breakdown.CriticPopulation(size=10, rng_seed=0)
    pop.evolve_one(program_population_failures={})
    pop.evolve_one(program_population_failures={})
    assert len(pop) == 10


def test_evolve_one_seeds_child_fitness_from_program_failures():
    pop = breakdown.CriticPopulation(size=10, rng_seed=0)
    failures = {"prog_1": 0.4, "prog_2": 0.6, "prog_3": 0.8}
    child = pop.evolve_one(program_population_failures=failures)
    assert child.fitness == pytest.approx((0.4 + 0.6 + 0.8) / 3)


# ---------- hardest_critics ----------

def test_hardest_critics_returns_top_n_by_fitness():
    pop = breakdown.CriticPopulation(size=10, rng_seed=0)
    # Manually inject known fitnesses.
    pop._members[0].fitness = 0.9
    pop._members[1].fitness = 0.5
    pop._members[2].fitness = 0.7
    top = pop.hardest_critics(n=2)
    assert len(top) == 2
    assert top[0].fitness == pytest.approx(0.9)


def test_hardest_critics_clipped_to_population_size():
    pop = breakdown.CriticPopulation(size=5, rng_seed=0)
    out = pop.hardest_critics(n=100)
    assert len(out) == 5


# ---------- stagnation_escalation ----------

def test_stagnation_escalation_raises_temperature():
    new_meta = breakdown.stagnation_escalation(
        island_id=0, patience=10,
        current_meta={"temperature": 0.5, "novelty_alpha": 0.3},
    )
    assert new_meta["temperature"] > 0.5


def test_stagnation_escalation_raises_novelty_alpha():
    new_meta = breakdown.stagnation_escalation(
        island_id=0, patience=10,
        current_meta={"temperature": 0.5, "novelty_alpha": 0.3},
    )
    assert new_meta["novelty_alpha"] > 0.3


def test_stagnation_escalation_clamps_novelty_alpha_to_one():
    new_meta = breakdown.stagnation_escalation(
        island_id=0, patience=10,
        current_meta={"temperature": 0.5, "novelty_alpha": 0.95},
    )
    assert new_meta["novelty_alpha"] <= 1.0


def test_stagnation_escalation_records_island_id():
    new_meta = breakdown.stagnation_escalation(
        island_id=3, patience=5,
        current_meta={"temperature": 1.0, "novelty_alpha": 0.5},
    )
    assert new_meta["last_escalated_island"] == 3
    assert new_meta["last_escalation_patience"] == 5
