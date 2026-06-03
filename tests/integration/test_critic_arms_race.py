"""Integration: coevolutionary critic vs program population.

FRAMEWORK.md Section 5.2: Hillis-style arms race. Critic population evolves
to maximize program failure rates, program population evolves to handle
critic-induced perturbations.
"""
from framework import breakdown


def test_critic_evolution_round():
    pop = breakdown.CriticPopulation(size=20, rng_seed=0)
    fake_failures = {"program_run_id_1": 0.3, "program_run_id_2": 0.1}
    crit = pop.evolve_one(program_population_failures=fake_failures)
    assert isinstance(crit, breakdown.CriticIndividual)


def test_hardest_critics_surfaced_after_evolution():
    pop = breakdown.CriticPopulation(size=20, rng_seed=0)
    for i in range(5):
        pop.evolve_one(program_population_failures={"p": 0.5 + 0.05 * i})
    top = pop.hardest_critics(n=3)
    assert len(top) <= 3
    # Sorted by fitness descending.
    fitnesses = [c.fitness for c in top]
    assert fitnesses == sorted(fitnesses, reverse=True)
