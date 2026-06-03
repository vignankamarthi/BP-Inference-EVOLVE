"""Breakdown layer: unsticking stagnant mechanisms.

Three mechanisms (FRAMEWORK.md Section 5):

  5.1 Inter-island migration
      When an island's best flatlines, mate its champion with a foreign
      island's champion. The crossover is expressed as a structured prompt
      directive carried in the child spec, since the actual program-spec
      mutation happens via the Claude Code session at the next iteration.
  5.2 Coevolutionary critic (Hillis 1990)
      Separate evolving population of compact 3-tuples
      (subject_subset, signal_perturbation, channel_permutation) per
      FRAMEWORK.md Section 9 decision 1. Critics evolve to maximize the
      program population's failure rate. Real arms race, not just monotone
      descent.
  5.3 Stagnation -> escalating temperature
      Per-island detector. Raise temperature, novelty_alpha, tabu strictness
      when patience exceeded.
"""
from dataclasses import dataclass, field
import math
import random
import uuid


# Default channel permutations for the BVP/EDA/RESP/SpO2 4-tuple. The
# permutations are over channel indices [0, 1, 2, 3].
_CHANNEL_COUNT = 4


@dataclass
class CriticIndividual:
    """A compact 3-tuple critic per FRAMEWORK.md Section 9 decision 1.

    Genome:
      - subject_subset: list of subject IDs to perturb on.
      - signal_perturbation: {"signal_name": {"type": str, "magnitude": float}}
      - channel_permutation: list[int] of length _CHANNEL_COUNT, permutation
        of [0..3] indicating channel reordering.

    Fitness = mean failure rate of the program population on this critic.
    Higher fitness = better critic (more programs fail).
    """
    critic_id: str
    subject_subset: list[str]
    signal_perturbation: dict
    channel_permutation: list[int]
    fitness: float = 0.0


def trigger_migration(stagnant_island_id: int, champion_run_id: str,
                      foreign_champion_run_id: str) -> dict:
    """Build a structured crossover directive for the next mutation prompt.

    Returns a dict that becomes part of the child spec context, telling the
    Claude Code mutation operator to produce a child that crosses two parent
    champions rather than mutating one.
    """
    return {
        "mutation_type": "migration_crossover",
        "stagnant_island_id": stagnant_island_id,
        "local_parent_run_id": champion_run_id,
        "foreign_parent_run_id": foreign_champion_run_id,
        "directive": (
            "Compose a child program that merges the model family / preprocessing "
            f"pipeline of {champion_run_id} with the training / fitness-shaping "
            f"choices of {foreign_champion_run_id}. The child should not be a "
            "minimal tweak of either parent. Preserve only what is mutually best."
        ),
    }


def _random_subset(rng: random.Random, all_subjects: list[str],
                   k_min: int, k_max: int) -> list[str]:
    if not all_subjects:
        return []
    k = rng.randint(k_min, min(k_max, len(all_subjects)))
    return rng.sample(all_subjects, k)


def _random_perturbation(rng: random.Random) -> dict:
    signals = ["Bvp", "Eda", "Resp", "SpO2"]
    perturb_types = ["gaussian_noise", "scale", "drop", "shift"]
    chosen_signal = rng.choice(signals)
    chosen_type = rng.choice(perturb_types)
    return {
        chosen_signal: {
            "type": chosen_type,
            "magnitude": round(rng.uniform(0.05, 0.5), 3),
        },
    }


def _random_permutation(rng: random.Random) -> list[int]:
    perm = list(range(_CHANNEL_COUNT))
    rng.shuffle(perm)
    return perm


class CriticPopulation:
    """Hillis-style coevolutionary critic population.

    Persisted via framework.ledger.write_critic_population (future). Here we
    keep the in-memory state.
    """

    def __init__(self, size: int,
                 candidate_subjects: list[str] | None = None,
                 rng_seed: int | None = None):
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.size = size
        self.candidate_subjects = candidate_subjects or [str(i) for i in range(1, 42)]
        self._rng = random.Random(rng_seed)
        self._members: list[CriticIndividual] = []
        # Seed population with random critics.
        for _ in range(size):
            self._members.append(self._random_critic(fitness=0.0))

    def _random_critic(self, fitness: float = 0.0) -> CriticIndividual:
        return CriticIndividual(
            critic_id=uuid.uuid4().hex[:12],
            subject_subset=_random_subset(self._rng, self.candidate_subjects,
                                          k_min=2, k_max=6),
            signal_perturbation=_random_perturbation(self._rng),
            channel_permutation=_random_permutation(self._rng),
            fitness=fitness,
        )

    def evolve_one(self, program_population_failures: dict[str, float]
                    ) -> CriticIndividual:
        """Add one new critic via mutation of an existing member.

        `program_population_failures` maps `program_run_id -> failure_rate`
        for the most recent evaluation against the current critic population.
        Used to seed the new critic's fitness as the mean failure rate of the
        parent's predecessor (smoothed). The newly generated critic starts
        with fitness equal to the mean failure rate of the program population.

        GENITOR-style replacement: lowest-fitness critic is evicted.
        """
        # Choose a parent: tournament of 3 by fitness (high fitness wins).
        candidates = self._rng.sample(self._members, min(3, len(self._members)))
        parent = max(candidates, key=lambda c: c.fitness)
        child = self._mutate(parent)

        # Seed child fitness with mean of program failures (heuristic).
        if program_population_failures:
            mean_fail = sum(program_population_failures.values()) / len(program_population_failures)
            child.fitness = mean_fail

        # GENITOR eviction of lowest-fitness critic.
        worst_idx = min(range(len(self._members)),
                        key=lambda i: self._members[i].fitness)
        self._members[worst_idx] = child
        return child

    def _mutate(self, parent: CriticIndividual) -> CriticIndividual:
        """Apply one of three mutation operators to a parent critic, return child."""
        ops = ("perturb_subset", "perturb_signal", "perturb_permutation")
        op = self._rng.choice(ops)

        new_subset = list(parent.subject_subset)
        new_pert = dict(parent.signal_perturbation)
        new_perm = list(parent.channel_permutation)

        if op == "perturb_subset":
            if new_subset and self._rng.random() < 0.5 and self.candidate_subjects:
                # Swap one subject for a different candidate.
                idx = self._rng.randrange(len(new_subset))
                pool = [s for s in self.candidate_subjects if s not in new_subset]
                if pool:
                    new_subset[idx] = self._rng.choice(pool)
            else:
                new_subset = _random_subset(self._rng, self.candidate_subjects, 2, 6)
        elif op == "perturb_signal":
            new_pert = _random_perturbation(self._rng)
        else:  # perturb_permutation
            new_perm = _random_permutation(self._rng)

        return CriticIndividual(
            critic_id=uuid.uuid4().hex[:12],
            subject_subset=new_subset,
            signal_perturbation=new_pert,
            channel_permutation=new_perm,
            fitness=parent.fitness * 0.9,  # slight optimism decay
        )

    def hardest_critics(self, n: int) -> list[CriticIndividual]:
        """Return top-n critics by fitness (descending)."""
        return sorted(self._members, key=lambda c: -c.fitness)[:n]

    def __len__(self) -> int:
        return len(self._members)


def stagnation_escalation(island_id: int, patience: int,
                          current_meta: dict) -> dict:
    """Per-island stagnation response. Return new meta_state with raised knobs.

    Multiplies temperature by 1.5, adds 0.2 to novelty_alpha (clamped to <=1),
    leaves other fields untouched (caller merges).
    """
    new_meta = dict(current_meta)
    new_meta["temperature"] = current_meta.get("temperature", 1.0) * 1.5
    new_meta["novelty_alpha"] = min(
        1.0, current_meta.get("novelty_alpha", 0.0) + 0.2)
    new_meta["last_escalated_island"] = island_id
    new_meta["last_escalation_patience"] = patience
    return new_meta
