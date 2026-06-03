"""FunSearch islands + GENITOR steady-state replacement.

FunSearch (Romera-Paredes et al., Nature 2024): M islands of K programs each,
periodic resets of bottom islands by reseeding from a global champion.

GENITOR (Whitley 1988): per-island replacement rule. Steady-state, not
generational. One child per iteration. Parent picked by rank-based
tournament selection. Lowest-fitness member of the island is evicted to
make room.

Fitness comparison uses the `fitness_key` (default 'balanced_acc') of the
fitness vector dict that's passed alongside each run_id.

Spec: FRAMEWORK.md Section 2.
"""
from dataclasses import dataclass, field
import math
import random


_DEFAULT_FITNESS_KEY = "balanced_acc"


@dataclass
class _Member:
    run_id: str
    fitness: dict


@dataclass
class IslandState:
    island_id: int
    member_run_ids: list
    best_fitness: float
    last_improvement_iter: int


class Islands:
    """Manages M FunSearch islands with GENITOR replacement.

    Population is in-memory. Persist via framework.ledger if you need
    crash-resume across sessions.
    """

    def __init__(self, m: int, k: int, reset_cadence: int,
                 fitness_key: str = _DEFAULT_FITNESS_KEY,
                 rng_seed: int | None = None):
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if reset_cadence < 1:
            raise ValueError(f"reset_cadence must be >= 1, got {reset_cadence}")
        self.m = m
        self.k = k
        self.reset_cadence = reset_cadence
        self.fitness_key = fitness_key
        self._islands: list[list[_Member]] = [[] for _ in range(m)]
        self._last_improvement_iter: list[int] = [0] * m
        self._best_fitness: list[float] = [-math.inf] * m
        self._iter: int = 0
        self._all_members: dict[str, _Member] = {}
        self._rng = random.Random(rng_seed)

    def __len__(self) -> int:
        return sum(len(island) for island in self._islands)

    def seed(self, island_id: int, run_id: str, fitness: dict) -> None:
        """Add an initial member without evicting anyone. Used for population seeding."""
        m = _Member(run_id=run_id, fitness=fitness)
        self._islands[island_id].append(m)
        self._all_members[run_id] = m
        fit = float(fitness.get(self.fitness_key, -math.inf))
        if fit > self._best_fitness[island_id]:
            self._best_fitness[island_id] = fit
            self._last_improvement_iter[island_id] = self._iter

    def sample_parent(self, island_id: int, tournament_size: int = 5) -> str:
        """Rank-based tournament selection. Sample `tournament_size` members
        at random, return the run_id of the highest-fitness one."""
        island = self._islands[island_id]
        if not island:
            raise ValueError(f"island {island_id} is empty; seed it first")
        size = min(tournament_size, len(island))
        candidates = self._rng.sample(island, size)
        winner = max(candidates,
                     key=lambda m: m.fitness.get(self.fitness_key, -math.inf))
        return winner.run_id

    def insert_child(self, island_id: int, child_run_id: str,
                     child_fitness: dict) -> None:
        """GENITOR replacement. Add child; if island is at capacity, evict
        the lowest-fitness member."""
        self._iter += 1
        new_member = _Member(run_id=child_run_id, fitness=child_fitness)
        island = self._islands[island_id]
        if len(island) >= self.k:
            worst_idx = min(
                range(len(island)),
                key=lambda i: island[i].fitness.get(self.fitness_key, -math.inf),
            )
            evicted = island.pop(worst_idx)
            self._all_members.pop(evicted.run_id, None)
        island.append(new_member)
        self._all_members[child_run_id] = new_member

        child_fit = float(child_fitness.get(self.fitness_key, -math.inf))
        if child_fit > self._best_fitness[island_id]:
            self._best_fitness[island_id] = child_fit
            self._last_improvement_iter[island_id] = self._iter

    def island_state(self, island_id: int) -> IslandState:
        island = self._islands[island_id]
        return IslandState(
            island_id=island_id,
            member_run_ids=[m.run_id for m in island],
            best_fitness=self._best_fitness[island_id],
            last_improvement_iter=self._last_improvement_iter[island_id],
        )

    def island_size(self, island_id: int) -> int:
        return len(self._islands[island_id])

    def global_champion(self) -> _Member | None:
        if not self._all_members:
            return None
        return max(self._all_members.values(),
                   key=lambda m: m.fitness.get(self.fitness_key, -math.inf))

    def maybe_reset_islands(self, current_iter: int,
                            global_champion_run_id: str,
                            global_champion_fitness: dict | None = None,
                            fraction_to_reset: float = 0.25) -> list[int]:
        """Periodic reset of the bottom `fraction_to_reset` of islands.

        Fires only when `current_iter % reset_cadence == 0` and current_iter > 0.
        Bottom islands (by best_fitness) get cleared and reseeded with a single
        member: the global champion (with provided fitness).

        Returns the list of island_ids that were reset.
        """
        if current_iter <= 0 or current_iter % self.reset_cadence != 0:
            return []

        n_to_reset = max(1, int(self.m * fraction_to_reset))
        # Sort islands by best_fitness ascending; bottom = worst
        order = sorted(range(self.m), key=lambda i: self._best_fitness[i])
        targets = order[:n_to_reset]

        for idx in targets:
            # Remove all members from global registry
            for member in self._islands[idx]:
                self._all_members.pop(member.run_id, None)
            self._islands[idx] = []
            self._best_fitness[idx] = -math.inf
            self._last_improvement_iter[idx] = current_iter
            # Reseed with champion
            if global_champion_run_id is not None:
                seed_fitness = global_champion_fitness or {}
                self.seed(idx, global_champion_run_id + f"_reseed_{idx}", seed_fitness)

        return targets

    def stagnant_islands(self, current_iter: int, patience: int) -> list[int]:
        """Return island_ids whose best fitness has not improved in `patience`
        generations as of `current_iter`."""
        return [
            i for i in range(self.m)
            if current_iter - self._last_improvement_iter[i] >= patience
        ]
