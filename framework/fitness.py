"""Scoring layer.

Combines four mechanisms (FRAMEWORK.md Section 3):

  3.1 Multi-objective Pareto via NSGA-II non-dominated sorting (Deb 2002)
      Axes: balanced_acc (max), generalization_gap (min), param_count (min),
      ece (min). Other directions accepted via `directions` argument.
  3.2 Novelty-augmented scoring (Lehman & Stanley 2011)
      novelty = mean k-NN distance from child confusion-matrix vector to
      population members in flattened-confusion space.
  3.3 Confidence-weighted accuracy (ECE-aware)
      score *= (1 - lambda * ECE)
  3.4 Failure-aware exploration boost
      Tracks rolling delta of recent fitness; when negative, activates boost
      that raises temperature, novelty_alpha, tabu strictness. Drives the
      meta_state in `framework.meta.step_meta_state`.

The `scalar_score` helper combines all four into a single tournament-selection
score: alpha * confidence_weighted(accuracy, ece, lam) +
       (1 - alpha) * novelty - rank_penalty * pareto_rank.
"""
import numpy as np


# Direction per axis: "max" means higher is better, "min" means lower is better.
DEFAULT_DIRECTIONS: dict[str, str] = {
    "balanced_acc": "max",
    "macro_f1": "max",
    "auc_ovr": "max",
    "ece": "min",
    "param_count": "min",
    "generalization_gap": "min",
    "train_seconds": "min",
    "inference_seconds": "min",
}

# Regression (BP) axes. The engine is metric-agnostic: pareto_rank takes any
# axes+directions, and novelty_score takes any vectors (here the per-target
# error-distribution signature replaces the confusion-matrix vector). The BP
# adapter passes these instead of DEFAULT_DIRECTIONS.
REGRESSION_DIRECTIONS: dict[str, str] = {
    "aami_margin": "max",          # primary compliance objective (worst target)
    "sbp_sd": "min",
    "dbp_sd": "min",
    "sbp_me_abs": "min",
    "dbp_me_abs": "min",
    "sbp_mae": "min",
    "dbp_mae": "min",
    "param_count": "min",
    "generalization_gap": "min",
    "train_seconds": "min",
    "inference_seconds": "min",
}

# Default Pareto front axes for the BP search: drive both targets' bias and
# spread toward the AAMI box while staying parsimonious and generalizing.
REGRESSION_PARETO_AXES = ["sbp_sd", "dbp_sd", "sbp_me_abs", "dbp_me_abs",
                          "param_count", "generalization_gap"]


def compliance_scalar(fitness: dict, novelty: float, alpha: float,
                      pareto_rank_value: int, rank_penalty: float = 0.1) -> float:
    """Regression analog of `scalar_score`. The primary objective is the AAMI
    margin (worst of SBP/DBP; higher = deeper inside the ME<=5, SD<=8 box),
    blended with novelty and a Pareto-front penalty.

      score = alpha * aami_margin + (1 - alpha) * novelty - rank_penalty * rank
    """
    margin = float(fitness.get("aami_margin", 0.0))
    return alpha * margin + (1.0 - alpha) * novelty - rank_penalty * pareto_rank_value


def _dominates(a: dict, b: dict, axes: list[str],
               directions: dict[str, str]) -> bool:
    """Return True if `a` Pareto-dominates `b` across `axes`.

    a dominates b iff a is at least as good on every axis AND strictly better
    on at least one axis.
    """
    better_on_any = False
    for axis in axes:
        a_val, b_val = a[axis], b[axis]
        if directions[axis] == "min":
            if a_val > b_val:
                return False
            if a_val < b_val:
                better_on_any = True
        else:  # "max"
            if a_val < b_val:
                return False
            if a_val > b_val:
                better_on_any = True
    return better_on_any


def pareto_rank(fitness_vectors: list[dict], axes: list[str],
                directions: dict[str, str] | None = None) -> list[int]:
    """NSGA-II non-dominated sorting (Deb 2002).

    Returns a list of integer ranks, one per fitness vector. Rank 0 is the
    first Pareto front (non-dominated). Rank 1 is the second front (dominated
    only by rank 0), etc.
    """
    if directions is None:
        directions = DEFAULT_DIRECTIONS
    for axis in axes:
        if axis not in directions:
            raise ValueError(f"axis {axis!r} missing from directions table")

    n = len(fitness_vectors)
    dominated_by_count = [0] * n
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    ranks = [0] * n

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(fitness_vectors[p], fitness_vectors[q], axes, directions):
                dominates_list[p].append(q)
            elif _dominates(fitness_vectors[q], fitness_vectors[p], axes, directions):
                dominated_by_count[p] += 1

    current_front = [p for p in range(n) if dominated_by_count[p] == 0]
    rank = 0
    while current_front:
        next_front = []
        for p in current_front:
            ranks[p] = rank
            for q in dominates_list[p]:
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    next_front.append(q)
        rank += 1
        current_front = next_front

    return ranks


def novelty_score(child_confusion: np.ndarray,
                  population_confusions: list[np.ndarray],
                  k: int = 5) -> float:
    """Mean k-NN distance from child to population in flattened confusion-matrix space.

    Returns 0.0 when the population is empty (no neighbors -> nothing novel
    to be measured against).
    """
    if not population_confusions:
        return 0.0
    child_flat = np.asarray(child_confusion, dtype=np.float64).flatten()
    pop_flat = np.stack([
        np.asarray(c, dtype=np.float64).flatten() for c in population_confusions
    ])
    diffs = pop_flat - child_flat
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    k_eff = min(k, len(dists))
    dists.sort()
    return float(dists[:k_eff].mean())


def confidence_weighted(accuracy: float, ece: float,
                        lam: float = 1.0) -> float:
    """Penalize sharp-confident-wrong via ECE. score = acc * (1 - lam * ece)."""
    return accuracy * (1.0 - lam * ece)


def failure_aware_boost(recent_deltas: list[float],
                        threshold: float) -> dict:
    """Detect failure regime over rolling window of fitness-delta values.

    `recent_deltas`: list of (child_fitness - parent_island_best_fitness)
                     values over the last N children.
    `threshold`: when the rolling mean falls below this, activate the boost.

    Returns a dict with adjustments to apply to meta_state:
      - activate_boost: bool
      - temperature_multiplier: float (multiplicative factor)
      - novelty_alpha_delta: float (additive factor)
      - tabu_k_delta: int (additive factor)
    """
    if not recent_deltas:
        return {
            "activate_boost": False,
            "temperature_multiplier": 1.0,
            "novelty_alpha_delta": 0.0,
            "tabu_k_delta": 0,
        }
    mean_delta = sum(recent_deltas) / len(recent_deltas)
    if mean_delta < threshold:
        return {
            "activate_boost": True,
            "temperature_multiplier": 1.5,
            "novelty_alpha_delta": 0.2,
            "tabu_k_delta": 10,
        }
    return {
        "activate_boost": False,
        "temperature_multiplier": 1.0,
        "novelty_alpha_delta": 0.0,
        "tabu_k_delta": 0,
    }


def scalar_score(pareto_rank_value: int, novelty: float, accuracy: float,
                 ece: float, alpha: float,
                 lam: float, rank_penalty: float = 0.1) -> float:
    """Combine all four scoring mechanisms into a scalar for tournament selection.

      cwa = confidence_weighted(accuracy, ece, lam)
      score = alpha * cwa + (1 - alpha) * novelty - rank_penalty * pareto_rank

    Higher is better. `alpha` trades off accuracy-vs-novelty; `lam` controls
    the ECE penalty; `rank_penalty` controls how much a worse Pareto front
    deducts.
    """
    cwa = confidence_weighted(accuracy, ece, lam)
    augmented = alpha * cwa + (1.0 - alpha) * novelty
    return augmented - rank_penalty * pareto_rank_value
