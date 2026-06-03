"""Level 1 meta-stochastic layer.

Two parameters drift across iterations (FRAMEWORK.md Section 6):

  6.1 Mix-ratio drift (p_lit)
      Random walk in [0.2, 0.8]. Gaussian step every iteration with sigma=0.05
      by default. High p_lit -> mutation prompt biases toward literature.
      Low p_lit -> bias toward novel cross-domain analogy.

  6.2 Failure-aware boost
      Master valve. When recent fitness deltas are negative, opens up:
      temperature rises, novelty_alpha rises, tabu_k rises, lineage_cap tightens.
      Implementation delegates to framework.fitness.failure_aware_boost; this
      module applies the returned adjustments to the meta-state dict.

These two are stored in ledger.meta_state. Level 2 (introspect.py) can mutate
their drift parameters as part of framework-genome mutation.
"""
import random

from framework.fitness import failure_aware_boost


_DEFAULT_RNG = random.Random()


def drift_mix_ratio(p_lit_current: float, sigma: float = 0.05,
                    bounds: tuple[float, float] = (0.2, 0.8),
                    rng: random.Random | None = None) -> float:
    """Single Gaussian step on p_lit, clamped to bounds."""
    if rng is None:
        rng = _DEFAULT_RNG
    lo, hi = bounds
    if not (lo < hi):
        raise ValueError(f"bounds must satisfy lo < hi, got {bounds}")
    step = rng.gauss(0.0, sigma)
    p_new = p_lit_current + step
    if p_new < lo:
        p_new = lo
    elif p_new > hi:
        p_new = hi
    return p_new


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def update_failure_boost(recent_deltas: list[float], window: int,
                         threshold: float, current_state: dict) -> dict:
    """Update boost state based on rolling window of fitness deltas.

    Reads the most recent `window` deltas and asks
    framework.fitness.failure_aware_boost for adjustments. Applies the
    returned multipliers / deltas to a copy of `current_state` and returns it.

    `current_state` keys: temperature, novelty_alpha, tabu_k, lineage_cap.
    """
    window_deltas = list(recent_deltas)[-window:] if window > 0 else []
    adj = failure_aware_boost(window_deltas, threshold)
    new = dict(current_state)
    new["failure_boost_active"] = adj["activate_boost"]
    new["temperature"] = current_state.get("temperature", 1.0) * adj["temperature_multiplier"]
    new["novelty_alpha"] = _clamp(
        current_state.get("novelty_alpha", 0.3) + adj["novelty_alpha_delta"],
        0.0, 1.0)
    new["tabu_k"] = int(current_state.get("tabu_k", 50)) + int(adj["tabu_k_delta"])
    # Tighten lineage cap (lower) when boost is active.
    if adj["activate_boost"]:
        new["lineage_cap"] = max(1, int(current_state.get("lineage_cap", 5)) - 1)
    return new


def relax_toward_defaults(state: dict, defaults: dict,
                          rate: float = 0.1) -> dict:
    """Exponential decay of state toward defaults. Used when not in failure regime."""
    new = dict(state)
    for key, default in defaults.items():
        current = state.get(key, default)
        if isinstance(current, (int, float)) and isinstance(default, (int, float)):
            new[key] = type(default)(current + rate * (default - current))
    return new


_DEFAULT_META_STATE_DEFAULTS = {
    "temperature": 1.0,
    "novelty_alpha": 0.3,
    "tabu_k": 50,
    "lineage_cap": 5,
}


def step_meta_state(current_meta: dict, recent_deltas: list[float],
                    window: int = 5, threshold: float = -0.02,
                    p_lit_sigma: float = 0.05,
                    p_lit_bounds: tuple[float, float] = (0.2, 0.8),
                    rng: random.Random | None = None,
                    defaults: dict | None = None) -> dict:
    """One iteration of Level 1 meta drift. Combines 6.1 mix-ratio drift and
    6.2 failure-aware boost.

    `defaults`: relaxation targets when failure_boost is INACTIVE. Pass the
    current genome's values here so Level 2 mutations persist; otherwise
    relaxation pulls back to the hardcoded `_DEFAULT_META_STATE_DEFAULTS`
    baseline and washes out any genome change to novelty_alpha/temperature.

    Returns a complete meta-state dict.
    """
    boosted = update_failure_boost(
        recent_deltas, window=window, threshold=threshold, current_state=current_meta)
    # If not in failure regime, relax temperature/alpha back toward defaults.
    if not boosted["failure_boost_active"]:
        boosted = relax_toward_defaults(
            boosted, defaults or _DEFAULT_META_STATE_DEFAULTS, rate=0.1)
    boosted["p_lit"] = drift_mix_ratio(
        current_meta.get("p_lit", 0.5), sigma=p_lit_sigma,
        bounds=p_lit_bounds, rng=rng)
    return boosted
