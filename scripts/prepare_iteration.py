#!/usr/bin/env python3
"""Run one prepare_batch over the ledger and print the parent/meta summary.

The mutation operator (the Claude Code session) reads this to design the next
batch of children. Mac-side only (ANTIPATTERNS 11).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "framework" / "__init__.py").exists():
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from framework.ledger import Ledger
from framework.iteration import prepare_batch


def main():
    led = Ledger(_ROOT / "ledger" / "experiments.db")
    batch = prepare_batch(led, island_count=8, tournament_size=3, rng_seed=42,
                          composite_scoring=True, evolve_meta=True,
                          stagnation_patience=5, migration_patience=10)
    m = batch[0]["meta"]
    print(f"meta: p_lit={m.p_lit:.3f} novelty_alpha={m.novelty_alpha:.3f} "
          f"temperature={m.temperature:.3f} failure_boost={m.failure_boost_active}")
    print(f"batch size: {len(batch)}\n")
    for e in batch:
        sp = e["parent_spec"]
        fam = sp["model"]["family"]
        reg = sp.get("calibration", {}).get("mode", "?")
        xpool = [s["model"]["family"] for s in e.get("crossover_pool", [])]
        print(f"island {e['island_id']}: parent={e['parent_run_id']} "
              f"{fam}/{reg} stagnant={e['stagnant']} gap={e['island_gap']} "
              f"xpool={xpool}")
    print("\n--- example mutation prompt (island 0, first 1400 chars) ---\n")
    print(batch[0]["prompt"][:1400])
    led.close()


if __name__ == "__main__":
    main()
