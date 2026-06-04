#!/usr/bin/env python3
"""Seed the ledger from the completed iteration-0 results (one-time bootstrap).

Places the 8 iter-0 specs (4 families x 2 regimes) one-per-island into
`ledger/experiments.db`, then writes each seed's REAL fitness so the loop's
tournament selection has a basis. The fitness vector is `best_val_metrics`
merged with the top-level `param_count` and `generalization_gap` (those two
live outside best_val_metrics in result.json but are Pareto axes).

Refuses to run if the ledger is already seeded (avoids double-seeding).
Mac-side only (ANTIPATTERNS 11): no cluster ops.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "framework" / "__init__.py").exists():
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from framework.ledger import Ledger
from framework.iteration import seed_population

ITER0 = _ROOT / "experiments" / "iter_0000"
ISLANDS = 8


def main():
    run_dirs = sorted(d for d in ITER0.glob("seed_*") if (d / "result.json").exists())
    if len(run_dirs) != ISLANDS:
        print(f"expected {ISLANDS} completed iter-0 runs, found {len(run_dirs)}",
              file=sys.stderr)
        sys.exit(1)
    specs = [json.loads((d / "spec.json").read_text()) for d in run_dirs]

    led = Ledger(_ROOT / "ledger" / "experiments.db")
    led.init_schema()
    if any(led.get_island_members(i) for i in range(ISLANDS)):
        print("ledger already seeded; aborting to avoid double-seed", file=sys.stderr)
        led.close()
        sys.exit(1)

    run_ids = seed_population(led, specs, island_count=ISLANDS)
    for i, (rid, d) in enumerate(zip(run_ids, run_dirs)):
        res = json.loads((d / "result.json").read_text())
        fit = dict(res.get("best_val_metrics", {}))
        fit["param_count"] = res.get("param_count")
        fit["generalization_gap"] = res.get("generalization_gap")
        led.write_result(rid, fit)
        print(f"island {i}: {d.name:24s} -> {rid}  "
              f"aami_margin={fit.get('aami_margin'):.3f}")
    led.close()
    print(f"\nledger bootstrapped: {ISLANDS} islands seeded with real iter-0 fitness")


if __name__ == "__main__":
    main()
