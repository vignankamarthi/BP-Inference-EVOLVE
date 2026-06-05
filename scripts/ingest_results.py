#!/usr/bin/env python3
"""Write a finished iteration's result.json files into the ledger, then print a
parent-vs-child comparison (aami_margin higher=better, sbp_sd lower=better).

Maps each child to its ledger run_id via the iteration manifest, and to its
iter-0 parent seed by family+regime. Idempotent (write_result is an UPDATE).
Mac-side only (ANTIPATTERNS 11).
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "framework" / "__init__.py").exists():
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from framework.ledger import Ledger

# iter-1 child -> iter-0 parent seed (by family + regime)
PARENT = {
    "iter1_mamba_ssm_cal_wide": "seed_mamba_ssm_cal",
    "iter1_mamba_ssm_free_deep": "seed_mamba_ssm_free",
    "iter1_minirocket_cal_widerf": "seed_minirocket_cal",
    "iter1_minirocket_free_sharp": "seed_minirocket_free",
    "iter1_resunet_sa_cal_wide": "seed_resunet_sa_cal",
    "iter1_resunet_sa_free_reg": "seed_resunet_sa_free",
    "iter1_runet_attn_cal_hifreq": "seed_runet_attn_cal",
    "iter1_runet_attn_free_stable": "seed_runet_attn_free",
}


def _fitness(result_path: Path):
    res = json.loads(result_path.read_text())
    if res.get("failed"):
        return None
    fit = dict(res.get("best_val_metrics", {}))
    fit["param_count"] = res.get("param_count")
    fit["generalization_gap"] = res.get("generalization_gap")
    return fit


def _delta(parent_val, child_val):
    if child_val is None:
        return "NA"
    if parent_val is None:
        return f"{child_val:.2f}"
    d = child_val - parent_val
    return f"{parent_val:.2f}->{child_val:.2f} ({'+' if d >= 0 else ''}{d:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=1)
    args = ap.parse_args()
    iter_dir = _ROOT / "experiments" / f"iter_{args.iter:04d}"
    parent_root = _ROOT / "experiments" / f"iter_{args.iter - 1:04d}"
    manifest = json.loads((iter_dir / "manifest.json").read_text())

    led = Ledger(_ROOT / "ledger" / "experiments.db")
    led.init_schema()

    print(f"{'child':30s} | {'aami_margin (higher better)':>28s} | "
          f"{'sbp_sd (lower better)':>26s}")
    print("-" * 92)
    ingested, missing = 0, []
    for e in manifest["experiments"]:
        name = e["name"]
        rj = _ROOT / e["run_dir"] / "result.json"
        if not rj.exists():
            missing.append(f"{name} (no result.json)")
            continue
        cfit = _fitness(rj)
        if cfit is None:
            missing.append(f"{name} (FAILED)")
            continue
        led.write_result(e["ledger_run_id"], cfit)
        ingested += 1

        pj = parent_root / PARENT.get(name, "") / "result.json"
        pfit = _fitness(pj) if pj.exists() else None
        am = _delta(pfit.get("aami_margin") if pfit else None, cfit.get("aami_margin"))
        ss = _delta(pfit.get("sbp_sd") if pfit else None, cfit.get("sbp_sd"))
        print(f"{name:30s} | {am:>28s} | {ss:>26s}")
    led.close()

    print(f"\ningested {ingested}/{len(manifest['experiments'])} into the ledger.")
    if missing:
        print("MISSING / FAILED:", "; ".join(missing))


if __name__ == "__main__":
    main()
