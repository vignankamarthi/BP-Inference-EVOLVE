#!/usr/bin/env python3
"""Render the seed batch (iteration 0): 4 PPG-only seeds x 2 calibration regimes.

Produces, under <out-root>/iter_<NNNN>/:
    <run_id>/spec.json + run.py      (one per spec, via framework.render)
    manifest.json                    {iteration, experiments: [{run_id, run_dir,
                                       family, regime, name}, ...]}

run_array.slurm reads manifest.json and runs experiments[SLURM_ARRAY_TASK_ID].
Track A (calibration-free) maps to array indices 0..3, Track B (per-subject)
to 4..7, so `sbatch --array=0-7%4 --export=ALL,MANIFEST=<...>/manifest.json`.

ANTIPATTERNS 11: this only RENDERS (Mac-side). Vignan pushes + sbatches.

Usage:
    python scripts/render_seed_batch.py            # iter 0, cal_fraction 0.2
    python scripts/render_seed_batch.py --iter 0 --cal-fraction 0.2 --force
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "framework" / "__init__.py").exists():
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from framework import render, seeds
from framework.iteration import validate_batch_family_quota


def _regime_of(spec: dict) -> str:
    return spec.get("calibration", {}).get("mode", "free")


def main():
    ap = argparse.ArgumentParser(description="Render the iteration-0 seed batch.")
    ap.add_argument("--out-root", type=Path, default=_ROOT / "experiments")
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--cal-fraction", type=float, default=0.2)
    ap.add_argument("--force", action="store_true",
                    help="Re-render even if a run dir already exists.")
    args = ap.parse_args()

    iter_dir = args.out_root / f"iter_{args.iter:04d}"
    specs = seeds.expand_regimes(seeds.default_seed_specs(),
                                 cal_fraction=args.cal_fraction)

    experiments = []
    for spec in specs:
        run_id = spec["name"]                         # unique: <seed>_{free,cal}
        run_dir = iter_dir / run_id
        if (run_dir / "result.json").exists() and not args.force:
            print(f"SKIP {run_id}: result.json exists (use --force to re-render)")
            continue
        render.render_spec_to_code(spec, run_dir)
        rel = run_dir.relative_to(_ROOT)
        experiments.append({
            "run_id": run_id,
            "run_dir": str(rel),                      # relative to project root
            "family": spec["model"]["family"],
            "regime": _regime_of(spec),
            "name": run_id,
        })
        print(f"rendered {run_id:28s} family={spec['model']['family']:18s} "
              f"regime={_regime_of(spec)}")

    manifest = {"iteration": args.iter, "experiments": experiments}

    # Sanity gate: the seed batch must span all 4 families (min_families=4),
    # none exceeding max_per_family (each family appears 2x here).
    violation = validate_batch_family_quota(manifest, max_per_family=3,
                                            min_families=4)
    if violation is not None:
        print(f"FATAL: family-quota violation: {violation.detail}", file=sys.stderr)
        sys.exit(1)

    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n = len(experiments)
    print(f"\nmanifest: {iter_dir / 'manifest.json'} ({n} experiments)")
    print("cluster (after push + cache build):")
    print(f"  sbatch --array=0-{n - 1}%4 "
          f"--export=ALL,MANIFEST={(iter_dir / 'manifest.json').relative_to(_ROOT)} "
          f"scripts/run_array.slurm")


if __name__ == "__main__":
    main()
