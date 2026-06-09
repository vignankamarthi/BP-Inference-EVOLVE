#!/usr/bin/env python3
"""Render a lever-screen batch (Phase B) for one factor on the screen winners.

Reads the architecture-screen results, ranks backbones by aami_margin, takes the
top-K survivors, and crosses them with one factor's levels x n_seeds (replicated
above the iter-1 ~0.1 mmHg noise floor). Each config is smoke-tested at its true
in_channels before render. Output goes to experiments/lever_<factor>/ with one
manifest and chunked sbatch commands (the GPU QOS cap is 8 submitted jobs, so a
batch > 8 is submitted in `--array` chunks, each after the prior drains).

Usage (AFTER the screen results are pulled + ingested):
  python scripts/render_lever_screen.py --factor input_rep --top-k 2 --n-seeds 3
Mac-side only (ANTIPATTERNS 11). Vignan pushes + sbatches.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "framework" / "__init__.py").exists():
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from bp_inference.ablation import FACTORS, enumerate_lever_configs, rf_levels
from framework import render

SCREEN = _ROOT / "experiments" / "screen_arch"
CAP = 8  # GPU QOS submitted-jobs cap


def _top_survivors(k: int):
    rows = []
    for d in sorted(SCREEN.glob("screen_*")):
        rj = d / "result.json"
        if not rj.exists():
            continue
        res = json.loads(rj.read_text())
        if res.get("failed"):
            continue
        spec = json.loads((d / "spec.json").read_text())
        margin = res.get("best_val_metrics", {}).get("aami_margin", -1e9)
        rows.append((margin, spec["model"]["family"], spec))
    if not rows:
        print("no screen results found; run + pull + ingest the architecture "
              "screen first (experiments/screen_arch/*/result.json).", file=sys.stderr)
        sys.exit(1)
    rows.sort(reverse=True, key=lambda r: r[0])
    survivors = []
    for margin, fam, spec in rows[:k]:
        survivors.append((fam, spec))
        print(f"  survivor: {fam:16s} aami_margin={margin:.3f}")
    return survivors


def _smoke(spec) -> int:
    import torch
    fam = spec["model"]["family"]
    in_ch = 1 + len(spec.get("preprocessing", {}).get("derivatives", []))
    mod = importlib.import_module(render.FAMILY_ENTRY_POINTS[fam][0])
    model = mod._factory(in_channels=in_ch, T=1250, model_cfg=spec["model"], n_targets=2)
    with torch.no_grad():
        out = model(torch.randn(3, 1250, in_ch))
    assert tuple(out.shape) == (3, 2) and torch.isfinite(out).all(), spec["name"]
    return int(sum(p.numel() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True, choices=sorted(list(FACTORS) + ["rf"]))
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()

    out_dir = _ROOT / "experiments" / f"lever_{args.factor}"
    if (out_dir / "manifest_gpu.json").exists():
        print(f"{out_dir} already rendered; aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"== top-{args.top_k} survivors from the architecture screen ==")
    survivors = _top_survivors(args.top_k)
    if args.factor == "rf":
        from bp_inference.ablation import RF_EXTENDED
        specs = []
        for fam, base in survivors:                # RF levels are per-backbone
            if fam not in RF_EXTENDED:
                print(f"  skip {fam}: no tunable RF knob (fixed/global RF)")
                continue
            specs += enumerate_lever_configs([(fam, base)], rf_levels(fam),
                                             n_seeds=args.n_seeds)
    else:
        specs = enumerate_lever_configs(survivors, FACTORS[args.factor],
                                        n_seeds=args.n_seeds)

    print(f"\n== smoke test (unique backbone x input-rep; {len(specs)} configs total) ==")
    experiments, smoked = [], {}
    for spec in specs:
        # unique model build = family + in_channels (via derivatives) + model cfg
        # (the RF lever varies model cfg at the same derivatives, so it must be
        # in the key or the bigger 'ext' models never get smoke-tested).
        key = (spec["model"]["family"],
               tuple(spec.get("preprocessing", {}).get("derivatives", [])),
               json.dumps(spec["model"], sort_keys=True))
        if key not in smoked:
            smoked[key] = _smoke(spec)
            print(f"  ok  {key[0]:14s} in_ch={1 + len(key[1])}  {smoked[key]:>10,} params")
        render.render_spec_to_code(spec, out_dir / spec["name"])
        experiments.append({
            "run_id": spec["name"],
            "run_dir": str((out_dir / spec["name"]).relative_to(_ROOT)),
            "family": spec["model"]["family"],
            "regime": spec["calibration"]["mode"], "name": spec["name"],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_gpu.json").write_text(json.dumps(
        {"factor": args.factor, "experiments": experiments}, indent=2))

    n = len(experiments)
    print(f"\n{n} configs -> experiments/lever_{args.factor}/")
    print(f"cluster: submit in chunks of <= {CAP} (each after the prior drains):")
    for start in range(0, n, CAP):
        end = min(start + CAP, n) - 1
        print(f"  sbatch --array={start}-{end}%4 --time=03:00:00 "
              f"--export=ALL,MANIFEST=experiments/lever_{args.factor}/manifest_gpu.json "
              f"scripts/run_array.slurm")


if __name__ == "__main__":
    main()
