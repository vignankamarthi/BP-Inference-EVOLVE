#!/usr/bin/env python3
"""Render the iteration-1 batch: 8 children (one per island), each a mutation of
its iter-0 seed aimed at lowering SBP error SD (the binding AAMI constraint).

Pipeline per child: (1) smoke-test the neural model forward on synthetic data
(fail fast before any cluster job, ANTIPATTERNS 17), (2) constraint-check
(rule_guards / ast_tabu / lineage_cap) and log the events, (3) render
spec.json + run.py, (4) record the ledger experiment + mutation trace. Then
write the 3 manifests and the family-quota gate.

Mac-side only (ANTIPATTERNS 11). Vignan pushes + sbatches. Idempotent-guarded:
refuses to run if iter-1 children already exist in the ledger.
"""
import importlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "framework" / "__init__.py").exists():
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from framework import render
from framework.constraints import ast_tabu, lineage_cap, rule_guards
from framework.iteration import validate_batch_family_quota
from framework.ledger import Ledger

ITER = 1
ITER_DIR = _ROOT / "experiments" / f"iter_{ITER:04d}"
MAX_PARAMS = 10_000_000
MAX_TRAIN_S = 28_800  # 8h cluster cap


def _spec(name, calib, feat, model, training):
    return {
        "name": name,
        "preprocessing": {"normalize": "per_channel_zscore"},
        "feature_extraction": feat,
        "model": model,
        "training": training,
        "calibration": calib,
        "data": {"signals": ["ppg"], "val_fraction": 0.2},
        "decode": {"strategy": "identity"},
    }


CAL = {"mode": "per_subject", "cal_fraction": 0.2}
FREE = {"mode": "free"}

# (island_id, spec). Bolded structural levers (loss/optimizer/alphas-length)
# keep same-family siblings fingerprint-distinct; capacity bumps kept modest so
# every GPU child finishes under the wall and minirocket stays in 24G.
CHILDREN = [
    (0, _spec("iter1_mamba_ssm_cal_wide", CAL, None,
              {"family": "mamba_ssm", "dim": 96, "n_layers": 3, "dropout": 0.2},
              {"loss": "smooth_l1", "optimizer": "adamw", "lr": 3e-4,
               "epochs": 40, "batch_size": 64, "seed": 42})),
    (1, _spec("iter1_mamba_ssm_free_deep", FREE, None,
              {"family": "mamba_ssm", "dim": 64, "n_layers": 4, "dropout": 0.2},
              {"loss": "mse", "optimizer": "adamw", "lr": 5e-4,
               "epochs": 50, "batch_size": 64, "seed": 42})),
    (2, _spec("iter1_minirocket_cal_widerf", CAL,
              {"family": "minirocket", "num_kernels": 1000, "kernel_length": 11,
               "random_state": 42},
              {"family": "ridge_regressor_cv",
               "alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
              {"loss": "ridge_regression_cv", "seed": 42})),
    (3, _spec("iter1_minirocket_free_sharp", FREE,
              {"family": "minirocket", "num_kernels": 1000, "kernel_length": 7,
               "random_state": 42},
              {"family": "ridge_regressor_cv",
               "alphas": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
              {"loss": "ridge_regression_cv", "seed": 42})),
    (4, _spec("iter1_resunet_sa_cal_wide", CAL, None,
              {"family": "resunet_sa", "base_channels": 32, "num_heads": 8,
               "dropout": 0.2},
              {"loss": "smooth_l1", "optimizer": "adam", "lr": 7e-4,
               "epochs": 40, "batch_size": 64, "seed": 42})),
    (5, _spec("iter1_resunet_sa_free_reg", FREE, None,
              {"family": "resunet_sa", "base_channels": 24, "num_heads": 4,
               "dropout": 0.3},
              {"loss": "mse", "optimizer": "adamw", "lr": 1e-3,
               "epochs": 50, "batch_size": 64, "seed": 42})),
    (6, _spec("iter1_runet_attn_cal_hifreq", CAL, None,
              {"family": "runet_attn", "base_channels": 32, "use_stft": True,
               "n_fft": 128, "hop": 8, "num_heads": 8, "dropout": 0.2},
              {"loss": "smooth_l1", "optimizer": "adamw", "lr": 5e-4,
               "epochs": 40, "batch_size": 64, "seed": 42})),
    (7, _spec("iter1_runet_attn_free_stable", FREE, None,
              {"family": "runet_attn", "base_channels": 32, "use_stft": True,
               "n_fft": 64, "hop": 16, "num_heads": 4, "dropout": 0.2},
              {"loss": "l1", "optimizer": "adamw", "lr": 3e-4,
               "epochs": 50, "batch_size": 64, "seed": 42})),
]


def _regime(spec):
    return spec.get("calibration", {}).get("mode", "free")


def smoke_test(spec) -> int | None:
    """Build the neural model + forward a synthetic batch. Returns param count.
    minirocket (sklearn) has no torch forward -> skipped."""
    fam = spec["model"]["family"]
    if render.FAMILY_COMPUTE.get(fam) != "gpu":
        return None
    import torch
    mod = importlib.import_module(render.FAMILY_ENTRY_POINTS[fam][0])
    model = mod._factory(in_channels=1, T=1250, model_cfg=spec["model"], n_targets=2)
    with torch.no_grad():
        out = model(torch.randn(3, 1250, 1))
    assert tuple(out.shape) == (3, 2), f"{spec['name']}: output shape {tuple(out.shape)} != (3, 2)"
    assert torch.isfinite(out).all(), f"{spec['name']}: non-finite output"
    return int(sum(p.numel() for p in model.parameters()))


def main():
    if (ITER_DIR / "manifest.json").exists():
        print(f"{ITER_DIR/'manifest.json'} exists; iter-1 already rendered. Aborting.",
              file=sys.stderr)
        sys.exit(1)

    led = Ledger(_ROOT / "ledger" / "experiments.db")
    led.init_schema()
    for i in range(8):
        if any(m.get("parent_id") for m in led.get_island_members(i)):
            print("iter-1 children already in ledger; aborting to avoid duplicates.",
                  file=sys.stderr)
            led.close()
            sys.exit(1)

    # ---- Pass 1: smoke-test every neural child BEFORE rendering anything ----
    print("== smoke test (synthetic forward) ==")
    for _, spec in CHILDREN:
        n = smoke_test(spec)
        tag = f"{n:,} params" if n is not None else "sklearn (skipped)"
        assert n is None or n < MAX_PARAMS, f"{spec['name']}: {n} >= max_params"
        print(f"  ok  {spec['name']:30s} {tag}")

    # ---- Pass 2: constraint-check, render, ledger ----
    print("\n== render + ledger ==")
    recent_fps = [t["fingerprint"] for t in led.recent_mutation_traces(window=64)]
    base_iter = led.current_iteration()
    experiments = []
    for k, (island_id, spec) in enumerate(CHILDREN):
        members = led.get_island_members(island_id)
        parent_rid = max(members,
                         key=lambda m: (m.get("fitness") or {}).get("aami_margin", -1e9)
                         )["run_id"]
        fp = render.fingerprint_spec(spec)
        it = base_iter + 1 + k
        for rule_name, viol in (
            ("rule_guards", rule_guards(spec, MAX_PARAMS, MAX_TRAIN_S)),
            ("ast_tabu", ast_tabu(fp, recent_fps)),
            ("lineage_cap", lineage_cap([parent_rid], cap=5)),
        ):
            led.write_constraint_event(
                iteration=it, child_fingerprint=fp, rule_name=rule_name,
                accepted=(viol is None),
                reason_code=None if viol is None else viol.rule,
                reason_detail=None if viol is None else viol.detail)
            if viol is not None:
                print(f"REJECT {spec['name']}: {viol.rule}: {viol.detail}", file=sys.stderr)
                led.close()
                sys.exit(1)

        run_dir = ITER_DIR / spec["name"]
        render.render_spec_to_code(spec, run_dir)
        child_rid = led.allocate_run_id()
        led.write_experiment(child_rid, spec, parent_id=parent_rid, island_id=island_id)
        led.write_mutation_trace(
            iteration=it, run_id=child_rid, parent_run_ids=[parent_rid],
            prompt_context=f"iter1 island {island_id}: SBP-SD-targeted mutation",
            child_spec=spec, fingerprint=fp,
            reasoning_summary=(f"mutate {_regime(spec)} {spec['model']['family']} "
                               f"toward lower SBP error SD"),
            accepted=True, run_dir=run_dir)
        recent_fps.append(fp)
        experiments.append({
            "run_id": spec["name"],
            "run_dir": str(run_dir.relative_to(_ROOT)),
            "family": spec["model"]["family"],
            "regime": _regime(spec),
            "name": spec["name"],
            "ledger_run_id": child_rid,
            "parent_run_id": parent_rid,
        })
        print(f"  rendered {spec['name']:30s} {spec['model']['family']:18s} "
              f"{_regime(spec):11s} parent={parent_rid} -> {child_rid}")
    led.close()

    manifest = {"iteration": ITER, "experiments": experiments}
    viol = validate_batch_family_quota(manifest, max_per_family=3, min_families=4)
    if viol is not None:
        print(f"FATAL family-quota: {viol.detail}", file=sys.stderr)
        sys.exit(1)

    cpu = [e for e in experiments if render.FAMILY_COMPUTE.get(e["family"]) == "cpu"]
    gpu = [e for e in experiments if render.FAMILY_COMPUTE.get(e["family"]) == "gpu"]
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    (ITER_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ITER_DIR / "manifest_cpu.json").write_text(json.dumps(
        {"iteration": ITER, "compute": "cpu", "experiments": cpu}, indent=2))
    (ITER_DIR / "manifest_gpu.json").write_text(json.dumps(
        {"iteration": ITER, "compute": "gpu", "experiments": gpu}, indent=2))

    print(f"\n{len(experiments)} children -> {ITER_DIR.relative_to(_ROOT)}/")
    print("cluster (after push + git pull) -- submit BOTH:")
    print(f"  sbatch --array=0-{len(cpu)-1} "
          f"--export=ALL,MANIFEST=experiments/iter_0001/manifest_cpu.json "
          f"scripts/run_array_cpu.slurm")
    print(f"  sbatch --array=0-{len(gpu)-1}%4 --time=03:00:00 "
          f"--export=ALL,MANIFEST=experiments/iter_0001/manifest_gpu.json "
          f"scripts/run_array.slurm")


if __name__ == "__main__":
    main()
