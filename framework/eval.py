"""Evaluation harness.

Three roles:

1. `evaluate_program(run_dir)`: read the fitness vector from
   `run_dir/result.json` (written by the cluster training run). Per the manual
   cluster workflow (ANTIPATTERNS rule 10), this function never trains; the
   training already happened on the cluster.

2. `subset_transfer_experiment(baseline_spec, k_grid, n_seeds, out_dir)`:
   generate `len(k_grid) * n_seeds` experiment directories under `out_dir`,
   each holding a spec.json and run.py customized for one (K, seed)
   combination. Vignan pushes them all to the cluster (HIP-D / E / F) and
   the cluster trains. Subject-subset filtering happens at train time via
   the spec's data.subset_size and data.subset_seed fields, which
   ai4pain.baselines.train_baseline honors. A manifest.json is written to
   out_dir summarizing the run.

3. `compute_subset_transfer_results(manifest_path, full_baseline_val_acc)`:
   after all experiments have run and result.json files are present, aggregate
   the val_bal_acc per K, compute the gap from the full-41 baseline, and
   choose the smallest K within tolerance (HIP-B).

Spec: FRAMEWORK.md Section 3 + decision 2 (K grid locked at {5, 10, 15, 20, 25}).
"""
from pathlib import Path
import copy
import json
import numpy as np

from framework import render


DEFAULT_K_GRID = (5, 10, 15, 20, 25)
DEFAULT_N_SEEDS = 3
DEFAULT_TOLERANCE = 0.02  # |subset_mean - full_baseline| < this -> acceptable


def evaluate_program(run_dir: Path,
                     k_subject_subset: list[str] | None = None) -> dict:
    """Read the fitness vector from a completed run's result.json.

    `k_subject_subset` is accepted as metadata but ignored here; subset filtering
    happens at train time inside ai4pain.baselines.train_baseline.

    Returns the `best_val_metrics` dict (balanced_acc, macro_f1, per_class_pr,
    confusion_3x3, auc_ovr, ece) augmented with the `param_count`,
    `train_seconds`, `inference_seconds`, and `generalization_gap` fields
    pulled from the top-level result.
    """
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"result.json not found at {result_path}. "
            f"Run HIP-D / HIP-E / HIP-F first.")
    with open(result_path) as f:
        result = json.load(f)

    fv = dict(result.get("best_val_metrics", {}))
    for key in ("param_count", "train_seconds", "inference_seconds",
                "generalization_gap"):
        if key in result:
            fv[key] = result[key]

    # Classification fallback: derive generalization_gap from the bal-acc
    # history if the run did not write it top-level (the regression harness
    # writes generalization_gap directly, so this block is skipped for BP).
    history = result.get("history", [])
    best = result.get("best_val_metrics", {})
    if "generalization_gap" not in fv and history and "balanced_acc" in best:
        val_target = best["balanced_acc"]
        best_epoch = max(history, key=lambda h: h.get("val_bal_acc", -1.0))
        if best_epoch.get("val_bal_acc") == val_target:
            fv["generalization_gap"] = (
                best_epoch.get("train_bal_acc", val_target) - val_target)

    return fv


def subset_transfer_experiment(baseline_spec: dict,
                                k_grid: list[int] | tuple[int, ...] = DEFAULT_K_GRID,
                                n_seeds: int = DEFAULT_N_SEEDS,
                                out_dir: Path = Path("experiments")) -> dict:
    """Generate one experiment directory per (K, seed) combination.

    Each spec is a deep copy of `baseline_spec` with `data.subset_size`,
    `data.subset_seed`, and a custom `training.seed` injected. Rendered via
    framework.render. A manifest.json summarizing all generated experiments
    is written to out_dir/manifest.json.

    Returns the manifest dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "baseline_spec_name": baseline_spec.get("name", "subset_transfer_baseline"),
        "k_grid": list(k_grid),
        "n_seeds": int(n_seeds),
        "tolerance": DEFAULT_TOLERANCE,
        "experiments": [],
    }

    for k in k_grid:
        for seed_idx in range(int(n_seeds)):
            run_id = f"subset_K{k:02d}_s{seed_idx}"
            run_dir = out_dir / run_id

            spec = copy.deepcopy(baseline_spec)
            spec["name"] = run_id
            spec.setdefault("data", {})
            spec["data"]["subset_size"] = int(k)
            spec["data"]["subset_seed"] = int(seed_idx)
            spec.setdefault("training", {})
            spec["training"]["seed"] = 100 + int(seed_idx)

            render.render_spec_to_code(spec, run_dir)

            manifest["experiments"].append({
                "run_id": run_id,
                "K": int(k),
                "seed_idx": int(seed_idx),
                "run_dir": str(run_dir),
            })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def compute_subset_transfer_results(manifest_path: Path,
                                     full_baseline_val_acc: float | None = None,
                                     tolerance: float = DEFAULT_TOLERANCE,
                                     metric_key: str = "balanced_acc") -> dict:
    """Aggregate the result.json files for all experiments in a manifest.

    For each K, compute mean and std of val_bal_acc across n_seeds. If
    `full_baseline_val_acc` is provided, also compute the absolute gap and
    pick the smallest K within `tolerance`.

    Returns:
      {
        "per_k": {K: {"mean": float, "std": float, "n_seeds": int, "gap": float}},
        "full_baseline": float | None,
        "chosen_k": int | None,
        "correlations": {"subset_to_full_gap": dict},
        "tolerance": float,
      }
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    out_dir = manifest_path.parent

    per_k_accs: dict[int, list[float]] = {}
    for exp in manifest["experiments"]:
        result_path = out_dir / exp["run_id"] / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        val_acc = result.get("best_val_metrics", {}).get(metric_key)
        if val_acc is None:
            continue
        per_k_accs.setdefault(exp["K"], []).append(float(val_acc))

    per_k = {}
    for k, accs in per_k_accs.items():
        per_k[k] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "n_seeds": len(accs),
        }
        if full_baseline_val_acc is not None:
            per_k[k]["gap"] = abs(per_k[k]["mean"] - float(full_baseline_val_acc))

    chosen_k = None
    if full_baseline_val_acc is not None:
        for k in sorted(per_k):
            if per_k[k].get("gap", float("inf")) <= tolerance:
                chosen_k = int(k)
                break

    correlations = {
        "subset_to_full_gap": (
            {k: per_k[k].get("gap") for k in per_k}
            if full_baseline_val_acc is not None else {}
        ),
    }

    return {
        "per_k": per_k,
        "full_baseline": full_baseline_val_acc,
        "chosen_k": chosen_k,
        "correlations": correlations,
        "tolerance": tolerance,
    }
