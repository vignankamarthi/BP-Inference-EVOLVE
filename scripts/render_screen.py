#!/usr/bin/env python3
"""Render the architecture-screen batch (Phase A of the strategic ablation).

The 7 new backbones + resunet_sa (the known-best existing family, as the anchor)
at a FIXED baseline: raw PPG, 10s window, equal-weight loss, cal regime, one
seed. Smoke-tested before render so no shape bug reaches a cluster job. Ranks the
backbones; the top 2-3 by aami_margin advance to the lever screens (input-rep ->
RF -> loss -> freq, replicated). runet_attn/mamba_ssm/minirocket already have
iter-0/iter-1 data, so they are not re-screened.

All 8 are GPU families -> one `--array=0-7%4` batch (within the 8-job QOS cap).
Mac-side only (ANTIPATTERNS 11). Vignan pushes + sbatches.
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

OUT = _ROOT / "experiments" / "screen_arch"

# (family, baseline model_cfg): modest, roughly comparable capacity.
BACKBONES = [
    ("resunet_sa", {"base_channels": 24, "num_heads": 4, "dropout": 0.2}),   # anchor (known best)
    ("tcn", {"channels": 48, "n_blocks": 6, "kernel_size": 7, "dropout": 0.2}),
    ("xresnet1d", {"base_channels": 32, "n_stages": 4, "blocks_per_stage": 2, "dropout": 0.2}),
    ("transformer1d", {"d_model": 64, "n_heads": 4, "n_layers": 3, "downsample": 8, "dropout": 0.2}),
    ("wavelet_net", {"channels": 32, "levels": 5, "dropout": 0.2}),
    ("inception1d", {"channels": 24, "n_blocks": 4, "dropout": 0.2}),
    ("s4", {"channels": 48, "n_blocks": 3, "kernel_len": 128, "dropout": 0.2}),
    ("bigru", {"channels": 32, "hidden": 64, "n_layers": 2, "dropout": 0.2}),
]


def baseline_spec(family: str, model_cfg: dict) -> dict:
    return {
        "name": f"screen_{family}",
        "preprocessing": {"normalize": "per_channel_zscore"},   # raw: no derivatives
        "feature_extraction": None,
        "model": {"family": family, **model_cfg},
        "training": {"loss": "smooth_l1", "optimizer": "adamw", "lr": 5e-4,
                     "epochs": 40, "batch_size": 64, "seed": 42},
        "calibration": {"mode": "per_subject", "cal_fraction": 0.2},
        "data": {"signals": ["ppg"], "val_fraction": 0.2},
        "decode": {"strategy": "identity"},
    }


def smoke(spec: dict) -> int:
    import torch
    fam = spec["model"]["family"]
    mod = importlib.import_module(render.FAMILY_ENTRY_POINTS[fam][0])
    model = mod._factory(in_channels=1, T=1250, model_cfg=spec["model"], n_targets=2)
    with torch.no_grad():
        out = model(torch.randn(3, 1250, 1))
    assert tuple(out.shape) == (3, 2), f"{spec['name']}: shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{spec['name']}: non-finite"
    return int(sum(p.numel() for p in model.parameters()))


def main():
    if (OUT / "manifest.json").exists():
        print(f"{OUT/'manifest.json'} exists; screen already rendered. Aborting.",
              file=sys.stderr)
        sys.exit(1)

    experiments = []
    print("== smoke test (synthetic forward at baseline cfg) ==")
    for fam, cfg in BACKBONES:
        spec = baseline_spec(fam, cfg)
        n = smoke(spec)
        assert render.FAMILY_COMPUTE[fam] == "gpu", f"{fam} not GPU; screen assumes GPU"
        assert n < 10_000_000, f"{spec['name']}: {n} params over cap"
        render.render_spec_to_code(spec, OUT / spec["name"])
        experiments.append({
            "run_id": spec["name"],
            "run_dir": str((OUT / spec["name"]).relative_to(_ROOT)),
            "family": fam, "regime": "per_subject", "name": spec["name"],
        })
        print(f"  ok  {spec['name']:24s} {n:>10,} params")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"iteration": "screen_arch", "experiments": experiments}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "manifest_gpu.json").write_text(json.dumps(
        {"compute": "gpu", "experiments": experiments}, indent=2))

    print(f"\n{len(experiments)} backbones -> experiments/screen_arch/")
    print("cluster (after push + git pull):")
    print(f"  sbatch --array=0-{len(experiments)-1}%4 --time=03:00:00 "
          f"--export=ALL,MANIFEST=experiments/screen_arch/manifest_gpu.json "
          f"scripts/run_array.slurm")


if __name__ == "__main__":
    main()
