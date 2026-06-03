#!/usr/bin/env python3
"""Build the PPG-only split cache from raw PulseDB. CLUSTER-ONLY (data ~963 GB).

Reads PulseDB Info + segment files (MATLAB v7.3 / HDF5) and writes one .npz per
split into <out-dir> in the contract of bp_inference.data.load_split:
    X (N, 1250, 1) PPG-only float32, sbp, dbp, subjects.

ANTIPATTERNS rule 2: only PPG_Record is requested from segment files; ECG/ABP
are never read. Rule 3/4: a subject-disjoint `validation` split is carved from
`train` and the leakage check is run before exit. Rule 11: this is run by hand
via SLURM (see scripts/build_cache.slurm); the framework never invokes it.

Expected PulseDB layout under --pulsedb-dir:
    MIMIC/PulseDB_MIMIC/p######.mat        (subject segment files)
    VitalDB/PulseDB_Vital/p######.mat
    Info_Files/{Train,CalFree_Test,CalBased_Test,AAMI_Test,AAMI_Cal}_Info.mat

Usage:
    python scripts/build_pulsedb_cache.py --pulsedb-dir <PULSEDB_ROOT> --out-dir data/raw
"""
import argparse
import sys
import time
from pathlib import Path

# Locate project root (dir containing bp_inference/) and put it on the path.
_ROOT = Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "bp_inference" / "__init__.py").exists():
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from bp_inference import export
from bp_inference.splits import subject_disjoint_check


def _resolve_subdir(root: Path, outer: str, inner: str) -> Path:
    """PulseDB archives extract with a nested subdir; fall back to flat."""
    nested = root / outer / inner
    return nested if nested.exists() else root / outer


def _make_ppg_loader(mimic_dir: Path, vital_dir: Path, subject_source: dict):
    """Return ppg_loader(subj_name, seg_indices) -> {seg_idx: PPG 1D array}.

    Resolves the subject's segment file (MIMIC vs VitalDB) and reads ONLY the
    PPG_Record field for the requested segments.
    """
    def ppg_loader(subj_name, seg_indices):
        subj_id = subj_name.split("_")[0]                  # "p072634_0" -> "p072634"
        source = subject_source.get(subj_name, "MIMIC")
        seg_path = (mimic_dir if source == "MIMIC" else vital_dir) / f"{subj_id}.mat"
        if not seg_path.exists():
            return {}
        try:
            loaded = export.load_subject_signals_h5py(
                seg_path, seg_indices, fields=[export.PPG_FIELD])
        except Exception as e:                              # noqa: BLE001 (log + skip)
            print(f"  WARN: failed to read {seg_path}: {e}", flush=True)
            return {}
        return {idx: sig.get(export.PPG_FIELD, np.array([]))
                for idx, sig in loaded.items()}
    return ppg_loader


def _subject_source_map(records) -> dict:
    """subj_name -> 'MIMIC'/'VitalDB' (explicit Source, else name-suffix fallback)."""
    out = {}
    for rec in records:
        name = rec["Subj_Name"]
        if "Source" in rec:
            out[name] = rec["Source"]
        elif name not in out:
            out[name] = "MIMIC" if name.rsplit("_", 1)[-1] == "0" else "VitalDB"
    return out


def main():
    ap = argparse.ArgumentParser(description="Build PPG-only PulseDB split cache.")
    ap.add_argument("--pulsedb-dir", required=True, type=Path,
                    help="Root with MIMIC/, VitalDB/, Info_Files/")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "data" / "raw")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    info_dir = args.pulsedb_dir / "Info_Files"
    mimic_dir = _resolve_subdir(args.pulsedb_dir, "MIMIC", "PulseDB_MIMIC")
    vital_dir = _resolve_subdir(args.pulsedb_dir, "VitalDB", "PulseDB_Vital")
    if not info_dir.exists():
        ap.error(f"Info_Files not found: {info_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_subjects: dict[str, set] = {}
    train_cache = None

    for info_name, split in export.INFO_TO_SPLIT.items():
        info_path = info_dir / info_name
        out_path = args.out_dir / f"{split}.npz"
        if not info_path.exists():
            print(f"SKIP {split}: {info_name} absent", flush=True)
            continue
        if out_path.exists():                              # resume after wall-time kill
            print(f"SKIP {split}: {out_path} already exists", flush=True)
            continue

        t0 = time.time()
        print(f"Building {split} from {info_name} ...", flush=True)
        records = export.load_info_file_h5py(info_path)
        loader = _make_ppg_loader(mimic_dir, vital_dir, _subject_source_map(records))
        cache = export.assemble_ppg_only(records, loader)
        export.save_split_cache(args.out_dir, split, cache)
        split_subjects[split] = set(cache["subjects"].tolist())
        print(f"  {split}: {len(cache['X'])} segments, "
              f"{len(split_subjects[split])} subjects [{time.time() - t0:.0f}s]", flush=True)
        if split == "train":
            train_cache = cache

    # Carve a subject-disjoint validation split from train.
    if train_cache is not None and not (args.out_dir / "validation.npz").exists():
        tr, va = export.carve_validation(
            train_cache, val_fraction=args.val_fraction, seed=args.seed)
        export.save_split_cache(args.out_dir, "train", tr)      # overwrite: train minus val
        export.save_split_cache(args.out_dir, "validation", va)
        split_subjects["train"] = set(tr["subjects"].tolist())
        split_subjects["validation"] = set(va["subjects"].tolist())
        print(f"  carved validation: train={len(tr['X'])} val={len(va['X'])} "
              f"segments", flush=True)

    # Leakage gate (ANTIPATTERNS 3): train/validation/calfree/aami_test disjoint.
    guarded = {k: v for k, v in split_subjects.items()
               if k in {"train", "validation", "calfree", "aami_test"}}
    subject_disjoint_check(guarded)
    print(f"VERIFIED subject-disjoint across {sorted(guarded)}", flush=True)
    print("Cache build complete.", flush=True)


if __name__ == "__main__":
    main()
