"""Build the PPG-only split cache that bp_inference.data.load_split reads.

Data lives ONLY on the cluster (PulseDB v2.0, ~963 GB). This module reads
PulseDB's MATLAB v7.3 (HDF5) Info + segment files directly and writes one
`.npz` per split in the contract of `bp_inference.data`:

    <out_dir>/<split>.npz : X (N, 1250, 1) PPG-only float32, sbp, dbp, subjects.

ANTIPATTERNS rule 2 (PPG ONLY): only the `PPG_Record` field is ever requested
from the segment files. ECG_F and ABP_Raw are never read, so the produced cache
cannot physically contain a non-PPG channel. (ABP in particular would leak the
target: SBP/DBP are peaks/troughs of the ABP waveform.)

ANTIPATTERNS rule 12 (synthetic test data): the h5py loaders are exercised in
tests against synthetic HDF5 fixtures, never the real PulseDB.

Loader layout (confirmed on the cluster by the prior project's probe):
  segment file  /Subj_Wins/<field>  -> Dataset (1, N) of h5py refs;
                each ref -> Dataset (1, 1250) float64
  info file     /<subset>/<field>   -> Dataset (1, N) of h5py refs;
                string fields deref to (len, 1) uint16 (chr); floats to (1,1) f64

The heavy h5py reading is ported (validated) from the prior project's
`scripts/generate_subsets.py`; the assembly is rewritten PPG-only.
"""
from pathlib import Path

import numpy as np

from bp_inference.splits import (
    mask_for_subjects,
    train_val_split_by_subject,
)

SIGNAL_LEN = 1250          # PulseDB segments: 10 s at 125 Hz
PPG_FIELD = "PPG_Record"   # raw, unnormalized PPG (locked: no pre-extraction norm)

# PulseDB Info filename -> our split name. CalBased/CalFree/AAMI are PulseDB's
# own subject-level subsets; `validation` is carved from `train` downstream.
INFO_TO_SPLIT: dict[str, str] = {
    "Train_Info.mat": "train",
    "CalFree_Test_Info.mat": "calfree",
    "CalBased_Test_Info.mat": "calbased",
    "AAMI_Test_Info.mat": "aami_test",
    "AAMI_Cal_Info.mat": "aami_cal",
}


# --------------------------------------------------------------------------
# h5py loaders (ported from the prior project's generate_subsets.py)
# --------------------------------------------------------------------------

def load_subject_signals_h5py(path, seg_indices, fields=(PPG_FIELD,)):
    """Load specific signal fields for specific segments from one subject file.

    Reads only the requested fields/segments via HDF5 object references, avoiding
    full struct deserialization. `seg_indices` are 1-indexed (MATLAB convention).

    Returns {matlab_idx: {field: np.ndarray (1D float64)}}. Out-of-range indices
    are omitted; missing fields map to empty arrays.
    """
    import h5py

    result: dict[int, dict[str, np.ndarray]] = {}
    with h5py.File(str(path), "r") as f:
        if "Subj_Wins" not in f:
            raise ValueError(f"No 'Subj_Wins' in {path}: {list(f.keys())}")
        sw = f["Subj_Wins"]

        n_segs = None
        for field_name in fields:
            if field_name in sw:
                ds = sw[field_name]
                n_segs = ds.shape[1] if ds.ndim == 2 else ds.shape[0]
                break
        if n_segs is None:
            return result

        for matlab_idx in seg_indices:
            py_idx = matlab_idx - 1
            if py_idx < 0 or py_idx >= n_segs:
                continue
            seg_data: dict[str, np.ndarray] = {}
            for field_name in fields:
                if field_name not in sw:
                    seg_data[field_name] = np.array([], dtype=np.float64)
                    continue
                ds = sw[field_name]
                ref = ds[0, py_idx] if ds.ndim == 2 else ds[py_idx]
                seg_data[field_name] = f[ref][()].flatten().astype(np.float64)
            result[matlab_idx] = seg_data
    return result


def load_info_file_h5py(info_path):
    """Load a PulseDB Info .mat (v7.3 HDF5) into a list of segment records.

    Each record: {Subj_Name (str), Subj_SegIDX (int), [Seg_SBP, Seg_DBP (float)],
    [Source (str)]}. Subj_Name + Subj_SegIDX are required; the rest are optional.
    """
    import h5py

    records: list[dict] = []
    with h5py.File(str(info_path), "r") as f:
        main_key = next((k for k in f.keys() if not k.startswith("#")), None)
        if main_key is None:
            raise ValueError(f"No data key found in {info_path}")
        main = f[main_key]

        for field in ("Subj_Name", "Subj_SegIDX"):
            if field not in main:
                raise ValueError(f"Missing required field '{field}' in {info_path}")

        ds_name = main["Subj_Name"]
        n = ds_name.shape[1] if ds_name.ndim == 2 else ds_name.shape[0]

        def read_string(ds, idx):
            ref = ds[0, idx] if ds.ndim == 2 else ds[idx]
            return "".join(chr(c) for c in f[ref][()].flatten())

        def read_float(ds, idx):
            ref = ds[0, idx] if ds.ndim == 2 else ds[idx]
            return float(f[ref][()].flatten()[0])

        has = {k: (k in main) for k in ("Source", "Seg_SBP", "Seg_DBP")}
        ds_segidx = main["Subj_SegIDX"]
        for i in range(n):
            rec = {
                "Subj_Name": read_string(ds_name, i),
                "Subj_SegIDX": int(read_float(ds_segidx, i)),
            }
            if has["Source"]:
                rec["Source"] = read_string(main["Source"], i)
            if has["Seg_SBP"]:
                rec["Seg_SBP"] = read_float(main["Seg_SBP"], i)
            if has["Seg_DBP"]:
                rec["Seg_DBP"] = read_float(main["Seg_DBP"], i)
            records.append(rec)
    return records


# --------------------------------------------------------------------------
# Pure assembly (PPG-only) -- fully unit-testable with an injected loader
# --------------------------------------------------------------------------

def assemble_ppg_only(info_records, ppg_loader, signal_len=SIGNAL_LEN):
    """Assemble the PPG-only contract from Info records + a PPG segment loader.

    `ppg_loader(subj_name, seg_indices) -> {seg_idx: np.ndarray}` returns the raw
    PPG waveform per requested segment for one subject. Segments with an empty
    PPG array or a NaN SBP/DBP label are dropped. Each waveform is right-zero
    padded or truncated to `signal_len`.

    Returns {X (M, signal_len, 1) float32, sbp (M,) float32, dbp (M,) float32,
    subjects (M,) str}.
    """
    from collections import defaultdict

    by_subject: dict[str, list[int]] = defaultdict(list)
    meta: dict[tuple, tuple] = {}
    for rec in info_records:
        s, idx = rec["Subj_Name"], int(rec["Subj_SegIDX"])
        by_subject[s].append(idx)
        meta[(s, idx)] = (rec.get("Seg_SBP", np.nan), rec.get("Seg_DBP", np.nan))

    X_rows: list[np.ndarray] = []
    sbp: list[float] = []
    dbp: list[float] = []
    subjects: list[str] = []

    for subj_name, seg_indices in by_subject.items():
        loaded = ppg_loader(subj_name, seg_indices)
        for idx in seg_indices:
            ppg = loaded.get(idx)
            if ppg is None or len(ppg) == 0:
                continue
            s_sbp, s_dbp = meta[(subj_name, idx)]
            if not np.isfinite(s_sbp) or not np.isfinite(s_dbp):
                continue
            row = np.zeros(signal_len, dtype=np.float32)
            n = min(len(ppg), signal_len)
            row[:n] = np.asarray(ppg, dtype=np.float32)[:n]
            X_rows.append(row)
            sbp.append(float(s_sbp))
            dbp.append(float(s_dbp))
            subjects.append(subj_name)

    X = (np.stack(X_rows)[:, :, None] if X_rows
         else np.zeros((0, signal_len, 1), dtype=np.float32))
    return {
        "X": X.astype(np.float32),
        "sbp": np.asarray(sbp, dtype=np.float32),
        "dbp": np.asarray(dbp, dtype=np.float32),
        "subjects": np.asarray(subjects),
    }


def carve_validation(train_cache, val_fraction=0.2, seed=0):
    """Split a train cache into (train, validation) by subject (rule 3 + 4).

    Validation is carved from TRAIN subjects only; CalFree / AAMI_Test stay blind.
    """
    subjects = train_cache["subjects"]
    train_subj, val_subj = train_val_split_by_subject(
        subjects, val_fraction=val_fraction, seed=seed)
    tr = mask_for_subjects(subjects, train_subj)
    va = mask_for_subjects(subjects, val_subj)

    def _sub(mask):
        return {k: (v[mask] if isinstance(v, np.ndarray) else v)
                for k, v in train_cache.items()}

    return _sub(tr), _sub(va)


def save_split_cache(out_dir, split, cache):
    """Atomically write `<out_dir>/<split>.npz` in the data.load_split contract."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{split}.npz"
    tmp = out_dir / f"_{split}_tmp.npz"     # ends in .npz so np.savez won't re-append
    np.savez(
        str(tmp),
        X=cache["X"].astype(np.float32),
        sbp=cache["sbp"].astype(np.float32),
        dbp=cache["dbp"].astype(np.float32),
        subjects=cache["subjects"],
    )
    tmp.replace(final)
    return final
