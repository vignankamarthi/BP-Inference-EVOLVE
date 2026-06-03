"""TDD: PPG-only PulseDB cache exporter.

Synthetic HDF5 fixtures replicate the PulseDB MATLAB v7.3 layout (confirmed on
the cluster by the prior project). NEVER reads the real PulseDB (ANTIPATTERNS 12).

PulseDB layout:
  segment file  /Subj_Wins/<field>  -> (1,N) h5py refs -> (1,1250) float64
  info file     /<subset>/<field>   -> (1,N) h5py refs; strings deref to
                                       (len,1) uint16, floats to (1,1) float64
"""
import numpy as np
import pytest

from bp_inference import data, export, splits

h5py = pytest.importorskip("h5py")


# --- synthetic HDF5 builders (mirror the real PulseDB struct-array layout) ---

def _make_segment_file(path, ppg_segments, field="PPG_Record"):
    """ppg_segments: list of 1D arrays. Writes /Subj_Wins/<field> ref array."""
    with h5py.File(str(path), "w") as f:
        refs = f.create_group("#refs#")
        sw = f.create_group("Subj_Wins")
        ref_array = np.empty((1, len(ppg_segments)), dtype=h5py.ref_dtype)
        for i, seg in enumerate(ppg_segments):
            ds = refs.create_dataset(f"r{i}", data=np.asarray(seg, float).reshape(1, -1))
            ref_array[0, i] = ds.ref
        sw.create_dataset(field, data=ref_array)
    return path


def _make_info_file(path, records, subset_name="Train_Subset"):
    """records: list of {Subj_Name, Subj_SegIDX, Seg_SBP, Seg_DBP, Source}."""
    string_fields = {"Subj_Name", "Source"}
    fields = list(records[0].keys())
    with h5py.File(str(path), "w") as f:
        refs = f.create_group("#refs#")
        main = f.create_group(subset_name)
        c = 0
        for field in fields:
            ref_array = np.empty((1, len(records)), dtype=h5py.ref_dtype)
            for i, rec in enumerate(records):
                val = rec[field]
                if field in string_fields:
                    arr = np.array([ord(ch) for ch in str(val)], dtype=np.uint16).reshape(-1, 1)
                else:
                    arr = np.array([[float(val)]], dtype=np.float64)
                ds = refs.create_dataset(f"r{c}", data=arr); c += 1
                ref_array[0, i] = ds.ref
            main.create_dataset(field, data=ref_array)
    return path


# --- h5py loader tests ---

def test_load_subject_signals_ppg_only(tmp_path):
    segs = [np.arange(1250.0) + k for k in range(3)]
    p = _make_segment_file(tmp_path / "p000001.mat", segs)
    out = export.load_subject_signals_h5py(p, seg_indices=[1, 3], fields=["PPG_Record"])
    assert set(out) == {1, 3}                       # MATLAB 1-indexed
    np.testing.assert_array_almost_equal(out[1]["PPG_Record"], segs[0])
    np.testing.assert_array_almost_equal(out[3]["PPG_Record"], segs[2])
    assert out[1]["PPG_Record"].ndim == 1


def test_load_subject_signals_out_of_range_skipped(tmp_path):
    p = _make_segment_file(tmp_path / "p000002.mat", [np.zeros(1250)])
    out = export.load_subject_signals_h5py(p, seg_indices=[1, 999], fields=["PPG_Record"])
    assert 1 in out and 999 not in out


def test_load_subject_signals_no_subj_wins_raises(tmp_path):
    p = tmp_path / "bad.mat"
    with h5py.File(str(p), "w") as f:
        f.create_dataset("garbage", data=[1, 2, 3])
    with pytest.raises(ValueError, match="Subj_Wins"):
        export.load_subject_signals_h5py(p, seg_indices=[1], fields=["PPG_Record"])


def test_load_info_file_parses_records(tmp_path):
    recs = [
        {"Subj_Name": "p072634_0", "Subj_SegIDX": 1.0, "Seg_SBP": 120.5, "Seg_DBP": 80.1, "Source": "MIMIC"},
        {"Subj_Name": "p085541_1", "Subj_SegIDX": 5.0, "Seg_SBP": 96.0, "Seg_DBP": 61.0, "Source": "VitalDB"},
    ]
    p = _make_info_file(tmp_path / "Train_Info.mat", recs)
    out = export.load_info_file_h5py(p)
    assert len(out) == 2
    assert out[0]["Subj_Name"] == "p072634_0"
    assert out[0]["Subj_SegIDX"] == 1 and isinstance(out[0]["Subj_SegIDX"], int)
    assert abs(out[1]["Seg_SBP"] - 96.0) < 1e-6
    assert out[1]["Source"] == "VitalDB"


# --- pure assembly tests (no HDF5: inject a fake PPG loader) ---

def _fake_loader(table):
    """table: {subj_name: {seg_idx: ppg_array}} -> ppg_loader(subj, idxs)."""
    def load(subj_name, seg_indices):
        return {i: table[subj_name][i] for i in seg_indices if i in table.get(subj_name, {})}
    return load


def test_assemble_ppg_only_shapes_and_targets():
    recs = [
        {"Subj_Name": "s1", "Subj_SegIDX": 1, "Seg_SBP": 120.0, "Seg_DBP": 80.0},
        {"Subj_Name": "s1", "Subj_SegIDX": 2, "Seg_SBP": 130.0, "Seg_DBP": 85.0},
        {"Subj_Name": "s2", "Subj_SegIDX": 1, "Seg_SBP": 110.0, "Seg_DBP": 70.0},
    ]
    table = {"s1": {1: np.ones(8), 2: np.full(8, 2.0)}, "s2": {1: np.full(8, 3.0)}}
    cache = export.assemble_ppg_only(recs, _fake_loader(table), signal_len=8)
    assert cache["X"].shape == (3, 8, 1) and cache["X"].dtype == np.float32
    assert cache["sbp"].shape == (3,) and cache["dbp"].shape == (3,)
    assert set(cache["subjects"]) == {"s1", "s2"}
    assert sorted(cache["sbp"].tolist()) == [110.0, 120.0, 130.0]


def test_assemble_pads_and_truncates_to_signal_len():
    recs = [
        {"Subj_Name": "s1", "Subj_SegIDX": 1, "Seg_SBP": 120.0, "Seg_DBP": 80.0},  # short
        {"Subj_Name": "s1", "Subj_SegIDX": 2, "Seg_SBP": 121.0, "Seg_DBP": 81.0},  # long
    ]
    table = {"s1": {1: np.ones(3), 2: np.arange(20.0)}}
    cache = export.assemble_ppg_only(recs, _fake_loader(table), signal_len=8)
    assert cache["X"].shape == (2, 8, 1)
    np.testing.assert_array_equal(cache["X"][0, :, 0], [1, 1, 1, 0, 0, 0, 0, 0])  # padded
    np.testing.assert_array_equal(cache["X"][1, :, 0], np.arange(8.0))            # truncated


def test_assemble_drops_nan_labels_and_empty_ppg():
    recs = [
        {"Subj_Name": "s1", "Subj_SegIDX": 1, "Seg_SBP": np.nan, "Seg_DBP": 80.0},  # NaN label
        {"Subj_Name": "s1", "Subj_SegIDX": 2, "Seg_SBP": 120.0, "Seg_DBP": 80.0},   # good
        {"Subj_Name": "s1", "Subj_SegIDX": 3, "Seg_SBP": 130.0, "Seg_DBP": 85.0},   # empty PPG
    ]
    table = {"s1": {1: np.ones(8), 2: np.ones(8), 3: np.array([])}}
    cache = export.assemble_ppg_only(recs, _fake_loader(table), signal_len=8)
    assert cache["X"].shape == (1, 8, 1)
    assert cache["sbp"][0] == 120.0


# --- validation carve + round-trip with data.load_split ---

def test_carve_validation_is_subject_disjoint():
    subs = np.array([f"s{i // 4}" for i in range(40)])     # 10 subjects, 4 segs each
    cache = {"X": np.zeros((40, 8, 1), np.float32),
             "sbp": np.zeros(40, np.float32), "dbp": np.zeros(40, np.float32),
             "subjects": subs}
    tr, va = export.carve_validation(cache, val_fraction=0.3, seed=0)
    tr_subj, va_subj = set(tr["subjects"]), set(va["subjects"])
    assert tr_subj and va_subj
    splits.subject_disjoint_check({"train": tr_subj, "validation": va_subj})
    assert len(tr["X"]) + len(va["X"]) == 40


def test_save_split_cache_round_trips_through_data_loader(tmp_path):
    rng = np.random.default_rng(0)
    cache = {"X": rng.standard_normal((6, 8, 1)).astype(np.float32),
             "sbp": rng.uniform(95, 175, 6).astype(np.float32),
             "dbp": rng.uniform(60, 95, 6).astype(np.float32),
             "subjects": np.array([f"s{i}" for i in range(6)])}
    export.save_split_cache(tmp_path, "calfree", cache)
    X, y, subj = data.load_split(tmp_path, "calfree")
    assert X.shape == (6, 8, 1) and y.shape == (6, 2) and len(subj) == 6
    np.testing.assert_array_almost_equal(y[:, 0], cache["sbp"])
