"""Integration: subset-transfer experiment exercises bp_inference.splits + framework.eval.

The loop runs in-fitness on a K-subject subset only after this experiment
validates K via HIP-B. This asserts the generation half of the workflow
(subset_transfer_experiment produces a manifest + 15 experiment dirs). Actual
cluster training is HIP-D / HIP-E / HIP-F. Result aggregation has its own unit
tests in test_eval.py.
"""
from pathlib import Path

from bp_inference import splits
from framework import eval as feval


def test_subset_transfer_experiment_generates_manifest_and_dirs(tmp_path: Path,
                                                                  sample_program_spec):
    manifest = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5, 10, 15, 20, 25],
        n_seeds=3,
        out_dir=tmp_path,
    )
    assert "experiments" in manifest
    assert len(manifest["experiments"]) == 15
    assert (tmp_path / "manifest.json").exists()
    for exp in manifest["experiments"]:
        run_dir = Path(exp["run_dir"])
        assert (run_dir / "spec.json").exists()
        assert (run_dir / "run.py").exists()


def test_subset_subjects_disjoint_from_val():
    """The K-subject fitness subset must come from TRAIN subjects, never val."""
    all_subjects = [f"S{i:02d}" for i in range(41)]
    train, val = splits.train_val_split_by_subject(
        all_subjects, val_fraction=0.12, seed=0)        # ~5 val subjects
    chosen = splits.k_subject_subset(train, k=10, seed=0)
    assert set(chosen).isdisjoint(set(val))
    assert set(chosen).issubset(set(train))
