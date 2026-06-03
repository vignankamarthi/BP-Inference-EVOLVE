"""Tests for framework.eval. Spec: FRAMEWORK.md Section 3 + decision 2."""
import json
from pathlib import Path
import pytest
from framework import eval as feval


def test_module_imports():
    assert callable(feval.evaluate_program)
    assert callable(feval.subset_transfer_experiment)
    assert callable(feval.compute_subset_transfer_results)


# ---------- evaluate_program ----------

def _write_fake_result(run_dir: Path, val_bal_acc: float = 0.5):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "fake",
        "best_val_metrics": {
            "balanced_acc": val_bal_acc,
            "macro_f1": val_bal_acc * 0.95,
            "per_class_pr": {"NP": (0.5, 0.5), "AP": (0.5, 0.5), "HP": (0.5, 0.5)},
            "confusion_3x3": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            "auc_ovr": 0.7,
            "ece": 0.08,
        },
        "history": [
            {"epoch": 0, "train_loss": 1.0, "train_bal_acc": 0.5,
             "val_bal_acc": val_bal_acc * 0.9, "val_macro_f1": 0.45},
            {"epoch": 1, "train_loss": 0.9, "train_bal_acc": 0.55,
             "val_bal_acc": val_bal_acc, "val_macro_f1": 0.5},
        ],
        "param_count": 50_000,
        "train_seconds": 120.0,
        "inference_seconds": 1.5,
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2))


def test_evaluate_program_reads_result_json(tmp_path: Path):
    _write_fake_result(tmp_path, val_bal_acc=0.55)
    fv = feval.evaluate_program(run_dir=tmp_path)
    for key in ("balanced_acc", "macro_f1", "ece", "param_count",
                "auc_ovr", "generalization_gap"):
        assert key in fv
    assert fv["balanced_acc"] == 0.55
    assert fv["param_count"] == 50_000


def test_evaluate_program_raises_when_result_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        feval.evaluate_program(run_dir=tmp_path)


def test_evaluate_program_computes_generalization_gap(tmp_path: Path):
    _write_fake_result(tmp_path, val_bal_acc=0.5)
    fv = feval.evaluate_program(run_dir=tmp_path)
    # history's best val_bal_acc was 0.5 at epoch 1 with train_bal_acc 0.55
    assert fv["generalization_gap"] == pytest.approx(0.55 - 0.5)


# ---------- subset_transfer_experiment ----------

def test_subset_transfer_experiment_generates_expected_count(tmp_path: Path, sample_program_spec):
    manifest = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5, 10, 15, 20, 25],
        n_seeds=3,
        out_dir=tmp_path,
    )
    assert len(manifest["experiments"]) == 15
    assert manifest["k_grid"] == [5, 10, 15, 20, 25]
    assert manifest["n_seeds"] == 3


def test_subset_transfer_experiment_writes_specs_and_runs(tmp_path: Path, sample_program_spec):
    manifest = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5, 10],
        n_seeds=2,
        out_dir=tmp_path,
    )
    for exp in manifest["experiments"]:
        run_dir = Path(exp["run_dir"])
        spec_path = run_dir / "spec.json"
        run_py = run_dir / "run.py"
        assert spec_path.exists()
        assert run_py.exists()


def test_subset_transfer_experiment_injects_subset_size_and_seed(tmp_path: Path, sample_program_spec):
    feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5, 10],
        n_seeds=2,
        out_dir=tmp_path,
    )
    spec_5_s0 = json.loads((tmp_path / "subset_K05_s0" / "spec.json").read_text())
    assert spec_5_s0["data"]["subset_size"] == 5
    assert spec_5_s0["data"]["subset_seed"] == 0
    spec_10_s1 = json.loads((tmp_path / "subset_K10_s1" / "spec.json").read_text())
    assert spec_10_s1["data"]["subset_size"] == 10
    assert spec_10_s1["data"]["subset_seed"] == 1


def test_subset_transfer_experiment_writes_manifest(tmp_path: Path, sample_program_spec):
    manifest = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5],
        n_seeds=1,
        out_dir=tmp_path,
    )
    manifest_on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest_on_disk == manifest


# ---------- compute_subset_transfer_results ----------

def test_compute_results_aggregates_per_k(tmp_path: Path, sample_program_spec):
    manifest = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5, 10],
        n_seeds=2,
        out_dir=tmp_path,
    )
    # Fake-write result.json into each experiment dir.
    accs = {(5, 0): 0.45, (5, 1): 0.47,
            (10, 0): 0.51, (10, 1): 0.50}
    for exp in manifest["experiments"]:
        _write_fake_result(Path(exp["run_dir"]),
                            val_bal_acc=accs[(exp["K"], exp["seed_idx"])])
    table = feval.compute_subset_transfer_results(
        manifest_path=tmp_path / "manifest.json",
        full_baseline_val_acc=0.52)
    assert 5 in table["per_k"]
    assert 10 in table["per_k"]
    assert table["per_k"][5]["mean"] == pytest.approx(0.46, abs=0.01)
    assert table["per_k"][10]["mean"] == pytest.approx(0.505, abs=0.01)
    # K=10 is within tolerance 0.02 of full_baseline 0.52, K=5 is not.
    assert table["chosen_k"] == 10


def test_compute_results_missing_full_baseline(tmp_path: Path, sample_program_spec):
    manifest = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5],
        n_seeds=1,
        out_dir=tmp_path,
    )
    _write_fake_result(Path(manifest["experiments"][0]["run_dir"]), val_bal_acc=0.45)
    table = feval.compute_subset_transfer_results(
        manifest_path=tmp_path / "manifest.json")
    assert table["chosen_k"] is None
    assert table["full_baseline"] is None
    assert 5 in table["per_k"]
