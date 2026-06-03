"""Tests for framework.seeds (BP-Inference-EVOLVE PPG-only regression seeds)."""
import pytest
from framework import seeds


def test_module_imports():
    assert callable(seeds.default_seed_specs)
    assert callable(seeds.diversify_population)


def test_default_seed_specs_returns_four_seeds():
    """MiniRocket (given) + 3 researched architectures (ANTIPATTERNS 8)."""
    assert len(seeds.default_seed_specs()) == 4


def test_default_seed_families():
    families = {s["model"]["family"] for s in seeds.default_seed_specs()}
    assert families == {"ridge_regressor_cv", "runet_attn",
                        "resunet_sa", "mamba_ssm"}


def test_old_basic_resnet_bigru_absent():
    """The prior ResNet-BiGRU sat at the PPG-only ceiling; it is not a seed."""
    families = {s["model"]["family"] for s in seeds.default_seed_specs()}
    assert "bigru" not in families
    assert "1d_cnn" not in families


def test_every_seed_is_ppg_only_with_calibration_gene():
    for spec in seeds.default_seed_specs():
        assert spec["data"]["signals"] == ["ppg"]      # ANTIPATTERNS rule 2
        assert spec["calibration"]["mode"] in ("free", "per_subject")
        assert spec["decode"]["strategy"] == "identity"  # regression, not argmax


def test_each_seed_has_required_top_level_keys():
    for spec in seeds.default_seed_specs():
        for key in ("name", "preprocessing", "feature_extraction", "model",
                    "training", "decode", "calibration", "data"):
            assert key in spec


def test_no_classification_loss_in_seeds():
    for spec in seeds.default_seed_specs():
        assert "ce_class_balanced" not in spec["training"].get("loss", "")


def test_minirocket_seed_uses_random_kernel_features():
    specs = {s["name"]: s for s in seeds.default_seed_specs()}
    s = specs["seed_minirocket"]
    assert s["feature_extraction"]["family"] == "minirocket"
    assert s["model"]["family"] == "ridge_regressor_cv"
    assert len(s["model"]["alphas"]) >= 4


def test_neural_seeds_have_no_feature_extraction():
    specs = {s["name"]: s for s in seeds.default_seed_specs()}
    for name in ("seed_runet_attn", "seed_resunet_sa", "seed_mamba_ssm"):
        assert specs[name]["feature_extraction"] is None


# ---------- diversify_population ----------

def test_diversify_population_returns_one_list_per_island():
    distributed = seeds.diversify_population(seeds.default_seed_specs(), island_count=4)
    assert len(distributed) == 4


def test_diversify_population_each_island_nonempty():
    distributed = seeds.diversify_population(seeds.default_seed_specs(), island_count=4)
    for island in distributed:
        assert len(island) >= 1


def test_diversify_population_rejects_invalid_island_count():
    with pytest.raises(ValueError):
        seeds.diversify_population(seeds.default_seed_specs(), island_count=0)


def test_diversify_population_handles_more_islands_than_seeds():
    distributed = seeds.diversify_population(seeds.default_seed_specs(), island_count=8)
    assert len(distributed) == 8


def test_diversify_population_handles_fewer_islands_than_seeds():
    distributed = seeds.diversify_population(seeds.default_seed_specs(), island_count=2)
    assert len(distributed) == 2
    assert sum(len(i) for i in distributed) == 4


def test_diversify_population_empty_seeds():
    distributed = seeds.diversify_population([], island_count=3)
    assert len(distributed) == 3
    assert all(len(i) == 0 for i in distributed)


# ---------- expand_regimes (Track A free / Track B per-subject) ----------

def test_expand_regimes_doubles_the_population():
    out = seeds.expand_regimes(seeds.default_seed_specs())
    assert len(out) == 8                                  # 4 seeds x 2 regimes


def test_expand_regimes_each_seed_gets_free_and_cal_variant():
    out = seeds.expand_regimes(seeds.default_seed_specs())
    names = {s["name"] for s in out}
    for base in ("seed_minirocket", "seed_runet_attn",
                 "seed_resunet_sa", "seed_mamba_ssm"):
        assert f"{base}_free" in names
        assert f"{base}_cal" in names


def test_expand_regimes_sets_calibration_genes():
    out = {s["name"]: s for s in seeds.expand_regimes(seeds.default_seed_specs())}
    assert out["seed_minirocket_free"]["calibration"] == {"mode": "free"}
    cal = out["seed_minirocket_cal"]["calibration"]
    assert cal["mode"] == "per_subject"
    assert cal["cal_fraction"] == 0.2                     # default, evolvable gene


def test_expand_regimes_custom_cal_fraction():
    out = {s["name"]: s for s in
           seeds.expand_regimes(seeds.default_seed_specs(), cal_fraction=0.1)}
    assert out["seed_runet_attn_cal"]["calibration"]["cal_fraction"] == 0.1


def test_expand_regimes_preserves_family_and_ppg_only():
    out = seeds.expand_regimes(seeds.default_seed_specs())
    for s in out:
        assert s["data"]["signals"] == ["ppg"]
        assert s["model"]["family"] in {"ridge_regressor_cv", "runet_attn",
                                        "resunet_sa", "mamba_ssm"}


def test_expand_regimes_does_not_mutate_inputs():
    base = seeds.default_seed_specs()
    seeds.expand_regimes(base)
    assert all(s["calibration"] == {"mode": "free"} for s in base)  # untouched
