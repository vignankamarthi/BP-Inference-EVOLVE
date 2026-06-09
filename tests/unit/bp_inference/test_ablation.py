"""Tests for the ablation factor enumeration, bp_inference.ablation.

A FACTOR is a list of (label, override) levels; enumerate_lever_configs crosses
survivor backbones x one factor's levels x replicate seeds into ready-to-render
specs. Replication is mandatory (iter-1 deltas were within the ~0.1 mmHg noise
floor), so every (backbone, level) is run at n_seeds distinct seeds.
"""
from bp_inference.ablation import FACTORS, enumerate_lever_configs

BASE = {
    "name": "screen_tcn",
    "preprocessing": {"normalize": "per_channel_zscore"},
    "feature_extraction": None,
    "model": {"family": "tcn", "channels": 48},
    "training": {"loss": "smooth_l1", "optimizer": "adamw", "lr": 5e-4,
                 "epochs": 40, "seed": 42},
    "calibration": {"mode": "per_subject", "cal_fraction": 0.2},
    "data": {"signals": ["ppg"], "val_fraction": 0.2},
    "decode": {"strategy": "identity"},
}


def test_input_rep_factor_has_four_levels():
    assert [lab for lab, _ in FACTORS["input_rep"]] == ["raw", "vpg", "apg", "vpg_apg"]


def test_enumerate_crosses_backbones_levels_seeds():
    backbones = [("tcn", BASE),
                 ("resunet_sa", {**BASE, "model": {"family": "resunet_sa"}})]
    specs = enumerate_lever_configs(backbones, FACTORS["input_rep"], n_seeds=3)
    assert len(specs) == 2 * 4 * 3                          # backbones x levels x seeds
    assert len({s["name"] for s in specs}) == len(specs)   # all names unique


def test_override_merges_without_clobbering_siblings():
    specs = enumerate_lever_configs([("tcn", BASE)], FACTORS["input_rep"], n_seeds=1)
    vpg = next(s for s in specs if s["name"].endswith("_vpg_s0"))
    assert vpg["preprocessing"]["derivatives"] == ["vpg"]
    assert vpg["preprocessing"]["normalize"] == "per_channel_zscore"   # sibling kept


def test_seeds_vary_per_replicate_and_base_training_preserved():
    specs = enumerate_lever_configs([("tcn", BASE)], [("raw", {})], n_seeds=3)
    assert sorted(s["training"]["seed"] for s in specs) == [42, 43, 44]
    assert all(s["training"]["loss"] == "smooth_l1" for s in specs)


def test_loss_factor_applies_weights():
    specs = enumerate_lever_configs([("tcn", BASE)], FACTORS["loss"], n_seeds=1)
    sbp = next(s for s in specs if "sbp2x" in s["name"])
    assert sbp["training"]["loss_weights"] == [2.0, 1.0]
    assert sbp["training"]["loss"] == "smooth_l1"          # base loss form preserved


def test_rf_levels_std_then_ext():
    from bp_inference.ablation import rf_levels
    lv = rf_levels("tcn")                                  # tcn has a tunable RF (n_blocks)
    assert [lab for lab, _ in lv] == ["std", "ext"]
    assert lv[0][1] == {}                                  # std = baseline, no override
    assert "model" in lv[1][1]                             # ext overrides model genes


def test_rf_levels_std_only_when_no_knob():
    from bp_inference.ablation import rf_levels
    lv = rf_levels("resunet_sa")                           # fixed/global RF, no knob
    assert [lab for lab, _ in lv] == ["std"]               # no ext arm rendered


def test_rf_ext_is_family_specific():
    from bp_inference.ablation import rf_levels
    assert rf_levels("wavelet_net")[1][1]["model"] != rf_levels("tcn")[1][1]["model"]


def test_rf_ext_merge_preserves_family_and_siblings():
    from bp_inference.ablation import _merge, rf_levels
    base = {**BASE, "model": {"family": "xresnet1d", "base_channels": 32}}
    merged = _merge(base, rf_levels("xresnet1d")[1][1])
    assert merged["model"]["family"] == "xresnet1d"        # family preserved
    assert merged["model"]["base_channels"] == 32          # sibling preserved
    assert merged["model"]["n_stages"] == 5                # RF gene set
