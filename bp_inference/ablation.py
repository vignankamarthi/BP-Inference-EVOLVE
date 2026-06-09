"""Ablation factor definitions + config enumeration for the strategic screen.

Each FACTOR is a list of (label, override) levels; an `override` is a partial
spec deep-merged into a backbone's baseline. `enumerate_lever_configs` crosses
survivor backbones x one factor's levels x replicate seeds into ready-to-render
specs. Carry-forward: the caller fixes the winning level into each baseline
before screening the next factor (cheap -> expensive: input_rep -> rf -> loss).
"""
import copy


# (label, override) per factor. The override is deep-merged into the backbone
# baseline spec, so it sets only the genes it names and preserves their siblings.
FACTORS = {
    "input_rep": [
        ("raw", {}),
        ("vpg", {"preprocessing": {"derivatives": ["vpg"]}}),
        ("apg", {"preprocessing": {"derivatives": ["apg"]}}),
        ("vpg_apg", {"preprocessing": {"derivatives": ["vpg", "apg"]}}),
    ],
    "loss": [
        ("equal", {}),
        ("sbp2x", {"training": {"loss_weights": [2.0, 1.0]}}),
        ("sbp3x", {"training": {"loss_weights": [3.0, 1.0]}}),
    ],
    # 'rf' is family-specific (extended depth/dilation/kernel_len differs per
    # backbone), so its levels are supplied per-survivor by the caller, not here.
}


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge `override` into a copy of `base`, one level into gene dicts."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **copy.deepcopy(v)}
        else:
            out[k] = copy.deepcopy(v)
    return out


def enumerate_lever_configs(backbones, factor_levels, n_seeds: int = 3,
                            base_seed: int = 42) -> list[dict]:
    """Cross `backbones` x `factor_levels` x `n_seeds` into ready-to-render specs.

    `backbones`: list of (name, baseline_spec). `factor_levels`: list of
    (label, override). Each output spec is the baseline deep-merged with the
    level override, named `lever_<backbone>_<label>_s<seed_idx>`, with
    training.seed set per replicate so the noise floor is measurable.
    """
    specs = []
    for bname, base in backbones:
        for label, override in factor_levels:
            merged = _merge(base, override)
            for s in range(n_seeds):
                spec = copy.deepcopy(merged)
                spec["name"] = f"lever_{bname}_{label}_s{s}"
                spec.setdefault("training", {})["seed"] = base_seed + s
                specs.append(spec)
    return specs


# Family-specific "extended receptive field" bumps (the RF lever is NOT a
# universal override: each backbone enlarges temporal context differently --
# depth for conv/resnet, more decomposition levels for wavelet, a longer global
# kernel for s4 -- so it is built per backbone, not stored in FACTORS).
# Only families with a gene that genuinely widens the receptive field. NOT
# resunet_sa / runet_attn: their depth is hardcoded and they already attend
# globally (a bottleneck self-attention spans the full window), so there is no
# RF knob to turn -- rf_levels returns std-only for them.
RF_EXTENDED = {
    "wavelet_net": {"levels": 7},          # was 5: more multi-resolution depth
    "tcn": {"n_blocks": 9},                # was 6: exponentially longer dilated RF
    "xresnet1d": {"n_stages": 5},          # was 4: deeper downsample stack
    "s4": {"kernel_len": 384},             # was 128: longer global conv kernel
    "inception1d": {"n_blocks": 5},        # was 3: more downsampling depth
    "mamba_ssm": {"n_layers": 4},          # was 2: deeper selective scan
}


def rf_levels(family: str):
    """RF lever levels for one backbone: ('std', no-op) + ('ext', model-gene
    bump that widens the receptive field). The override targets `model`, so the
    deep-merge keeps the family and the other model genes."""
    levels = [("std", {})]
    ext = RF_EXTENDED.get(family)
    if ext:
        levels.append(("ext", {"model": dict(ext)}))
    return levels
