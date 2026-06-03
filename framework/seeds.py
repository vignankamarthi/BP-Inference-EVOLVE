"""Multi-seed initialization (BP-Inference-EVOLVE, PPG-only regression).

Returns the 4 seed program specs:
  1. MiniRocket + RidgeCV regressor (the given seed; ROCKET-family random
     convolutional kernels, Dempster et al. 2020/2021).
  2. rU-Net + STFT + multi-head attention (PulseDB SOTA family adapted to a
     single PPG channel; Chen et al. 2025, IEEE JBHI).
  3. Self-attention ResUNet with squeeze-excite (2025, IEEE Sensors Journal).
  4. Lightweight selective state-space (Mamba-style) U-Net (Mamba-UNet 2025).

Every seed is single-channel PPG (ANTIPATTERNS rule 2), outputs two regression
heads (SBP, DBP), carries a `calibration` gene the loop fixes per track, and
decodes with the identity strategy (regression, not argmax). The old basic
ResNet-BiGRU is deliberately absent: it sat at the published PPG-only ceiling.

Calibration gene schema (consumed by bp_inference.train.evaluate_regime):
  {"mode": "free"}                                  -> Track A, no adaptation.
  {"mode": "per_subject", "cal_fraction": <float>}  -> Track B. cal_fraction is
    an EVOLVABLE hyperparameter gene (the loop tweaks it like lr/dropout; it is
    fingerprint-invisible, so it never trips ast_tabu). Default 0.2, sensible
    evolvable band ~0.05-0.30. The rU-Net transfer result that reached
    compliance used 0.10; we do NOT pin it, we let the search find it.
    evaluate_regime self-bounds it per subject (>=1 calibrate, >=1 eval segment),
    so a degenerate value cannot crash a run.

`diversify_population` distributes seeds across N islands, filling islands with
fewer seeds by replicating the assigned seed (the loop's first round of
mutation diversifies them in-place).
"""


def default_seed_specs() -> list[dict]:
    """The 4 PPG-only BP regression seeds (BP-Inference-EVOLVE).

    All are single-channel (PPG), two regression heads (SBP, DBP), with a
    `calibration` gene the loop fixes per track (Track A free, Track B
    per_subject). `decode.strategy = "identity"` (regression, not argmax).
    Seed set: MiniRocket (given) + 3 researched architectures (ANTIPATTERNS 8).
    """
    return [
        {
            # Given seed. ROCKET-family random conv kernels + RidgeCV regressor
            # (Dempster et al. 2020/2021). Fast, neural-adjacent counter-baseline.
            "name": "seed_minirocket",
            "preprocessing": {"normalize": "per_channel_zscore"},
            "feature_extraction": {"family": "minirocket", "num_kernels": 1000,
                                    "kernel_length": 9, "random_state": 42},
            "model": {"family": "ridge_regressor_cv",
                      "alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
            "training": {"loss": "ridge_regression_cv", "seed": 42},
            "calibration": {"mode": "free"},
            "data": {"signals": ["ppg"], "val_fraction": 0.2},
            "decode": {"strategy": "identity"},
        },
        {
            # rU-Net + STFT + multi-head attention, PPG-only single-channel.
            # PulseDB SOTA family (Chen et al. 2025, IEEE JBHI).
            "name": "seed_runet_attn",
            "preprocessing": {"normalize": "per_channel_zscore"},
            "feature_extraction": None,
            "model": {"family": "runet_attn", "base_channels": 32,
                      "use_stft": True, "n_fft": 64, "hop": 16,
                      "num_heads": 4, "dropout": 0.2},
            "training": {"loss": "smooth_l1", "optimizer": "adamw",
                         "lr": 5e-4, "epochs": 40, "batch_size": 64, "seed": 42},
            "calibration": {"mode": "free"},
            "data": {"signals": ["ppg"], "val_fraction": 0.2},
            "decode": {"strategy": "identity"},
        },
        {
            # Self-attention ResUNet with squeeze-excite, PPG-only.
            # PulseDB calibration-free AAMI family (2025, IEEE Sensors J).
            "name": "seed_resunet_sa",
            "preprocessing": {"normalize": "per_channel_zscore"},
            "feature_extraction": None,
            "model": {"family": "resunet_sa", "base_channels": 24,
                      "num_heads": 4, "dropout": 0.2},
            "training": {"loss": "smooth_l1", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 40, "batch_size": 64, "seed": 42},
            "calibration": {"mode": "free"},
            "data": {"signals": ["ppg"], "val_fraction": 0.2},
            "decode": {"strategy": "identity"},
        },
        {
            # Lightweight selective state-space (Mamba-style) U-Net, PPG-only.
            # Frontier seed (Mamba-UNet 2025; BM-BPW EMBC 2025).
            "name": "seed_mamba_ssm",
            "preprocessing": {"normalize": "per_channel_zscore"},
            "feature_extraction": None,
            "model": {"family": "mamba_ssm", "dim": 48, "n_layers": 2,
                      "dropout": 0.2},
            "training": {"loss": "smooth_l1", "optimizer": "adamw",
                         "lr": 5e-4, "epochs": 40, "batch_size": 64, "seed": 42},
            "calibration": {"mode": "free"},
            "data": {"signals": ["ppg"], "val_fraction": 0.2},
            "decode": {"strategy": "identity"},
        },
    ]


def expand_regimes(seed_specs: list[dict], cal_fraction: float = 0.2) -> list[dict]:
    """Expand each seed into its Track A and Track B calibration variant.

    Track A (calibration-free, the hero frontier): calibration={'mode':'free'},
    name '<seed>_free'. Track B (calibration-based): calibration=
    {'mode':'per_subject','cal_fraction':cal_fraction}, name '<seed>_cal'.
    `cal_fraction` is the evolvable Track-B gene (default 0.2; see the gene
    schema docstring above). Inputs are not mutated (deep-copied).

    Returns 2*len(seed_specs) specs, all Track A first then all Track B, so a
    `--array=0-(2n-1)` maps the first half to Track A and the second to Track B.
    """
    import copy

    free, cal = [], []
    for spec in seed_specs:
        base = spec.get("name", "seed")
        a = copy.deepcopy(spec)
        a["name"] = f"{base}_free"
        a["calibration"] = {"mode": "free"}
        free.append(a)
        b = copy.deepcopy(spec)
        b["name"] = f"{base}_cal"
        b["calibration"] = {"mode": "per_subject", "cal_fraction": cal_fraction}
        cal.append(b)
    return free + cal


def diversify_population(seed_specs: list[dict],
                          island_count: int) -> list[list[dict]]:
    """Distribute the seeds across `island_count` islands.

    If island_count >= len(seed_specs): one seed per island, remaining islands
    get a copy of a randomly cycled seed.
    If island_count < len(seed_specs): pack multiple seeds per island in a
    round-robin.

    Returns a list of length island_count; each element is a list of seed
    spec dicts assigned to that island.
    """
    if island_count < 1:
        raise ValueError(f"island_count must be >= 1, got {island_count}")
    if not seed_specs:
        return [[] for _ in range(island_count)]

    islands: list[list[dict]] = [[] for _ in range(island_count)]
    for i, spec in enumerate(seed_specs):
        islands[i % island_count].append(dict(spec))

    # Fill empty islands by copying from the most-populated ones (deterministic).
    n_seeds = len(seed_specs)
    if island_count > n_seeds:
        for j in range(n_seeds, island_count):
            source_idx = j % n_seeds
            islands[j].append(dict(seed_specs[source_idx]))

    return islands
