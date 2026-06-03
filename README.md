# BP-Inference-EVOLVE

PPG-only cuffless blood-pressure estimation on PulseDB (continuous SBP/DBP
regression from photoplethysmography alone), characterised against clinical
compliance standards (AAMI, BHS) under calibration-free and calibration-based
regimes. The study was produced by applying a multi-level **Stochastic
Experiment Loop** framework -- a stochastic, LLM-driven evolutionary discovery
loop -- to this task as its second case study (the first was `AI4Pain-2026`).
The framework itself (under `framework/`) is preserved here as scientific
provenance and as a copyable reference for applying the same loop to future
domains.

The repository is two things at once.

**First, a PPG-only cuffless BP compliance study.** Continuous systolic and
diastolic pressure regression from the single PPG/BVP channel on PulseDB v2.0
(MIMIC-III + VitalDB). The deliverable is a family of *discovered* PPG-only
architectures and a characterisation of their AAMI / BHS compliance under two
regimes: calibration-free (the hard, genuinely-open frontier) and
calibration-based (per-subject). It re-attacks the earlier
`Blood-Pressure-Inference-with-BVP` project, which landed at the published
PPG-only ceiling (~SBP 13.6 / DBP 7.97 mmHg) and could not reach AAMI. ECG is
deliberately excluded: ECG enables pulse-arrival/transit-time, the lever every
AAMI-compliant PulseDB result has relied on, so using it would reproduce known
work rather than push the open PPG-only frontier.

**Second, the second case study of a stochastic evolutionary framework** the
author is developing as a longer-term research artifact. The framework operates
on three levels: Level 0 (population of candidate programs -- architectures,
preprocessing, training recipes -- trained and scored per iteration), Level 1
(FunSearch islands + GENITOR steady-state replacement + multi-objective
Pareto/novelty/failure-aware scoring + rule guards + AST tabu + lineage cap +
migration + coevolutionary critic + mix-ratio-drift meta-stochastic), and
Level 2 (self-introspection that mutates Level 1 on an empirical compound-
detector cadence). The mutation operator is a Claude Code session, not an
external API client.

## Problem and compliance target

- **Input:** PPG / BVP only. Single channel. ECG banned. Demographics off by
  default (available only as a clearly-labeled ablation arm).
- **Output:** SBP and DBP (mmHg), continuous regression.
- **AAMI:** mean error (bias) ME <= 5 mmHg AND standard deviation of error
  SD <= 8 mmHg, for both SBP and DBP. (AAMI bounds ME and SD, not MAE.)
- **BHS:** grade A/B/C/D from cumulative |error| at 5 / 10 / 15 mmHg.
- **The frontier:** the published PPG-only ceiling is SBP 13.9 / DBP 8.5 mmHg
  calibration-free and SBP 9.0 / DBP 5.8 with subject-specific calibration
  (Moulaeifard, Charlton & Strodthoff 2025). None pass AAMI for SBP. PPG-only
  AAMI, especially calibration-free SBP, is open.

## Status

Scaffold complete: generic engine + PPG-only adapter, four seed families, full
TDD suite (279 tests, synthetic data only). No cluster runs yet. Next: wire the
PulseDB split cache on NEU Explorer, render the seed batch across both regimes,
and begin the evolutionary search. Results tables will be added here per regime
and per target as runs land.

## Model families (seeds)

- MiniRocket + RidgeCV regressor (random convolutional kernels; Dempster 2021)
- rU-Net + STFT + multi-head attention (time-frequency U-Net; Chen et al. 2025)
- Self-attention ResUNet with squeeze-excite (2025, IEEE Sensors J)
- Mamba / selective state-space U-Net (Mamba-UNet 2025)

All single-channel (PPG), two regression heads (SBP, DBP). The loop recombines
and mutates from these via island migration and cross-family grafting.

## Paper scope

The paper presents the discovered PPG-only architectures and their AAMI / BHS
compliance findings under both regimes, including honest reporting of the
calibration-free frontier where AAMI is unreachable. **The paper does NOT
discuss the Stochastic Experiment Loop**; the framework is preserved here only
as a research artifact for application to future projects, not as the paper's
subject.

## Repo layout

```
framework/             Three-level evolutionary framework (model-agnostic)
bp_inference/          PPG-only adapter: data loaders, subject-disjoint splits,
                       calibration regimes, AAMI/BHS/IEEE-1708 metrics, the
                       regression train harness, and the four seed families
scripts/               Manual cluster helpers (Vignan runs by hand)
  run_array.slurm      Iteration array job (per-batch evolutionary search)
  run_submission.slurm Single-task runner (H200, .venv 3.11)
tests/                 Mirror layout: unit/framework/, unit/bp_inference/, integration/
experiments/           Per-run artifact directories (spec.json + run.py + result.json)
ledger/                SQLite database of all child runs (gitignored)
data/                  PulseDB split cache (gitignored, cluster-only ~963 GB)
```

Static identity: `CLAUDE.md`. Dynamic memory: `MEMORY.md`. Hard rules:
`ANTIPATTERNS.md`. Phased plan: `PLAN.md`. (Internal docs, gitignored.)

## Hard constraints

- PPG / BVP only. ECG banned as a model input. Demographics off by default.
- No subject leakage; subject-disjoint splits only; preprocessing fits on train
  subjects only. Calibration-free and calibration-based reported separately.
- AAMI bounds ME and SD (not MAE); always report MAE/RMSE/ME/SD/r/R2 + BHS grade.
- No external data or pretrained weights; models train from random init.
- No commits or pushes from Claude (`/Commit-Initiation` plans only).
- No programmatic cluster invocations from framework Python (Vignan runs
  `ssh`/`sbatch`/`git pull` by hand; git is the only Mac<->cluster bridge).
- No external LLM SDKs (`anthropic`, `openai`, etc. NOT in `requirements.txt`).
  The mutation operator is the Claude Code session driving this repo.
- Strict TDD: a failing test in `tests/...` before any implementation.

## Hardware

NEU Explorer cluster, Python 3.11 venv, torch 2.5.1+cu121, 1x H200 GPU per job,
8h SLURM time limit, 8 concurrent job slots. PPG segments are short (10 s at
125 Hz = 1250 samples), so the seeds are small relative to the cap.
