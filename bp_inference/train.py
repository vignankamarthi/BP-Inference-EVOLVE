"""Shared regression training harness for the neural BP seed families.

The BP analog of ai4pain.baselines.run_pytorch_model. Every neural seed exposes
a `model_factory(in_channels, T, model_cfg, n_targets) -> nn.Module` and calls
`run_regression_model`. The harness is invariant across families: load, subject
split, z-score inputs, standardize targets, train (robust regression loss),
evaluate under the spec's calibration regime, atomic-write result.json.

result.json contract (kept identical in shape to the AI4Pain case study so
`framework.eval.evaluate_program` reads it unchanged):
  {name, best_val_metrics, final_val_metrics, history, param_count,
   train_seconds, inference_seconds, generalization_gap, device, spec}
`best_val_metrics` is the regression suite from bp_inference.metrics; the
primary "higher is better" key is `aami_margin`.
"""
import json
import time
from pathlib import Path

import numpy as np

from bp_inference import calibration
from bp_inference.data import enforce_ppg_only, load_split
from bp_inference.metrics import full_metric_suite
from bp_inference.splits import (k_subject_subset, mask_for_subjects,
                                 train_val_split_by_subject)


# ----------------------------------------------------------------------------
# Non-torch helpers (unit-tested without a GPU).
# ----------------------------------------------------------------------------
def per_channel_zscore(Xtr: np.ndarray, Xv: np.ndarray):
    """Fit (mean, std) per channel on train, apply to both. (N, T, C) arrays."""
    mean = Xtr.mean(axis=(0, 1), keepdims=True)
    std = Xtr.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((Xtr - mean) / std).astype(np.float32), ((Xv - mean) / std).astype(np.float32)


def add_derivative_channels(X: np.ndarray, derivatives) -> np.ndarray:
    """Append PPG-derived velocity/acceleration channels (VPG/APG) to X.

    X is (N, T, C) with the raw PPG in channel 0. `derivatives` is a subset of
    ['vpg', 'apg']: VPG = d/dt PPG (1st diff), APG = d2/dt2 PPG (2nd diff), each
    length-preserved (prepend the edge so T is unchanged) and stacked as extra
    channels. The SOURCE stays PPG-only (ANTIPATTERNS rule 2); the multi-channel
    result is a LABELED ABLATION arm, never the hero. Empty list -> X unchanged.
    """
    if not derivatives:
        return X
    ppg = X[..., 0]                                  # (N, T)
    chans = [X[..., c] for c in range(X.shape[-1])]  # keep existing channel(s)
    vpg = np.diff(ppg, axis=-1, prepend=ppg[..., :1])
    if "vpg" in derivatives:
        chans.append(vpg)
    if "apg" in derivatives:
        chans.append(np.diff(vpg, axis=-1, prepend=vpg[..., :1]))
    return np.stack(chans, axis=-1).astype(X.dtype)


def standardize_targets(ytr: np.ndarray):
    """Return (mean, std) fit on train targets for stable regression."""
    mean = ytr.mean(axis=0)
    std = ytr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def evaluate_regime(y_true: np.ndarray, y_pred: np.ndarray, subjects,
                    calib_cfg: dict, seed: int = 0):
    """Apply the calibration gene to in-loop validation predictions.

    free        -> predictions used as-is.
    per_subject -> for each subject, a deterministic `cal_fraction` of its
                   segments calibrates an additive offset (true - pred); the
                   offset is applied to the remaining EVAL segments, which alone
                   are scored (ANTIPATTERNS rule 5: offset never sees eval labels).

    Returns (y_true_eval, y_pred_eval).
    """
    regime = (calib_cfg or {}).get("mode", "free")
    if regime == "free":
        return y_true, calibration.calibration_free(y_pred)
    if regime != "per_subject":
        raise ValueError(f"unknown calibration mode {regime!r}")

    cal_fraction = float((calib_cfg or {}).get("cal_fraction", 0.2))
    subjects = np.asarray(subjects).ravel()
    rng = np.random.default_rng(seed)
    cal_mask = np.zeros(len(subjects), dtype=bool)
    for subj in np.unique(subjects):
        idx = np.where(subjects == subj)[0]
        n_cal = max(1, int(round(len(idx) * cal_fraction)))
        if n_cal >= len(idx):                 # keep at least one eval segment
            n_cal = len(idx) - 1 if len(idx) > 1 else 0
        chosen = rng.permutation(idx)[:n_cal]
        cal_mask[chosen] = True
    eval_mask = ~cal_mask

    offsets = calibration.fit_subject_offsets(
        y_true[cal_mask], y_pred[cal_mask], subjects[cal_mask])
    y_pred_adj = calibration.apply_subject_offsets(
        y_pred[eval_mask], subjects[eval_mask], offsets)
    return y_true[eval_mask], y_pred_adj


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(path)


def _loss_fn(name: str, weights=None):
    import torch
    name = (name or "smooth_l1").lower()
    cls = (torch.nn.MSELoss if name in ("mse", "l2")
           else torch.nn.L1Loss if name in ("l1", "mae")
           else torch.nn.SmoothL1Loss)       # robust default (Huber)
    if not weights:
        return cls()
    # Per-target weighting (e.g. SBP-heavy, since SBP SD is the binding AAMI axis).
    w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
    base = cls(reduction="none")

    def weighted(pred, target):
        per = base(pred, target)             # (B, n_targets)
        ww = w.to(per.device)
        return (per * ww).sum() / (ww.sum() * per.shape[0])

    return weighted


def _load_train_val(data_root: Path, spec: dict):
    """Load train split, carve a subject-disjoint validation set, apply the
    optional K-subject subset for fast in-loop fitness.
    """
    signals = tuple(spec.get("data", {}).get("signals", ["ppg"]))
    enforce_ppg_only(signals)
    seed = int(spec.get("training", {}).get("seed", 42))

    X, y, subjects = load_split(data_root, "train", signals=signals)

    data_cfg = spec.get("data", {})
    subset_size = data_cfg.get("subset_size")
    if subset_size:
        chosen = k_subject_subset(subjects, int(subset_size),
                                  int(data_cfg.get("subset_seed", 0)))
        m = mask_for_subjects(subjects, chosen)
        X, y, subjects = X[m], y[m], subjects[m]

    train_subj, val_subj = train_val_split_by_subject(
        subjects, val_fraction=float(data_cfg.get("val_fraction", 0.2)), seed=seed)
    tr = mask_for_subjects(subjects, train_subj)
    va = mask_for_subjects(subjects, val_subj)
    return (X[tr], y[tr], subjects[tr]), (X[va], y[va], subjects[va])


def _batched_predict(model, X, bs: int):
    """Forward `X` through `model` in batches of `bs`, return (N, n_targets) numpy.

    A single forward on a whole split (100k+ segments) OOMs memory-heavy models
    (the U-Net seeds build multi-resolution feature maps + attention). Batching
    the eval forward, like training already is, keeps activation memory bounded.
    """
    import numpy as _np
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), max(1, bs)):
            outs.append(model(X[i:i + bs]).detach().cpu().numpy())
    if not outs:
        return _np.zeros((0, 0), dtype=_np.float64)
    return _np.concatenate(outs, axis=0)


def run_regression_model(model_factory, spec: dict, data_root: Path,
                         out_dir: Path, name_tag: str = "torch") -> dict:
    """Train one neural seed end-to-end and write result.json."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = spec.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    (Xtr, ytr, _), (Xv, yv, sv) = _load_train_val(data_root, spec)
    derivs = spec.get("preprocessing", {}).get("derivatives", [])
    Xtr = add_derivative_channels(Xtr, derivs)
    Xv = add_derivative_channels(Xv, derivs)
    Xtr, Xv = per_channel_zscore(Xtr, Xv)
    t_mean, t_std = standardize_targets(ytr)
    ytr_std = ((ytr - t_mean) / t_std).astype(np.float32)

    device = _device()
    T = Xtr.shape[1]
    model = model_factory(in_channels=Xtr.shape[2], T=T,
                          model_cfg=spec.get("model", {}), n_targets=2).to(device)

    epochs = int(train_cfg.get("epochs", 30))
    bs = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    optim = (torch.optim.AdamW if train_cfg.get("optimizer", "adam").lower() == "adamw"
             else torch.optim.Adam)(model.parameters(), lr=lr)
    loss_fn = _loss_fn(train_cfg.get("loss", "smooth_l1"),
                       weights=train_cfg.get("loss_weights"))
    calib_cfg = spec.get("calibration", {"mode": "free"})

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr_std).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=bs, shuffle=True)

    def predict(Xb):
        # Batched forward (bug-fix): never push a full split through at once.
        return _batched_predict(model, Xb, bs) * t_std + t_mean   # -> mmHg

    history, best_metrics, final_metrics = [], {}, {}
    best_margin = -np.inf
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for xb, yb in loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
            nb += 1

        val_pred = predict(Xv_t)
        yv_eval, vp_eval = evaluate_regime(yv, val_pred, sv, calib_cfg, seed=seed)
        final_metrics = full_metric_suite(yv_eval, vp_eval)
        history.append({
            "epoch": epoch,
            "train_loss": ep_loss / max(nb, 1),
            "val_aami_margin": final_metrics["aami_margin"],
            "val_sbp_mae": final_metrics["sbp_mae"],
            "val_dbp_mae": final_metrics["dbp_mae"],
        })
        if final_metrics["aami_margin"] > best_margin:
            best_margin = final_metrics["aami_margin"]
            best_metrics = final_metrics
        print(f"[{name_tag}] ep {epoch}: loss={ep_loss/max(nb,1):.4f} "
              f"margin={final_metrics['aami_margin']:.3f} "
              f"sbp_mae={final_metrics['sbp_mae']:.2f} "
              f"dbp_mae={final_metrics['dbp_mae']:.2f}", flush=True)

    train_seconds = time.time() - t0
    # generalization gap: train vs val mean MAE (higher => more overfit).
    train_pred = predict(Xtr_t)
    train_metrics = full_metric_suite(ytr, train_pred)
    gen_gap = float((final_metrics["sbp_mae"] + final_metrics["dbp_mae"]) / 2.0
                    - (train_metrics["sbp_mae"] + train_metrics["dbp_mae"]) / 2.0)

    t1 = time.time()
    _ = predict(Xv_t)
    inference_seconds = time.time() - t1

    result = {
        "name": spec.get("name", name_tag),
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "history": history,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "generalization_gap": gen_gap,
        "device": str(device),
        "spec": spec,
    }
    _atomic_write_json(out_dir / "result.json", result)
    print(f"[{name_tag}] best aami_margin: {best_margin:.3f} -> "
          f"{out_dir / 'result.json'}", flush=True)
    return result


def run_from_dir_with_factory(model_factory, run_dir: Path, data_root: Path,
                              name_tag: str) -> dict:
    """Shared run_from_dir body: read spec.json, train, write result.json."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    return run_regression_model(model_factory, spec, Path(data_root), run_dir,
                                name_tag=name_tag)
