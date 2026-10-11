"""Fitting, prediction, grouped cross-validation and checkpoints.

Training rules baked in (all from the research plan):

* Splits are by trajectory.  Candidate- or state-level splitting leaks.
* Hyper-parameters and early stopping use a *meta-validation* split carved out
  of the training trajectories; the held-out fold is never used for tuning.
* Normalisation statistics come from training rows only.
* ``predict`` never reads labels.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .data import (DEFAULT_WEIGHTS, MetaDataset, Standardizer, split_by_group, grouped_kfold,
                   utility, utility_h1)
from .layout import FeatureLayout, H2_HEADS
from .models import build_from_config, build_model, num_parameters


@dataclass
class TrainConfig:
    kind: str = "direct"                 # one_step | direct | dynamics
    hidden: int = 32
    dropout: float = 0.1
    latent: int = 32
    update_dim: int = 16
    use_h1_aux: bool = True              # keep True for a matched Direct-vs-Dynamics test
    use_history: bool = True             # one_step only
    lr: float = 3e-3
    weight_decay: float = 1e-3
    max_epochs: int = 200
    batch_states: int = 32
    patience: int = 25
    h1_weight: float = 0.5               # weight of the t+1 loss for direct/dynamics
    rank_weight: float = 0.0             # within-state pairwise loss on LHV2 (off by default)
    huber_delta: float = 1.0
    utility_weights: Tuple[float, float, float] = DEFAULT_WEIGHTS
    grad_clip: float = 1.0
    seed: int = 0
    per_model_norm: bool = False
    val_frac: float = 0.2
    semantic_hash: str = ""              # D15; filled from the data by ``fit``

    def to_dict(self) -> Dict:
        d = dataclasses.asdict(self)
        d["utility_weights"] = list(self.utility_weights)
        return d


@dataclass
class FitResult:
    model: nn.Module
    standardizer: Standardizer
    layout: FeatureLayout
    config: TrainConfig
    history: List[Dict] = field(default_factory=list)
    best_epoch: int = -1
    n_params: int = 0


@dataclass
class Predictions:
    h2: Optional[np.ndarray]             # [N,3] raw units, or None (one-step model)
    h1: Optional[np.ndarray]             # [N,n_h1] raw units, or None
    layout: FeatureLayout
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS
    latent: Optional[np.ndarray] = None  # [3,N,z] for Dynamics

    def score(self) -> np.ndarray:
        """Ranking score: predicted LHV2 if the model has an H=2 head, else the
        predicted one-step utility."""
        if self.h2 is not None:
            return utility(self.h2, self.weights)
        return utility_h1(self.h1, self.layout.h1_heads, self.weights)


# --------------------------------------------------------------------------
def _tensors(arrs: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v[rows])).float() for k, v in arrs.items()}


def _state_batches(state_index: np.ndarray, batch_states: int, rng) -> List[np.ndarray]:
    uniq = np.unique(state_index)
    rng.shuffle(uniq)
    rows_of = {s: np.where(state_index == s)[0] for s in uniq}
    return [np.concatenate([rows_of[s] for s in uniq[i:i + batch_states]])
            for i in range(0, len(uniq), batch_states)]


def _huber(pred, target, delta):
    """Huber on non-NaN targets; mean over observed entries (0 if none)."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return torch.nn.functional.huber_loss(pred[mask], target[mask], delta=delta)


def _pair_loss(pred_u, true_u, state_idx):
    same = state_idx.unsqueeze(0) == state_idx.unsqueeze(1)
    better = (true_u.unsqueeze(1) - true_u.unsqueeze(0)) > 1e-9
    pairs = same & better
    if not pairs.any():
        return pred_u.sum() * 0.0
    diff = pred_u.unsqueeze(1) - pred_u.unsqueeze(0)
    return torch.nn.functional.softplus(-diff[pairs]).mean()


def _loss(model, kind, batch, cfg: TrainConfig, rank_coef, state_idx):
    out = model(batch["state"], batch["cand"], batch["hist"], batch["hist_mask"])
    if kind == "one_step":
        return _huber(out["h1"], batch["y_h1"], cfg.huber_delta)
    loss = _huber(out["h2"], batch["y_h2"], cfg.huber_delta)
    if "h1" in out and cfg.h1_weight > 0:
        loss = loss + cfg.h1_weight * _huber(out["h1"], batch["y_h1"], cfg.huber_delta)
    if cfg.rank_weight > 0:
        loss = loss + cfg.rank_weight * _pair_loss(out["h2"] @ rank_coef,
                                                   batch["y_h2"] @ rank_coef, state_idx)
    return loss


def fit(train: MetaDataset, val: Optional[MetaDataset], cfg: TrainConfig) -> FitResult:
    """Train one model.  With ``val`` -> early stopping on validation loss and
    restore of the best epoch; without it, trains for ``max_epochs``."""
    if cfg.kind != "one_step" and np.isnan(train.y_h2).any():
        raise ValueError("H=2 labels contain NaN")
    if train.semantic_hash:
        cfg = dataclasses.replace(cfg, semantic_hash=train.semantic_hash)
    if val is not None and val.semantic_hash != train.semantic_hash:
        raise ValueError("train and validation data have different semantic hashes")
    torch.manual_seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    norm = Standardizer(per_model=cfg.per_model_norm).fit(train)
    model = build_model(cfg.kind, train.layout, hidden=cfg.hidden, dropout=cfg.dropout,
                        latent=cfg.latent, update_dim=cfg.update_dim,
                        use_h1_aux=cfg.use_h1_aux, use_history=cfg.use_history)
    tr = norm.transform(train)
    va = norm.transform(val) if val is not None else None

    w = np.asarray(cfg.utility_weights, np.float64)
    w = w / w.sum()
    ystd = norm.stats["y2"][1]
    u_scale = float(np.std(train.lhv2(cfg.utility_weights))) or 1.0
    rank_coef = torch.from_numpy((w * ystd / u_scale).astype(np.float32))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    result = FitResult(model=model, standardizer=norm, layout=train.layout, config=cfg,
                       n_params=num_parameters(model))
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        total, nb = 0.0, 0
        for rows in _state_batches(train.state_index, cfg.batch_states, rng):
            batch = _tensors(tr, rows)
            sidx = torch.from_numpy(train.state_index[rows])
            opt.zero_grad()
            loss = _loss(model, cfg.kind, batch, cfg, rank_coef, sidx)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss at epoch %d" % epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(rows)
            nb += len(rows)
        rec = {"epoch": epoch, "train_loss": total / max(nb, 1)}
        if va is not None:
            model.eval()
            with torch.no_grad():
                vb = _tensors(va, np.arange(len(val)))
                vloss = float(_loss(model, cfg.kind, vb, cfg, rank_coef,
                                    torch.from_numpy(val.state_index)))
            rec["val_loss"] = vloss
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = copy.deepcopy(model.state_dict())
                result.best_epoch = epoch
            else:
                bad += 1
        result.history.append(rec)
        if va is not None and bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        result.best_epoch = cfg.max_epochs - 1
    model.eval()
    return result


def predict(fit_result: FitResult, ds: MetaDataset) -> Predictions:
    """Forecast for every row of ``ds``.  Reads inputs only, never labels."""
    if ds.layout.fingerprint() != fit_result.layout.fingerprint():
        raise ValueError("dataset layout differs from the layout the model was trained on")
    want = fit_result.config.semantic_hash
    if want and ds.semantic_hash and ds.semantic_hash != want:
        raise ValueError("dataset semantic hash %s differs from the model's %s (definitions changed)"
                         % (ds.semantic_hash, want))
    norm = fit_result.standardizer
    arrs = norm.transform(ds)
    inputs = {k: arrs[k] for k in ("state", "cand", "hist", "hist_mask")}
    model = fit_result.model
    model.eval()
    with torch.no_grad():
        out = model(*[torch.from_numpy(np.ascontiguousarray(inputs[k])).float()
                      for k in ("state", "cand", "hist", "hist_mask")])
    h2 = norm.inverse_h2(out["h2"].numpy()) if "h2" in out else None
    h1 = norm.inverse_h1(out["h1"].numpy()) if "h1" in out else None
    latent = out["z"].numpy() if "z" in out else None
    return Predictions(h2=h2, h1=h1, layout=ds.layout,
                       weights=fit_result.config.utility_weights, latent=latent)


# --------------------------------------------------------------------------
@dataclass
class CVResult:
    pred: Predictions                     # seed-averaged out-of-fold predictions
    per_seed: List[Predictions]
    fold_of_row: np.ndarray
    fits: List[FitResult]


def cross_validate(ds: MetaDataset, cfg: TrainConfig, *, n_splits: int = 5,
                   seeds: Sequence[int] = (0,), split_seed: int = 0,
                   by: str = "trajectory") -> CVResult:
    """Out-of-fold predictions with disjoint trajectories between train and test.

    Each training fold gives up ``cfg.val_frac`` of its trajectories to
    meta-validation (early stopping).  Predictions of several seeds are averaged.
    """
    n = len(ds)
    per_seed = []
    fold_of_row = np.full(n, -1, dtype=np.int64)
    fits = []  # type: List[FitResult]
    folds = grouped_kfold(ds, n_splits, seed=split_seed, by=by)
    for seed in seeds:
        h2 = np.full((n, 3), np.nan) if cfg.kind != "one_step" else None
        h1 = np.full((n, len(ds.layout.h1_heads)), np.nan)
        for k, (tr_idx, te_idx) in enumerate(folds):
            fold_of_row[te_idx] = k
            train_all = ds.subset(tr_idx)
            try:
                a, b = split_by_group(train_all, cfg.val_frac, seed=seed + 1000 * k, by=by)
                train, val = train_all.subset(a), train_all.subset(b)
            except ValueError:
                train, val = train_all, None
            fit_cfg = dataclasses.replace(cfg, seed=seed)
            res = fit(train, val, fit_cfg)
            fits.append(res)
            p = predict(res, ds.subset(te_idx))
            if h2 is not None:
                h2[te_idx] = p.h2
            if p.h1 is not None:
                h1[te_idx] = p.h1
        per_seed.append(Predictions(h2=h2, h1=h1 if not np.isnan(h1).all() else None,
                                    layout=ds.layout, weights=cfg.utility_weights))
    avg_h2 = np.mean([p.h2 for p in per_seed], axis=0) if per_seed[0].h2 is not None else None
    h1s = [p.h1 for p in per_seed if p.h1 is not None]
    avg_h1 = np.mean(h1s, axis=0) if h1s else None
    return CVResult(pred=Predictions(h2=avg_h2, h1=avg_h1, layout=ds.layout,
                                     weights=cfg.utility_weights),
                    per_seed=per_seed, fold_of_row=fold_of_row, fits=fits)


# --------------------------------------------------------------------------
# Checkpoints (format 2: layout-aware; Sprint 1's format 1 is untouched)
# --------------------------------------------------------------------------
CHECKPOINT_VERSION = 2


def save_fit(fit_result: FitResult, path, *, metadata: Optional[Dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "layout": fit_result.layout.to_dict(),
        "layout_fingerprint": fit_result.layout.fingerprint(),
        "model_config": dict(fit_result.model.config),
        "train_config": fit_result.config.to_dict(),
        "standardizer": fit_result.standardizer.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in fit_result.model.state_dict().items()},
        "best_epoch": int(fit_result.best_epoch),
        "metadata": metadata or {},
    }
    with path.open("xb") as f:           # never overwrite an existing checkpoint
        torch.save(payload, f)
    return path


def load_fit(path, *, expected_layout: Optional[FeatureLayout] = None,
             expected_semantic_hash: Optional[str] = None) -> FitResult:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version %r" % (ck.get("format_version"),))
    layout = FeatureLayout.from_dict(ck["layout"])
    if layout.fingerprint() != ck["layout_fingerprint"]:
        raise ValueError("checkpoint layout is corrupt")
    if expected_layout is not None and expected_layout.fingerprint() != layout.fingerprint():
        raise ValueError("checkpoint was trained on a different feature layout")
    if (expected_semantic_hash is not None
            and ck["train_config"].get("semantic_hash", "") != expected_semantic_hash):
        raise ValueError("checkpoint semantic hash differs from the expected one")
    model = build_from_config(ck["model_config"])
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    tc = dict(ck["train_config"])
    tc["utility_weights"] = tuple(tc["utility_weights"])
    return FitResult(model=model, standardizer=Standardizer.from_dict(ck["standardizer"]),
                     layout=layout, config=TrainConfig(**tc), best_epoch=ck["best_epoch"],
                     n_params=num_parameters(model))


def constant_baseline(train: MetaDataset, ds: MetaDataset) -> Predictions:
    """D13 baseline: predict the group-weighted training mean for every row
    (all candidates tie, so ranking metrics sit at chance by construction)."""
    from .metrics import group_weighted_mean
    h2 = np.tile(group_weighted_mean(train.y_h2, train.group), (len(ds), 1))
    y1 = train.y_h1
    h1 = np.tile(group_weighted_mean(y1, train.group), (len(ds), 1)) if y1.shape[1] else None
    return Predictions(h2=h2, h1=h1, layout=ds.layout)
