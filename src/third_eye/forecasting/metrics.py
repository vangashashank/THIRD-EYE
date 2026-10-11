"""Evaluation metrics for candidate forecasting (E1, E2, E3, E12).

Conventions
-----------
* Everything is computed on *held-out* rows (held-out trajectories).
* Rows are candidates; ``state_index`` identifies the sibling set (the K
  candidates that share a parent).  Ranking only makes sense within siblings.
* Ties are handled by expected credit (no arbitrary tie-breaking).
* Confidence intervals bootstrap over **states** (clusters), never over rows.

Pooled vs within-state Spearman
-------------------------------
Pooled Spearman over all rows can be high just because the model predicts
which *states* are easy (state-level offset), without being able to pick the
better candidate.  ``within_state_spearman`` removes each state's mean from
both prediction and truth first, so it only credits sibling discrimination.
Report both; decide in advance which one Gate 2 uses (we recommend
within-state).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def state_groups(state_index: np.ndarray) -> List[np.ndarray]:
    """Row-index arrays, one per distinct id (ids may be ints or strings)."""
    _, state_index = np.unique(np.asarray(state_index), return_inverse=True)
    order = np.argsort(state_index, kind="stable")
    sorted_idx = state_index[order]
    cuts = np.where(np.diff(sorted_idx) != 0)[0] + 1
    return list(np.split(order, cuts))


def center_by_state(x: np.ndarray, state_index: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    for rows in state_groups(state_index):
        out[rows] -= out[rows].mean()
    return out


def _nan_if_const(a, b):
    return np.ptp(a) == 0 or np.ptp(b) == 0


def pooled_spearman(y_true, y_pred) -> float:
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    if a.size < 3 or _nan_if_const(a, b):
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def within_state_spearman(y_true, y_pred, state_index) -> float:
    return pooled_spearman(center_by_state(y_true, state_index),
                           center_by_state(y_pred, state_index))


def mean_state_kendall(y_true, y_pred, state_index) -> float:
    """Mean over states of Kendall tau-b between sibling truth and prediction.
    States where either side is constant are skipped."""
    taus = []
    for rows in state_groups(state_index):
        a, b = np.asarray(y_true)[rows], np.asarray(y_pred)[rows]
        if len(rows) < 2 or _nan_if_const(a, b):
            continue
        taus.append(stats.kendalltau(a, b)[0])
    return float(np.mean(taus)) if taus else float("nan")


def _best_set(v, tol=1e-12):
    return np.isclose(v, v.max(), rtol=0.0, atol=tol)


def top1_accuracy(y_true, y_pred, state_index) -> Tuple[float, float]:
    """(accuracy, chance) with expected credit under ties.

    Credit for a state = fraction of the predicted-best set that is also in the
    true-best set.  ``chance`` = expected credit of a uniformly random pick.
    """
    credit, chance = [], []
    for rows in state_groups(state_index):
        t, p = np.asarray(y_true, float)[rows], np.asarray(y_pred, float)[rows]
        tb, pb = _best_set(t), _best_set(p)
        credit.append((tb & pb).sum() / pb.sum())
        chance.append(tb.mean())
    return float(np.mean(credit)), float(np.mean(chance))


def _expected_dcg(rel, score):
    """DCG of ordering by ``score`` descending, tied blocks share the mean
    discount (expected DCG under random tie-breaking)."""
    n = len(rel)
    disc = 1.0 / np.log2(np.arange(n) + 2.0)
    order = np.argsort(-score, kind="stable")
    s_sorted = score[order]
    dcg, i = 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and np.isclose(s_sorted[j + 1], s_sorted[i], rtol=0.0, atol=1e-12):
            j += 1
        block = order[i:j + 1]
        dcg += rel[block].mean() * disc[i:j + 1].sum()
        i = j + 1
    return dcg


def ndcg(y_true, y_pred, state_index) -> float:
    """Mean NDCG over states; relevance = truth minus the state's worst candidate
    (so it is non-negative).  States whose siblings are all equal count as 1."""
    vals = []
    for rows in state_groups(state_index):
        t, p = np.asarray(y_true, float)[rows], np.asarray(y_pred, float)[rows]
        rel = t - t.min()
        ideal = _expected_dcg(rel, rel)
        vals.append(1.0 if ideal <= 0 else _expected_dcg(rel, p) / ideal)
    return float(np.mean(vals))


def selection_regret(y_true, y_pred, state_index) -> float:
    """Mean (best true value - expected true value of the chosen candidate)."""
    out = []
    for rows in state_groups(state_index):
        t, p = np.asarray(y_true, float)[rows], np.asarray(y_pred, float)[rows]
        pb = _best_set(p)
        out.append(t.max() - t[pb].mean())
    return float(np.mean(out))


def ranking_report(y_true, y_pred, state_index) -> Dict[str, float]:
    top1, chance = top1_accuracy(y_true, y_pred, state_index)
    return {
        "pooled_spearman": pooled_spearman(y_true, y_pred),
        "within_state_spearman": within_state_spearman(y_true, y_pred, state_index),
        "mean_state_kendall": mean_state_kendall(y_true, y_pred, state_index),
        "top1": top1,
        "top1_chance": chance,
        "ndcg": ndcg(y_true, y_pred, state_index),
        "regret": selection_regret(y_true, y_pred, state_index),
        "n_states": float(len(state_groups(state_index))),
    }


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, names: Sequence[str],
                      state_index: Optional[np.ndarray] = None,
                      baseline: Optional[np.ndarray] = None,
                      sign_tol: float = 1e-9) -> Dict[str, Dict[str, float]]:
    """Per-head MAE/RMSE, skill vs a constant baseline, sign accuracy,
    calibration slope (OLS of truth on prediction; 1 = calibrated), Spearman."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    base = y_true.mean(0) if baseline is None else np.asarray(baseline, float)
    out = {}
    for j, name in enumerate(names):
        t, p = y_true[:, j], y_pred[:, j]
        err = p - t
        mse_base = float(np.mean((t - base[j]) ** 2))
        nz = np.abs(t) > sign_tol
        slope = float("nan")
        if np.ptp(p) > 0:
            slope = float(np.polyfit(p, t, 1)[0])
        rec = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(math.sqrt(np.mean(err ** 2))),
            "skill_vs_constant": float(1.0 - np.mean(err ** 2) / mse_base) if mse_base > 0 else float("nan"),
            "sign_accuracy": float(np.mean(np.sign(p[nz]) == np.sign(t[nz]))) if nz.any() else float("nan"),
            "calibration_slope": slope,
            "pooled_spearman": pooled_spearman(t, p),
        }
        if state_index is not None:
            rec["within_state_spearman"] = within_state_spearman(t, p, state_index)
        out[name] = rec
    return out


# --------------------------------------------------------------------------
# Cluster bootstrap over states
# --------------------------------------------------------------------------
Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def _make_sampler(state_index, cluster_index=None):
    """Return ``draw(rng) -> (rows, new_state_index)``.

    Default resamples whole STATES.  With ``cluster_index`` (e.g. lineage group,
    D02/D13) it resamples whole clusters, which is the honest unit when states
    inside a cluster are correlated.  Copies of a state/cluster get fresh ids so
    within-state metrics never merge duplicates.
    """
    state_index = np.asarray(state_index)
    if cluster_index is None:
        groups = state_groups(state_index)

        def draw(rng):
            pick = rng.randint(0, len(groups), size=len(groups))
            rows = np.concatenate([groups[i] for i in pick])
            return rows, np.repeat(np.arange(len(pick)), [len(groups[i]) for i in pick])
        return draw, len(groups)
    clusters = state_groups(np.asarray(cluster_index))
    big = int(state_index.max()) + 1

    def draw(rng):
        pick = rng.randint(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[i] for i in pick])
        copy = np.concatenate([np.full(len(clusters[i]), c) for c, i in enumerate(pick)])
        return rows, state_index[rows] + copy * big
    return draw, len(clusters)


MIN_RELIABLE_CLUSTERS = 10


def bootstrap_ci(metric: Metric, y_true, y_pred, state_index, *, n_boot: int = 500,
                 seed: int = 0, alpha: float = 0.05,
                 cluster_index=None) -> Dict[str, float]:
    """Point estimate and percentile CI.  Resamples states, or whole clusters
    (lineage groups) when ``cluster_index`` is given.  ``warning`` is set when
    there are too few resampling units for the interval to mean much."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    draw, n_units = _make_sampler(state_index, cluster_index)
    rng = np.random.RandomState(seed)
    point = metric(y_true, y_pred, np.asarray(state_index))
    vals = []
    for _ in range(n_boot):
        rows, ns = draw(rng)
        v = metric(y_true[rows], y_pred[rows], ns)
        if np.isfinite(v):
            vals.append(v)
    out = {"point": float(point), "lo": float("nan"), "hi": float("nan"),
           "n_boot_valid": len(vals), "n_units": n_units}
    if vals:
        lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["lo"], out["hi"] = float(lo), float(hi)
    if n_units < MIN_RELIABLE_CLUSTERS:
        out["warning"] = ("only %d resampling units: interval is unreliable, treat as preliminary"
                          % n_units)
    return out


def paired_bootstrap_diff(metric: Metric, y_true, pred_a, pred_b, state_index, *,
                          n_boot: int = 500, seed: int = 0, alpha: float = 0.05,
                          cluster_index=None) -> Dict[str, float]:
    """metric(A) - metric(B) on the same resampled states.  ``p_two_sided`` is
    twice the smaller tail mass of the bootstrap difference around 0."""
    y_true, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    draw, n_units = _make_sampler(state_index, cluster_index)
    rng = np.random.RandomState(seed)
    point = metric(y_true, a, np.asarray(state_index)) - metric(y_true, b, np.asarray(state_index))
    diffs = []
    for _ in range(n_boot):
        rows, ns = draw(rng)
        d = metric(y_true[rows], a[rows], ns) - metric(y_true[rows], b[rows], ns)
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return {"diff": float(point), "lo": float("nan"), "hi": float("nan"),
                "p_two_sided": float("nan"), "n_units": n_units}
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    out = {"diff": float(point), "lo": float(lo), "hi": float(hi),
           "p_two_sided": float(min(p, 1.0)), "n_units": n_units}
    if n_units < MIN_RELIABLE_CLUSTERS:
        out["warning"] = "only %d resampling units: treat as preliminary" % n_units
    return out


# convenient scalar metrics for bootstrapping
def m_within_state_spearman(t, p, s): return within_state_spearman(t, p, s)
def m_top1(t, p, s): return top1_accuracy(t, p, s)[0]
def m_ndcg(t, p, s): return ndcg(t, p, s)
def m_regret(t, p, s): return selection_regret(t, p, s)


# --------------------------------------------------------------------------
# E1: immediate vs future mismatch
# --------------------------------------------------------------------------
def immediate_future_mismatch(u_h1, u_h2, state_index, margin: float = 0.0) -> Dict[str, float]:
    """Share of states where the best-immediate candidate is not the best H=2 one.

    ``mismatch_rate``: best sets (under ties) are disjoint.
    ``meaningful_mismatch_rate``: additionally the H=2 value lost by following
    the immediate winner exceeds ``margin`` (use it to ignore noise-level flips).
    """
    mism, meaningful = [], []
    for rows in state_groups(np.asarray(state_index)):
        a, b = np.asarray(u_h1, float)[rows], np.asarray(u_h2, float)[rows]
        ab, bb = _best_set(a), _best_set(b)
        flip = not (ab & bb).any()
        mism.append(flip)
        loss = b.max() - b[ab].mean()
        meaningful.append(flip and loss > margin)
    return {"mismatch_rate": float(np.mean(mism)),
            "meaningful_mismatch_rate": float(np.mean(meaningful)),
            "n_states": float(len(mism))}


# --------------------------------------------------------------------------
# D13: constant baseline and per-group reporting
# --------------------------------------------------------------------------
def group_weighted_mean(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Mean of per-group means: every lineage group counts once, however many
    rows it has.  This is the "predict the training average" baseline."""
    y = np.asarray(y, dtype=np.float64)
    means = [y[group == g].mean(axis=0) for g in np.unique(group)]
    return np.mean(means, axis=0)


def per_group_report(y_true, y_pred, state_index, group) -> Dict[str, Dict[str, float]]:
    """Ranking report per lineage group (shows whether one group carries the result)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for g in np.unique(group):
        rows = np.where(group == g)[0]
        out[str(g)] = ranking_report(y_true[rows], y_pred[rows], np.asarray(state_index)[rows])
    return out
