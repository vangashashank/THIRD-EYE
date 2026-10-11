"""Candidate selection, offline policy replay, go/no-go gates, gated primal-dual.

Offline replay vs recursive runs
--------------------------------
``replay_policy`` scores a selection rule on *cached* meta-labels: for every
state, which candidate would the policy have picked, and what was that
candidate's true H=1 / H=2 outcome?  It is cheap and available as soon as the
meta-dataset exists.  It is NOT the T-step recursive experiment (E4), which
needs the harness to actually apply the chosen updates; replay only measures
one-step-ahead selection quality.

Ties in a policy's scores are resolved by expected value (uniform over the
tied set), which is what a seeded random tie-break gives on average.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import DEFAULT_WEIGHTS, MetaDataset, utility
from .layout import CANDIDATE, H2_HEADS
from .metrics import (bootstrap_ci, immediate_future_mismatch, state_groups, top1_accuracy,
                      within_state_spearman)


def _best(v: np.ndarray) -> np.ndarray:
    return np.isclose(v, v.max(), rtol=0.0, atol=1e-12)


def argmax_candidate(scores: Sequence[float], rng: Optional[np.random.RandomState] = None) -> int:
    """Index of the best score for ONE state (deterministic: lowest index; with
    ``rng``: uniform among ties).  Use this inside the recursive loop."""
    s = np.asarray(scores, dtype=np.float64)
    ties = np.where(_best(s))[0]
    if rng is None or len(ties) == 1:
        return int(ties[0])
    return int(rng.choice(ties))


# --------------------------------------------------------------------------
# Baseline scorers (one score per row; higher = preferred)
# --------------------------------------------------------------------------
def random_scores(ds: MetaDataset) -> np.ndarray:
    return np.zeros(len(ds))                       # all tied -> expected-value random pick


def heuristic_scores(ds: MetaDataset, column: str, higher_is_better: bool = True) -> np.ndarray:
    """Simple quality heuristic, e.g. ``verifier_pass_rate`` (+) or ``pre_update_nll`` (-)."""
    cols = ds.layout.columns(CANDIDATE)
    if column not in cols:
        raise KeyError("candidate column %r not in layout" % column)
    v = ds.cand[:, cols.index(column)].astype(np.float64)
    return v if higher_is_better else -v


def greedy_immediate_oracle_scores(ds: MetaDataset, weights=DEFAULT_WEIGHTS) -> np.ndarray:
    """Uses the TRUE t+1 outcome (expensive in deployment): the greedy baseline."""
    return ds.h1_utility(weights)


def future_oracle_scores(ds: MetaDataset, weights=DEFAULT_WEIGHTS) -> np.ndarray:
    """Uses the TRUE H=2 outcome: the upper bound for any forecaster."""
    return ds.lhv2(weights)


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------
@dataclass
class ReplayResult:
    lhv2: np.ndarray            # per state: expected true LHV2 of the chosen candidate
    h1_value: np.ndarray        # per state: expected true one-step utility of the chosen
    regret: np.ndarray          # per state: best true LHV2 - chosen
    harmful: np.ndarray         # per state: P(chosen one-step utility < 0)
    catastrophic: np.ndarray    # per state: P(chosen one-step utility < -catastrophic)
    retention_loss: np.ndarray  # per state: expected max(0, -true H=2 retention delta) of chosen
    state_index: np.ndarray     # dense 0..S-1 id per entry above


def replay_policy(ds: MetaDataset, scores: np.ndarray, weights=DEFAULT_WEIGHTS,
                  catastrophic: float = 0.02) -> ReplayResult:
    lhv, h1u = ds.lhv2(weights), ds.h1_utility(weights)
    out = {k: [] for k in ("lhv2", "h1", "regret", "harm", "cat", "ret")}
    for rows in state_groups(ds.state_index):
        chosen = _best(np.asarray(scores, float)[rows])
        c = rows[chosen]
        out["lhv2"].append(lhv[c].mean())
        out["h1"].append(h1u[c].mean())
        out["regret"].append(lhv[rows].max() - lhv[c].mean())
        out["harm"].append((h1u[c] < 0).mean())
        out["cat"].append((h1u[c] < -catastrophic).mean())
        out["ret"].append(np.clip(-ds.y_h2[c, 2], 0, None).mean())
    arr = {k: np.asarray(v) for k, v in out.items()}
    return ReplayResult(arr["lhv2"], arr["h1"], arr["regret"], arr["harm"], arr["cat"], arr["ret"],
                        np.arange(len(arr["lhv2"])))


def _mean_ci(x: np.ndarray, n_boot=1000, seed=0, alpha=0.05) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    boots = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def compare_policies(ds: MetaDataset, scores_by_policy: Dict[str, np.ndarray],
                     weights=DEFAULT_WEIGHTS, n_boot: int = 1000, seed: int = 0,
                     catastrophic: float = 0.02) -> List[Dict]:
    """One row per policy: mean chosen LHV2 / regret / harmful-update rate with
    state-level bootstrap CIs, plus top-1 and within-state Spearman vs LHV2."""
    rows = []
    lhv = ds.lhv2(weights)
    for name, sc in scores_by_policy.items():
        r = replay_policy(ds, sc, weights, catastrophic)
        m, lo, hi = _mean_ci(r.lhv2, n_boot, seed)
        hm, hlo, hhi = _mean_ci(r.harmful, n_boot, seed)
        top1, chance = top1_accuracy(lhv, sc, ds.state_index)
        rows.append({
            "policy": name,
            "mean_chosen_lhv2": m, "lhv2_lo": lo, "lhv2_hi": hi,
            "mean_regret": float(r.regret.mean()),
            "harmful_update_rate": hm, "harmful_lo": hlo, "harmful_hi": hhi,
            "catastrophic_rate": float(r.catastrophic.mean()),
            "mean_retention_loss": float(r.retention_loss.mean()),
            "top1": top1, "top1_chance": chance,
            "within_state_spearman": within_state_spearman(lhv, sc, ds.state_index),
        })
    return rows


# --------------------------------------------------------------------------
# Gates (executable versions of the plan's go/no-go table)
# --------------------------------------------------------------------------
def tradeoff_fraction(ds: MetaDataset, delta: float = 0.0) -> float:
    """Share of states with a real gain-vs-retention trade-off: some candidate
    improves target at H=2 (> 0) while its retention change is < -delta."""
    flags = []
    for rows in state_groups(ds.state_index):
        y = ds.y_h2[rows]
        flags.append(bool(np.any((y[:, 0] > 0) & (y[:, 2] < -delta))))
    return float(np.mean(flags))


def gate1(ds: MetaDataset, weights=DEFAULT_WEIGHTS, mismatch_min: float = 0.20,
          harmful_min: float = 0.15, margin: float = 0.0) -> Dict:
    """Gate 1 (phenomenon exists): >= 20% immediate-vs-H=2 disagreement OR
    >= 15% of candidate updates harmful on at least one of target/OOD/retention.

    Harmful is evaluated on the H=2 deltas (and on H=1 where available); both
    are reported so the team can state which one the gate used.
    """
    mm = immediate_future_mismatch(ds.h1_utility(weights), ds.lhv2(weights), ds.state_index, margin)
    harm_h2 = float(np.mean((ds.y_h2 < 0).any(axis=1)))
    harm_h1 = float(np.mean((np.nan_to_num(ds.y_h1, nan=0.0) < 0).any(axis=1))) if ds.y_h1.size else float("nan")
    passed = bool(mm["mismatch_rate"] >= mismatch_min or harm_h2 >= harmful_min)
    return {"passed": passed, "mismatch": mm, "harmful_any_head_h2": harm_h2,
            "harmful_any_head_h1": harm_h1,
            "thresholds": {"mismatch_min": mismatch_min, "harmful_min": harmful_min}}


def gate2(report: Dict[str, float], ci: Optional[Dict[str, Dict[str, float]]] = None,
          rho_min: float = 0.30, top1_min: float = 0.45,
          rho_key: str = "within_state_spearman") -> Dict:
    """Gate 2 (H=2 is learnable) on held-out-trajectory metrics.

    ``passed_point`` follows the plan literally (point estimates).  ``passed_ci``
    is stricter: the lower CI bound of Spearman must be > 0 and of top-1 above
    chance, so a lucky small sample cannot pass.  The plan does not say which
    Spearman it means; ``rho_key`` makes the choice explicit.
    """
    ok_point = bool(report[rho_key] >= rho_min and report["top1"] >= top1_min)
    out = {"passed_point": ok_point, "rho_key": rho_key, "rho": report[rho_key],
           "top1": report["top1"], "top1_chance": report["top1_chance"]}
    if ci is not None:
        out["passed_ci"] = bool(ok_point and ci["rho"]["lo"] > 0 and ci["top1"]["lo"] > report["top1_chance"])
    return out


def dynamics_gate(direct: Dict[str, float], dynamics: Dict[str, float],
                  tol: float = 0.01) -> Dict:
    """Dynamics value gate on ranking metrics: promote if it improves at least
    one primary metric and degrades none by more than ``tol``.  (Trajectory and
    transfer axes are judged on recursive/transfer runs, outside this function.)"""
    improved, degraded = [], []
    for key, higher_is_better in (("within_state_spearman", True), ("top1", True),
                                  ("ndcg", True), ("regret", False)):
        gain = dynamics[key] - direct[key]
        if not higher_is_better:
            gain = -gain
        if gain > tol:
            improved.append(key)
        elif gain < -tol:
            degraded.append(key)
    return {"promote": bool(improved and not degraded), "improved": improved,
            "degraded": degraded}


def pd_activation_gate(gate2_passed: bool, retention_sign_accuracy: float,
                       retention_within_spearman: float, tradeoff_frac: float) -> Dict:
    """Primal-dual activation: Gate 2 passed AND the retention head has signal
    (sign accuracy > 60% or positive rank correlation) AND >= ~10% of states
    show a real gain-vs-retention trade-off.  Otherwise skip PD entirely."""
    signal = bool(retention_sign_accuracy > 0.60 or retention_within_spearman > 0)
    trade = bool(tradeoff_frac >= 0.10)
    return {"activate": bool(gate2_passed and signal and trade),
            "gate2_passed": bool(gate2_passed), "retention_head_has_signal": signal,
            "tradeoffs_common_enough": trade}


# --------------------------------------------------------------------------
# Gated primal-dual selector (Third Eye-PD).  Off unless pd_activation_gate passes.
# --------------------------------------------------------------------------
@dataclass
class PDConfig:
    eps_forget: float = 0.0     # allowed predicted forgetting (score units, >= 0)
    eps_unstable: float = 0.0   # allowed predicted "instability"; see constraint_values
    eta: float = 0.5            # dual step size
    lambda_max: float = 50.0
    use_instability: bool = False


def constraint_values(h2_pred: np.ndarray, use_instability: bool = False) -> np.ndarray:
    """[K, C] SIGNED constraint values g (constraint is g <= eps) from predicted H=2 deltas.

    forgetting   = -(predicted retention change)      (> 0 means retention drops)
    instability  = -min(predicted target change, predicted OOD change)
                   (a *proposed* definition; the plan does not fix one)

    Signed on purpose: with slack (g < eps) the dual variable can decay.  A
    clipped, non-negative g with eps = 0 could only ever ratchet lambda upward.
    """
    cols = [-h2_pred[:, 2]]
    if use_instability:
        cols.append(-np.minimum(h2_pred[:, 0], h2_pred[:, 1]))
    return np.stack(cols, axis=1)


class PrimalDualSelector:
    """Maximise predicted utility subject to g_j(candidate) <= eps_j, with
    adaptive dual variables instead of hand-set penalty weights.

        choose  argmax_k  u_k - sum_j lambda_j g_jk
        update  lambda_j <- clip(lambda_j + eta (g_j(chosen) - eps_j), 0, lambda_max)

    Duals rise while chosen updates exceed a constraint and decay (down to 0)
    while they sit inside it, so unconstrained behaviour is recovered whenever
    the constraint is not binding.  In the long run the *average* constraint
    value of the chosen updates is driven toward eps.
    """

    def __init__(self, cfg: PDConfig):
        self.cfg = cfg
        self.eps = np.array([cfg.eps_forget] + ([cfg.eps_unstable] if cfg.use_instability else []))
        self.lam = np.zeros(len(self.eps))

    def select(self, utilities: np.ndarray, g: np.ndarray) -> int:
        return argmax_candidate(utilities - g @ self.lam)

    def update(self, g_chosen: np.ndarray) -> None:
        self.lam = np.clip(self.lam + self.cfg.eta * (g_chosen - self.eps), 0.0, self.cfg.lambda_max)


def pd_replay(ds: MetaDataset, h2_pred: np.ndarray, scores: np.ndarray, cfg: PDConfig,
              eps_violation: Optional[float] = None) -> Dict:
    """Stream states in (trajectory, state_id) order through the selector using
    only *predicted* quantities, then score the choices with TRUE labels.
    A choice "violates" when its true forgetting exceeds ``eps_violation``
    (default: the configured ``eps_forget``)."""
    eps_violation = cfg.eps_forget if eps_violation is None else eps_violation
    order = sorted(state_groups(ds.state_index),
                   key=lambda r: (str(ds.trajectory[r[0]]), int(ds.state_id[r[0]])))
    sel = PrimalDualSelector(cfg)
    lhv = ds.lhv2()
    chosen_lhv, viol, mag, lams = [], [], [], []
    for rows in order:
        g = constraint_values(h2_pred[rows], cfg.use_instability)
        k = sel.select(scores[rows], g)
        sel.update(g[k])
        lams.append(sel.lam.copy())
        row = rows[k]
        chosen_lhv.append(lhv[row])
        true_forget = -float(ds.y_h2[row, 2])
        viol.append(true_forget > eps_violation)
        mag.append(max(0.0, true_forget))
    return {"mean_chosen_lhv2": float(np.mean(chosen_lhv)),
            "true_violation_rate": float(np.mean(viol)),
            "mean_true_forgetting": float(np.mean(mag)),
            "final_lambda": lams[-1].tolist(), "lambda_trace": np.asarray(lams)}


def fixed_penalty_replay(ds: MetaDataset, h2_pred: np.ndarray, scores: np.ndarray,
                         penalty: float, eps_violation: float = 0.0) -> Dict:
    """Fixed scalarisation baseline: argmax u_k - penalty * predicted forgetting
    (same signed g as the primal-dual selector, constant penalty)."""
    lhv = ds.lhv2()
    chosen, viol, mag = [], [], []
    for rows in state_groups(ds.state_index):
        g = constraint_values(h2_pred[rows])[:, 0]
        k = argmax_candidate(scores[rows] - penalty * g)
        row = rows[k]
        chosen.append(lhv[row])
        f = -float(ds.y_h2[row, 2])
        viol.append(f > eps_violation)
        mag.append(max(0.0, f))
    return {"penalty": penalty, "mean_chosen_lhv2": float(np.mean(chosen)),
            "true_violation_rate": float(np.mean(viol)), "mean_true_forgetting": float(np.mean(mag))}
