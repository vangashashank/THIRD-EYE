"""Meta-dataset: record validation, arrays, grouped splits, normalisation.

Record format (one JSON object per *candidate*; JSONL file)
-----------------------------------------------------------
::

    {
      "model": "qwen3-4b", "task": "math",
      "trajectory_id": "qwen3-4b/math/chain-007",  # one recursive chain; the CV unit
      "state_id": 2, "candidate_id": 1,
      "state":     {<every STATE column of the layout>: float},      # shared by the K candidates
      "candidate": {<every CANDIDATE column of the layout>: float},
      "history": {"values": {"<channel>": [H floats], ...},          # oldest -> newest,
                  "mask": [H floats]},                                # real entries right-aligned
      "labels": {"delta_target_h2": f, "delta_ood_h2": f, "delta_retention_h2": f,
                 "delta_target_h1": f, ["delta_ood_h1": f, "delta_retention_h1": f]}
    }

Deltas are *improvements* (after - before for accuracies; before - after for
losses), so larger is always better.  ``labels`` are read only when building
supervision arrays; they are never part of the input arrays.

Why trajectories are the split unit: the K candidates of one state share their
parent, and consecutive states of one chain share history, so splitting by
candidate or by state would leak.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .layout import CANDIDATE, H2_HEADS, HISTORY, STATE, FeatureGroup, FeatureLayout

DEFAULT_WEIGHTS = (0.50, 0.25, 0.25)   # D04: target, OOD, retention


class RecordError(ValueError):
    """A meta-record violates the contract."""


# --------------------------------------------------------------------------
# Utility (pre-declared scalar aggregate)
# --------------------------------------------------------------------------
def utility(delta, weights: Sequence[float] = DEFAULT_WEIGHTS) -> np.ndarray:
    """Scalar value of a (target, ood, retention) delta triple: weighted mean.

    ``LHV2`` is ``utility`` applied to the H=2 deltas.  The default is the
    D04 weighting 0.50 target / 0.25 OOD / 0.25 retention.  Freeze the weights
    before looking at any held-out result; do not retune them afterwards.
    """
    d = np.asarray(delta, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if d.shape[-1] != 3 or w.shape != (3,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("need (..., 3) deltas and 3 non-negative weights")
    return d @ (w / w.sum())


def utility_h1(delta_h1, heads: Sequence[str], weights: Sequence[float] = DEFAULT_WEIGHTS):
    """Same aggregate over whichever one-step heads exist (weights renormalised)."""
    d = np.asarray(delta_h1, dtype=np.float64)
    idx = [H2_HEADS.index(h) for h in heads]
    w = np.asarray(weights, dtype=np.float64)[idx]
    return d @ (w / w.sum())


def expected_history_count(state_id: int, history_len: int, rule: str = "applied_h1") -> int:
    """Real history rows expected at decision ``state_id`` (D10).

    applied_h1: the one-step result of update r is known at state r+1, so
                min(t, H) rows (0,1,2,3,3 for t=0..4).
    applied_h2: update r's H=2 result is known at state r+2, so
                min(max(t-1, 0), H) rows (0,0,1,2,3 for t=0..4).
    """
    if rule == "applied_h1":
        return min(state_id, history_len)
    if rule == "applied_h2":
        return min(max(state_id - 1, 0), history_len)
    raise ValueError("unknown history rule %r" % (rule,))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@dataclass
class ValidationReport:
    n_records: int
    n_states: int
    n_trajectories: int
    candidates_per_state: List[int]
    models: List[str]
    tasks: List[str]
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return ("%d records, %d states, %d trajectories, K in %s, models=%s tasks=%s, %d warnings"
                % (self.n_records, self.n_states, self.n_trajectories,
                   sorted(set(self.candidates_per_state)), self.models, self.tasks,
                   len(self.warnings)))


def _finite(x, where):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise RecordError("%s must be a finite number, got %r" % (where, x))
    return float(x)


def label_keys(layout: FeatureLayout) -> Tuple[List[str], List[str]]:
    h2 = ["delta_%s_h2" % h for h in H2_HEADS]
    h1 = ["delta_%s_h1" % h for h in layout.h1_heads]
    return h2, h1


def validate_records(
    records: Sequence[Dict],
    layout: FeatureLayout,
    *,
    strict_history_count: bool = True,
    history_tolerance: float = 1e-3,
    history_rule: str = "applied_h1",
    expected_semantic_hash: Optional[str] = None,
    history_weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> ValidationReport:
    """Hard-fail on contract violations; return soft warnings.

    Checks: exact columns, finiteness, history shape/mask/zero-padding/right
    alignment, ``current_gen_index == state_id``, history length equals
    ``min(state_id, H)`` (all prior rounds were applied), shared state+history
    across the K siblings of a state, one model/task per trajectory, no label-
    named input columns, and (warning) that the newest history entry at state
    t+1 matches an h1 label of some candidate at state t, which catches
    off-by-one errors in history alignment.
    """
    if not records:
        raise RecordError("no records")
    h2_keys, h1_keys = label_keys(layout)
    state_cols, cand_cols = layout.columns(STATE), layout.columns(CANDIDATE)
    for col in state_cols + cand_cols:
        if col.startswith("delta_") or col.startswith("lhv"):
            raise RecordError("input column %r is named like a label" % col)
    H, channels = layout.history_len, list(layout.history_channels)

    states = {}  # (traj, state_id) -> list of records
    traj_meta = {}  # traj -> (model, task)
    traj_group = {}  # traj -> lineage group
    for i, r in enumerate(records):
        where = "record %d" % i
        for k in ("model", "task", "trajectory_id", "state_id", "candidate_id",
                  "state", "candidate", "labels"):
            if k not in r:
                raise RecordError("%s: missing key %r" % (where, k))
        if not isinstance(r["state_id"], int) or r["state_id"] < 0:
            raise RecordError("%s: state_id must be a non-negative int" % where)
        if not isinstance(r["candidate_id"], int) or r["candidate_id"] < 0:
            raise RecordError("%s: candidate_id must be a non-negative int" % where)
        if set(r["state"]) != set(state_cols):
            raise RecordError("%s: state columns differ from layout (diff=%s)"
                              % (where, sorted(set(r["state"]) ^ set(state_cols))))
        if set(r["candidate"]) != set(cand_cols):
            raise RecordError("%s: candidate columns differ from layout (diff=%s)"
                              % (where, sorted(set(r["candidate"]) ^ set(cand_cols))))
        for c in state_cols:
            _finite(r["state"][c], "%s state[%s]" % (where, c))
        for c in cand_cols:
            _finite(r["candidate"][c], "%s candidate[%s]" % (where, c))
        if "current_gen_index" in r["state"] and r["state"]["current_gen_index"] != r["state_id"]:
            raise RecordError("%s: current_gen_index != state_id" % where)
        for k in h2_keys + h1_keys:
            if k not in r["labels"]:
                raise RecordError("%s: missing label %r" % (where, k))
            _finite(r["labels"][k], "%s labels[%s]" % (where, k))

        if layout.has_history:
            h = r.get("history")
            if not h or "values" not in h or "mask" not in h:
                raise RecordError("%s: history needs 'values' and 'mask'" % where)
            if set(h["values"]) != set(channels):
                raise RecordError("%s: history channels differ from layout" % where)
            mask = h["mask"]
            if len(mask) != H or any(m not in (0, 1) for m in mask):
                raise RecordError("%s: history mask must be %d binary entries" % (where, H))
            if list(mask) != sorted(mask):
                raise RecordError("%s: real history entries must be right-aligned "
                                  "(mask = zeros then ones)" % where)
            for ch in channels:
                vals = h["values"][ch]
                if len(vals) != H:
                    raise RecordError("%s: history[%s] must have %d entries" % (where, ch, H))
                for v, m in zip(vals, mask):
                    _finite(v, "%s history[%s]" % (where, ch))
                    if m == 0 and v != 0:
                        raise RecordError("%s: masked history value must be 0" % where)
            if {"prev_accepted_value", "prev_target_change", "prev_ood_change",
                    "prev_retention_change"} <= set(channels):
                vals = h["values"]
                for k in range(H):
                    expect = float(utility([vals["prev_target_change"][k], vals["prev_ood_change"][k],
                                            vals["prev_retention_change"][k]], history_weights))
                    if abs(vals["prev_accepted_value"][k] - expect) > 1e-4:
                        raise RecordError(
                            "%s: prev_accepted_value[%d]=%g but the lhv2 utility of that slot's "
                            "deltas is %g (D10: it must be the utility of the one-step deltas)"
                            % (where, k, vals["prev_accepted_value"][k], expect))
            want = expected_history_count(r["state_id"], H, history_rule)
            if strict_history_count and sum(mask) != want:
                raise RecordError("%s: history has %d real entries but rule %s expects %d "
                                  "(state_id=%d, H=%d)"
                                  % (where, sum(mask), history_rule, want, r["state_id"], H))

        t = r["trajectory_id"]
        if traj_meta.setdefault(t, (r["model"], r["task"])) != (r["model"], r["task"]):
            raise RecordError("%s: trajectory %r mixes models/tasks" % (where, t))
        grp = r.get("group_id", t)
        if traj_group.setdefault(t, grp) != grp:
            raise RecordError("%s: trajectory %r appears in two lineage groups" % (where, t))
        if expected_semantic_hash is not None and r.get("semantic_hash") != expected_semantic_hash:
            raise RecordError("%s: semantic_hash %r != expected %r (definitions changed?)"
                              % (where, r.get("semantic_hash"), expected_semantic_hash))
        states.setdefault((t, r["state_id"]), []).append(r)

    warnings = []  # type: List[str]
    ks = []
    for (t, s), group in states.items():
        ids = [g["candidate_id"] for g in group]
        if len(set(ids)) != len(ids):
            raise RecordError("state (%s, %d): duplicate candidate_id" % (t, s))
        if len(group) < 2:
            raise RecordError("state (%s, %d): needs >= 2 candidates to rank" % (t, s))
        ks.append(len(group))
        for g in group[1:]:
            if g["state"] != group[0]["state"]:
                raise RecordError("state (%s, %d): siblings have different state features" % (t, s))
            if layout.has_history and g["history"] != group[0]["history"]:
                raise RecordError("state (%s, %d): siblings have different history" % (t, s))
    if len(set(ks)) > 1:
        warnings.append("candidates per state vary: %s" % sorted(set(ks)))

    if (history_rule == "applied_h1" and layout.has_history
            and "prev_target_change" in channels and "target" in layout.h1_heads):
        ci = channels.index("prev_target_change")
        bad = 0
        checked = 0
        for (t, s), group in states.items():
            prev = states.get((t, s - 1))
            if not prev or not group[0]["history"]["mask"][-1]:
                continue
            newest = group[0]["history"]["values"]["prev_target_change"][-1]
            options = [p["labels"]["delta_target_h1"] for p in prev]
            checked += 1
            if min(abs(newest - o) for o in options) > history_tolerance:
                bad += 1
        if bad:
            warnings.append(
                "history alignment: at %d/%d states the newest prev_target_change matches no "
                "delta_target_h1 of the previous state's candidates (tol %g). Check the "
                "round indexing, or that history uses realised one-step deltas."
                % (bad, checked, history_tolerance))

    if len(set(traj_group.values())) < 10:
        warnings.append("only %d lineage groups: held-out uncertainty will be unreliable"
                        % len(set(traj_group.values())))
    return ValidationReport(
        n_records=len(records), n_states=len(states), n_trajectories=len(traj_meta),
        candidates_per_state=ks, models=sorted({m for m, _ in traj_meta.values()}),
        tasks=sorted({k for _, k in traj_meta.values()}), warnings=warnings)


def read_jsonl(path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
@dataclass
class MetaDataset:
    layout: FeatureLayout
    state: np.ndarray          # [N, Ds] float32
    cand: np.ndarray           # [N, Dc] float32
    hist: np.ndarray           # [N, H, C] float32 (masked entries are 0)
    hist_mask: np.ndarray      # [N, H] float32
    y_h2: np.ndarray           # [N, 3] float32, order = H2_HEADS
    y_h1: np.ndarray           # [N, n_h1] float32 (NaN = unlabeled)
    model: np.ndarray          # [N] str
    task: np.ndarray           # [N] str
    trajectory: np.ndarray     # [N] str
    state_index: np.ndarray    # [N] int, dense id of (trajectory, state_id)
    state_id: np.ndarray       # [N] int
    candidate_id: np.ndarray   # [N] int
    group: np.ndarray = None   # [N] str lineage group (D02); defaults to trajectory
    semantic_hash: str = ""    # D15

    def __post_init__(self):
        if self.group is None:
            self.group = self.trajectory

    def __len__(self):
        return int(self.state.shape[0])

    @property
    def n_states(self) -> int:
        return int(len(np.unique(self.state_index)))

    def subset(self, idx) -> "MetaDataset":
        idx = np.asarray(idx)
        return MetaDataset(
            layout=self.layout, state=self.state[idx], cand=self.cand[idx],
            hist=self.hist[idx], hist_mask=self.hist_mask[idx], y_h2=self.y_h2[idx],
            y_h1=self.y_h1[idx], model=self.model[idx], task=self.task[idx],
            trajectory=self.trajectory[idx], state_index=self.state_index[idx],
            state_id=self.state_id[idx], candidate_id=self.candidate_id[idx],
            group=self.group[idx], semantic_hash=self.semantic_hash)

    def without_labels(self) -> "MetaDataset":
        """Inputs only (labels set to NaN).  ``predict`` must give identical
        output on this and on the labelled dataset; a test enforces it."""
        out = self.subset(np.arange(len(self)))
        out.y_h2 = np.full_like(self.y_h2, np.nan)
        out.y_h1 = np.full_like(self.y_h1, np.nan)
        return out

    def lhv2(self, weights=DEFAULT_WEIGHTS) -> np.ndarray:
        return utility(self.y_h2, weights)

    def h1_utility(self, weights=DEFAULT_WEIGHTS) -> np.ndarray:
        return utility_h1(self.y_h1, self.layout.h1_heads, weights)

    def drop_groups(self, *names: str) -> "MetaDataset":
        """Remove feature groups (E8).  History is dropped by naming its group."""
        new_layout = self.layout.drop(*names)
        keep = {STATE: [], CANDIDATE: []}  # type: Dict[str, List[int]]
        for kind in (STATE, CANDIDATE):
            off = 0
            for g in self.layout.groups:
                if g.kind != kind:
                    continue
                if g.name not in names:
                    keep[kind].extend(range(off, off + len(g.columns)))
                off += len(g.columns)
        out = self.subset(np.arange(len(self)))
        out.layout = new_layout
        out.state = self.state[:, keep[STATE]]
        out.cand = self.cand[:, keep[CANDIDATE]]
        if not new_layout.has_history:
            out.hist = np.zeros((len(self), 0, 0), dtype=np.float32)
            out.hist_mask = np.zeros((len(self), 0), dtype=np.float32)
        return out

    def with_relative_candidates(self) -> "MetaDataset":
        """Append candidate features centred on the state's sibling mean.

        Legitimate at ranking time: all K candidates of a state are available
        together.  Gives the model direct access to *differences* between
        siblings, which is what ranking needs.
        """
        rel = np.zeros_like(self.cand)
        for s in np.unique(self.state_index):
            rows = np.where(self.state_index == s)[0]
            rel[rows] = self.cand[rows] - self.cand[rows].mean(axis=0, keepdims=True)
        cols = tuple("rel_" + c for c in self.layout.columns(CANDIDATE))
        groups = self.layout.groups + (FeatureGroup("relative_candidate", CANDIDATE, cols),)
        out = self.subset(np.arange(len(self)))
        out.layout = FeatureLayout(self.layout.name + "+rel", groups,
                                   history_len=self.layout.history_len,
                                   h1_heads=self.layout.h1_heads)
        out.cand = np.concatenate([self.cand, rel], axis=1)
        return out


def from_records(records: Sequence[Dict], layout: FeatureLayout, *, validate: bool = True,
                 **validate_kwargs) -> MetaDataset:
    """Build arrays.  ``validate_kwargs`` go to ``validate_records`` (history_rule, ...)."""
    if validate:
        validate_records(records, layout, **validate_kwargs)
    H, channels = layout.history_len, list(layout.history_channels)
    C = len(channels)
    n = len(records)
    state_cols, cand_cols = layout.columns(STATE), layout.columns(CANDIDATE)
    h2_keys, h1_keys = label_keys(layout)
    state = np.zeros((n, len(state_cols)), np.float32)
    cand = np.zeros((n, len(cand_cols)), np.float32)
    hist = np.zeros((n, H if layout.has_history else 0, C if layout.has_history else 0), np.float32)
    hmask = np.zeros((n, H if layout.has_history else 0), np.float32)
    y2 = np.zeros((n, 3), np.float32)
    y1 = np.full((n, len(h1_keys)), np.nan, np.float32)
    sidx = np.zeros(n, np.int64)
    key_to_idx = {}  # type: Dict[Tuple[str, int], int]
    for i, r in enumerate(records):
        state[i] = [r["state"][c] for c in state_cols]
        cand[i] = [r["candidate"][c] for c in cand_cols]
        if layout.has_history:
            hist[i] = np.stack([r["history"]["values"][ch] for ch in channels], axis=1)
            hmask[i] = r["history"]["mask"]
        y2[i] = [r["labels"][k] for k in h2_keys]
        for j, k in enumerate(h1_keys):
            y1[i, j] = r["labels"][k]
        sidx[i] = key_to_idx.setdefault((r["trajectory_id"], r["state_id"]), len(key_to_idx))
    return MetaDataset(
        layout=layout, state=state, cand=cand, hist=hist, hist_mask=hmask, y_h2=y2, y_h1=y1,
        model=np.array([r["model"] for r in records]),
        task=np.array([r["task"] for r in records]),
        trajectory=np.array([r["trajectory_id"] for r in records]),
        state_index=sidx,
        state_id=np.array([r["state_id"] for r in records], dtype=np.int64),
        candidate_id=np.array([r["candidate_id"] for r in records], dtype=np.int64),
        group=np.array([r.get("group_id", r["trajectory_id"]) for r in records]),
        semantic_hash=next(iter({r.get("semantic_hash", "") for r in records}), ""))


# --------------------------------------------------------------------------
# Grouped splits
# --------------------------------------------------------------------------
def _group_values(ds: MetaDataset, by: str) -> np.ndarray:
    if by not in ("trajectory", "group", "model", "task"):
        raise ValueError("group by 'group', 'trajectory', 'model' or 'task'")
    return getattr(ds, by)


def grouped_kfold(ds: MetaDataset, n_splits: int, seed: int = 0,
                  by: str = "trajectory") -> List[Tuple[np.ndarray, np.ndarray]]:
    """K folds with disjoint groups; all candidates of a state stay together."""
    values = _group_values(ds, by)
    groups = np.unique(values)
    if len(groups) < n_splits:
        raise ValueError("need >= %d %s groups for %d folds, have %d"
                         % (n_splits, by, n_splits, len(groups)))
    rng = np.random.RandomState(seed)
    order = rng.permutation(groups)
    fold_of = {g: i % n_splits for i, g in enumerate(order)}
    folds = np.array([fold_of[v] for v in values])
    out = []
    for k in range(n_splits):
        out.append((np.where(folds != k)[0], np.where(folds == k)[0]))
    return out


def split_by_group(ds: MetaDataset, frac: float, seed: int = 0,
                   by: str = "trajectory") -> Tuple[np.ndarray, np.ndarray]:
    """Hold out ``frac`` of the groups (at least one, leaving at least one)."""
    values = _group_values(ds, by)
    groups = np.unique(values)
    if len(groups) < 2:
        raise ValueError("need >= 2 groups to split")
    rng = np.random.RandomState(seed)
    order = rng.permutation(groups)
    n_val = int(min(max(1, round(frac * len(groups))), len(groups) - 1))
    val_groups = set(order[:n_val])
    is_val = np.array([v in val_groups for v in values])
    return np.where(~is_val)[0], np.where(is_val)[0]


def fixed_split(ds: MetaDataset, assignment: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Row indices per split from a fixed lineage-group -> split-name manifest
    (D02, e.g. 8 train / 2 validation / 2 test groups).  Every group in the data
    must be assigned and no assigned group may be absent."""
    groups = set(np.unique(ds.group).tolist())
    if groups != set(assignment):
        raise ValueError("assignment must cover exactly the data's groups; diff=%s"
                         % sorted(groups ^ set(assignment)))
    out = {}
    for name in sorted(set(assignment.values())):
        members = {g for g, v in assignment.items() if v == name}
        out[name] = np.where(np.isin(ds.group, list(members)))[0]
    return out


def leave_one_group_out(ds: MetaDataset, by: str = "model"):
    """Yield (held-out value, train idx, test idx).  For E6/E7 transfer."""
    values = _group_values(ds, by)
    for g in np.unique(values):
        yield g, np.where(values != g)[0], np.where(values == g)[0]


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def _safe_std(x, floor=1e-6):
    s = x.std(axis=0)
    return np.where(s < floor, 1.0, s)


@dataclass
class Standardizer:
    """Column-wise z-scoring fitted on training rows only.

    ``per_model=True`` fits feature statistics separately for each backbone
    (using unlabeled features only), which removes scale differences between
    model families.  For a held-out backbone call ``add_model_stats`` with its
    *unlabeled* features before ``transform``.  Labels are always scaled with
    global training statistics.
    """

    per_model: bool = False
    stats: Dict = field(default_factory=dict)

    def _feature_stats(self, ds: MetaDataset) -> Dict:
        st = {"state": (ds.state.mean(0), _safe_std(ds.state)),
              "cand": (ds.cand.mean(0), _safe_std(ds.cand))}
        if ds.layout.has_history and ds.hist.shape[2] > 0:
            m = ds.hist_mask.astype(bool)
            real = ds.hist[m]  # [n_real, C]
            if real.shape[0] > 0:
                st["hist"] = (real.mean(0), _safe_std(real))
            else:
                C = ds.hist.shape[2]
                st["hist"] = (np.zeros(C, np.float32), np.ones(C, np.float32))
        return st

    def fit(self, ds: MetaDataset) -> "Standardizer":
        self.stats = {"global": self._feature_stats(ds)}
        if self.per_model:
            for m in np.unique(ds.model):
                self.stats[str(m)] = self._feature_stats(ds.subset(np.where(ds.model == m)[0]))
        self.stats["y2"] = (ds.y_h2.mean(0), _safe_std(ds.y_h2))
        y1 = ds.y_h1
        if y1.shape[1] > 0 and not np.all(np.isnan(y1)):
            self.stats["y1"] = (np.nanmean(y1, 0), np.where(np.nanstd(y1, 0) < 1e-6, 1.0, np.nanstd(y1, 0)))
        else:
            self.stats["y1"] = (np.zeros(y1.shape[1], np.float32), np.ones(y1.shape[1], np.float32))
        return self

    def add_model_stats(self, ds: MetaDataset) -> "Standardizer":
        for m in np.unique(ds.model):
            self.stats[str(m)] = self._feature_stats(ds.subset(np.where(ds.model == m)[0]))
        return self

    def transform(self, ds: MetaDataset) -> Dict[str, np.ndarray]:
        n = len(ds)
        state = np.zeros_like(ds.state)
        cand = np.zeros_like(ds.cand)
        hist = np.zeros_like(ds.hist)
        if self.per_model:
            groups = [(str(m), np.where(ds.model == m)[0]) for m in np.unique(ds.model)]
        else:
            groups = [("global", np.arange(n))]
        for key, rows in groups:
            if key not in self.stats:
                raise KeyError("no normalisation stats for model %r; call add_model_stats" % key)
            st = self.stats[key]
            state[rows] = (ds.state[rows] - st["state"][0]) / st["state"][1]
            cand[rows] = (ds.cand[rows] - st["cand"][0]) / st["cand"][1]
            if "hist" in st:
                hist[rows] = ((ds.hist[rows] - st["hist"][0]) / st["hist"][1]) \
                    * ds.hist_mask[rows][..., None]
        out = {"state": state, "cand": cand, "hist": hist, "hist_mask": ds.hist_mask.copy()}
        out["y_h2"] = (ds.y_h2 - self.stats["y2"][0]) / self.stats["y2"][1]
        out["y_h1"] = (ds.y_h1 - self.stats["y1"][0]) / self.stats["y1"][1]
        return out

    def inverse_h2(self, z: np.ndarray) -> np.ndarray:
        return z * self.stats["y2"][1] + self.stats["y2"][0]

    def inverse_h1(self, z: np.ndarray) -> np.ndarray:
        return z * self.stats["y1"][1] + self.stats["y1"][0]

    # JSON-/torch.save-friendly
    def to_dict(self) -> Dict:
        def conv(v):
            if isinstance(v, tuple):
                return [np.asarray(a, dtype=np.float64).tolist() for a in v]
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v
        return {"per_model": self.per_model, "stats": conv(self.stats)}

    @staticmethod
    def from_dict(d: Dict) -> "Standardizer":
        def conv(v):
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], list):
                return (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32))
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v
        return Standardizer(per_model=bool(d["per_model"]), stats=conv(d["stats"]))
