"""Planted-signal meta-datasets for testing the pipeline (NOT research data).

Purpose: controls.  A pipeline that cannot recover a planted signal is broken;
a pipeline that "finds" signal in pure noise is leaking.  Both are tested.

Generative story (per state, K sibling candidates)
--------------------------------------------------
* ``imm`` : one-step consequence of each candidate (3 heads), a sparse linear
  function of candidate features.
* ``fut`` : an independent sparse function of other candidate features.
* H=1 label = imm + noise.
* H=2 label = w * imm + (1 - w) * fut + state offset + history effect + noise,
  with ``w = immediate_weight``.  ``w = 1``: the immediate winner is the future
  winner.  ``w = 0``: immediate value says nothing about future value (the
  scenario Third Eye exists for).  Future retention is anti-correlated with
  future target gain so gain-vs-retention trade-offs exist.
* ``null=True`` replaces every label by noise (negative control).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .data import MetaDataset, from_records
from .layout import CANDIDATE, STATE, FeatureLayout, H2_HEADS

LABEL_SCALE = 0.03  # roughly "accuracy points"


def _sparse(rng, dim, n_active):
    w = np.zeros(dim)
    idx = rng.choice(dim, size=min(n_active, dim), replace=False)
    w[idx] = rng.choice([-1.0, 1.0], size=len(idx)) * rng.uniform(0.6, 1.0, size=len(idx))
    return w


def make_synthetic_records(
    layout: FeatureLayout, *, n_trajectories: int = 40, states_per_trajectory: int = 5,
    k: int = 3, immediate_weight: float = 0.3, signal: float = 1.0, noise: float = 0.4,
    state_offset_sd: float = 0.5, history_effect: float = 0.4, null: bool = False,
    model: str = "synthetic-a", task: str = "math", seed: int = 0,
    structure_seed: int = 1234, n_groups: Optional[int] = None,
    history_rule: str = "applied_h1", semantic_hash: str = "",
) -> List[Dict]:
    """Records in the format of ``data.py``.  ``structure_seed`` fixes the planted
    functions so several datasets (e.g. "models") can share one mechanism while
    differing in ``seed``.  ``n_groups`` assigns chains to lineage groups
    round-robin.  ``history_rule="applied_h2"`` delays history by one extra round
    (an approximation of the D10 variant: same values, one round later)."""
    srng = np.random.RandomState(structure_seed)
    rng = np.random.RandomState(seed)
    state_cols, cand_cols = layout.columns(STATE), layout.columns(CANDIDATE)
    Dc, Ds = len(cand_cols), len(state_cols)
    W_imm = np.stack([_sparse(srng, Dc, 4) for _ in H2_HEADS], axis=1)      # [Dc,3]
    W_fut = np.stack([_sparse(srng, Dc, 4) for _ in H2_HEADS], axis=1)
    W_fut[:, 2] = -0.6 * W_fut[:, 0] + 0.4 * W_fut[:, 2]                    # trade-off
    V_state = srng.normal(0, 0.3, size=(Ds, 3))
    H, channels = layout.history_len, list(layout.history_channels)
    ret_hist = channels.index("prev_retention_change") if "prev_retention_change" in channels else None
    gen_col = state_cols.index("current_gen_index") if "current_gen_index" in state_cols else None

    records = []  # type: List[Dict]
    for ti in range(n_trajectories):
        traj = "%s/%s/chain-%03d" % (model, task, ti)
        s_prev = rng.normal(size=Ds)
        applied = []  # list of dicts of realised one-step deltas, oldest first
        for t in range(states_per_trajectory):
            s = 0.6 * s_prev + 0.8 * rng.normal(size=Ds)
            s_prev = s
            if gen_col is not None:
                s = s.copy()
                s[gen_col] = float(t)
            shared = rng.normal(size=Dc)
            X = 0.5 * shared[None, :] + rng.normal(size=(k, Dc))
            offset = rng.normal(0, state_offset_sd, size=3)
            imm = LABEL_SCALE * signal * (X @ W_imm + (s @ V_state)[None, :] * 0.3)
            fut = LABEL_SCALE * signal * (X @ W_fut)
            last = applied[-1] if applied else None
            hist_term = np.zeros(3)
            if last is not None and ret_hist is not None:
                hist_term[2] = history_effect * LABEL_SCALE * last[2]
            y1 = imm + LABEL_SCALE * noise * rng.normal(size=imm.shape)
            y2 = (immediate_weight * imm + (1 - immediate_weight) * fut
                  + LABEL_SCALE * offset[None, :] + hist_term[None, :]
                  + LABEL_SCALE * noise * rng.normal(size=imm.shape))
            if null:
                y1 = LABEL_SCALE * rng.normal(size=imm.shape)
                y2 = LABEL_SCALE * (offset[None, :] + rng.normal(size=imm.shape))

            visible = applied if history_rule == "applied_h1" else applied[:-1]
            window = visible[-H:] if H > 0 else []
            pad = H - len(window)
            hist_vals = {}
            if layout.has_history:
                for ci, ch in enumerate(channels):
                    col = [0.0] * pad
                    for a in window:
                        if ch == "prev_accepted_value":
                            col.append(float(np.mean(a)))
                        elif ch == "prev_target_change":
                            col.append(float(a[0]))
                        elif ch == "prev_ood_change":
                            col.append(float(a[1]))
                        elif ch == "prev_retention_change":
                            col.append(float(a[2]))
                        else:
                            col.append(0.0)
                    hist_vals[ch] = col
                mask = [0.0] * pad + [1.0] * len(window)

            for j in range(k):
                labels = {"delta_%s_h2" % h: float(y2[j, i]) for i, h in enumerate(H2_HEADS)}
                for h in layout.h1_heads:
                    labels["delta_%s_h1" % h] = float(y1[j, H2_HEADS.index(h)])
                rec = {"model": model, "task": task, "trajectory_id": traj,
                       "state_id": t, "candidate_id": j,
                       "state": {c: float(s[i]) for i, c in enumerate(state_cols)},
                       "candidate": {c: float(X[j, i]) for i, c in enumerate(cand_cols)},
                       "labels": labels}
                if layout.has_history:
                    rec["history"] = {"values": hist_vals, "mask": mask}
                if n_groups:
                    rec["group_id"] = "%s/%s/group-%02d" % (model, task, ti % n_groups)
                if semantic_hash:
                    rec["semantic_hash"] = semantic_hash
                records.append(rec)

            # accept a random candidate; its realised one-step deltas enter history
            pick = rng.randint(k)
            applied.append(y1[pick].copy())
    return records


def make_synthetic_dataset(layout: FeatureLayout, **kwargs) -> MetaDataset:
    return from_records(make_synthetic_records(layout, **kwargs), layout)
