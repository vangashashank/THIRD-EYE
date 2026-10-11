"""Feature layouts for the Third Eye forecasters.

A layout says which columns each input group has.  Everything downstream
(arrays, models, ablations, checkpoints) is derived from a layout, so a
feature-group ablation (E8) is just ``layout.drop("short_probe_dynamics")``
followed by a re-fit with the narrower dimensions.

Two layouts are provided:

* ``contract_layout``: the 33-field ``configs/feature_schema.yaml`` contract,
  with ``layerwise_grad_norms`` replaced by a fixed number of relative-depth
  bins so that models with different depth (Qwen3-4B vs Llama-3.2-3B vs
  Gemma-3-4B) produce vectors of the same width (needed for E6/E7 transfer).
* ``legacy_layout``: Sprint 1's smoke features (3 state / 4 candidate / 3
  history), kept so the old pilot can still be expressed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

STATE = "state"
CANDIDATE = "candidate"
HISTORY = "history"
KINDS = (STATE, CANDIDATE, HISTORY)

H2_HEADS = ("target", "ood", "retention")
HISTORY_CHANNELS = (
    "prev_accepted_value",
    "prev_target_change",
    "prev_ood_change",
    "prev_retention_change",
)


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    kind: str
    columns: Tuple[str, ...]

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("Unknown group kind: %r" % (self.kind,))
        if not self.columns:
            raise ValueError("Group %r has no columns" % (self.name,))
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Duplicate column in group %r" % (self.name,))


@dataclass(frozen=True)
class FeatureLayout:
    """Ordered feature groups plus label-head names.

    ``h1_heads`` are the heads available for the one-step (t+1) labels.  The
    YAML contract only has ``delta_target_h1``; the pilot already records t+1
    OOD/retention too, so ``("target", "ood", "retention")`` is allowed and
    gives a fairer one-step baseline (same utility, shorter horizon).
    """

    name: str
    groups: Tuple[FeatureGroup, ...]
    history_len: int = 3
    h1_heads: Tuple[str, ...] = ("target",)

    def __post_init__(self):
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate group names")
        if len([g for g in self.groups if g.kind == HISTORY]) > 1:
            raise ValueError("At most one history group is supported")
        for head in self.h1_heads:
            if head not in H2_HEADS:
                raise ValueError("Unknown h1 head %r" % (head,))
        if self.history_len < 0:
            raise ValueError("history_len must be >= 0")
        flat = self.columns(STATE) + self.columns(CANDIDATE)
        if len(set(flat)) != len(flat):
            raise ValueError("Column names must be unique across state/candidate groups")

    # ---- structure -------------------------------------------------------
    @property
    def group_names(self) -> List[str]:
        return [g.name for g in self.groups]

    def columns(self, kind: str) -> List[str]:
        out = []  # type: List[str]
        for g in self.groups:
            if g.kind == kind:
                out.extend(g.columns)
        return out

    def dim(self, kind: str) -> int:
        return len(self.columns(kind))

    @property
    def has_history(self) -> bool:
        return self.history_len > 0 and any(g.kind == HISTORY for g in self.groups)

    @property
    def history_channels(self) -> Tuple[str, ...]:
        for g in self.groups:
            if g.kind == HISTORY:
                return g.columns
        return ()

    def group_slice(self, group_name: str) -> slice:
        """Column slice of one state/candidate group inside its kind's matrix."""
        group = self._group(group_name)
        if group.kind == HISTORY:
            raise ValueError("History groups have channels, not a column slice")
        start = 0
        for g in self.groups:
            if g.kind != group.kind:
                continue
            if g.name == group_name:
                return slice(start, start + len(g.columns))
            start += len(g.columns)
        raise AssertionError("unreachable")

    def _group(self, name: str) -> FeatureGroup:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError("Unknown group %r; have %s" % (name, self.group_names))

    def drop(self, *group_names: str) -> "FeatureLayout":
        for n in group_names:
            self._group(n)  # raises if unknown
        kept = tuple(g for g in self.groups if g.name not in group_names)
        return FeatureLayout(
            name="%s-minus-%s" % (self.name, "+".join(group_names)),
            groups=kept,
            history_len=self.history_len,
            h1_heads=self.h1_heads,
        )

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "history_len": self.history_len,
            "h1_heads": list(self.h1_heads),
            "groups": [
                {"name": g.name, "kind": g.kind, "columns": list(g.columns)}
                for g in self.groups
            ],
        }

    @staticmethod
    def from_dict(d: Dict) -> "FeatureLayout":
        return FeatureLayout(
            name=d["name"],
            history_len=int(d["history_len"]),
            h1_heads=tuple(d["h1_heads"]),
            groups=tuple(
                FeatureGroup(g["name"], g["kind"], tuple(g["columns"]))
                for g in d["groups"]
            ),
        )

    def fingerprint(self) -> str:
        """Order-sensitive hash.  Checkpoints refuse to load on mismatch."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def depth_bin_names(num_bins: int) -> Tuple[str, ...]:
    return tuple("layer_depth_bin_%d" % i for i in range(num_bins))


def contract_layout(
    num_depth_bins: int = 6,
    history_len: int = 3,
    h1_heads: Sequence[str] = ("target",),
) -> FeatureLayout:
    """Layout implementing ``configs/feature_schema.yaml`` (version 0.1-pilot).

    Widths: state = 5 (+1 generation index), candidate = 16 + num_depth_bins,
    history = 4 channels x ``history_len`` (+ mask handled separately).
    """
    if num_depth_bins < 1:
        raise ValueError("num_depth_bins must be >= 1")
    groups = (
        FeatureGroup(
            "current_state_performance", STATE,
            ("target_val_accuracy", "ood_proxy_accuracy", "retention_anchor_accuracy",
             "gen_confidence_mean", "gen_confidence_std"),
        ),
        FeatureGroup("generation_index", STATE, ("current_gen_index",)),
        FeatureGroup(
            "candidate_data_stats", CANDIDATE,
            ("token_length_mean", "token_length_std", "pre_update_nll", "mean_confidence",
             "verifier_pass_rate", "difficulty_proxy", "embedding_diversity", "duplicate_rate"),
        ),
        FeatureGroup(
            "gradient_interference_stats", CANDIDATE,
            ("grad_norm", "grad_cosine_retention") + depth_bin_names(num_depth_bins)
            + ("sign_agreement",),
        ),
        FeatureGroup(
            "short_probe_dynamics", CANDIDATE,
            ("train_loss_slope", "anchor_loss_slope", "kl_drift_anchor",
             "lora_delta_norm", "val_loss_slope"),
        ),
        FeatureGroup("recursive_history", HISTORY, HISTORY_CHANNELS),
    )
    return FeatureLayout("contract-0.1-pilot-bins%d" % num_depth_bins, groups,
                         history_len=history_len, h1_heads=tuple(h1_heads))


def legacy_layout(history_len: int = 3) -> FeatureLayout:
    """Sprint 1 smoke features (loss-based)."""
    groups = (
        FeatureGroup("parent_losses", STATE,
                     ("parent_target_loss", "parent_ood_loss", "parent_retention_loss")),
        FeatureGroup("proposal", CANDIDATE,
                     ("learning_rate_div_1e-4", "epochs", "batch_size", "max_length_div_128")),
        FeatureGroup("applied_history", HISTORY,
                     ("applied_target_loss_reduction", "applied_ood_loss_reduction",
                      "applied_retention_loss_reduction")),
    )
    return FeatureLayout("legacy-sprint1", groups, history_len=history_len)
