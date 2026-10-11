"""Isolated validation of the partially agreed Sprint 2 contract.

No training modules are imported. Structural validation never certifies research
measurements or agreement on the contract's unresolved semantic definitions.
"""
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from math import prod
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Tuple, Union

import numpy as np
import yaml

SEMANTIC_VERSION = "0.2-sprint2"
CORE_DECISIONS = 5
METRICS = ("target", "ood", "retention")
UTILITY_WEIGHTS = np.array([0.50, 0.25, 0.25], dtype=np.float64)
UTILITY_WEIGHTS.setflags(write=False)


_INPUT_FIELDS = {
    "current_state_performance": (
        "target_val_accuracy", "ood_proxy_accuracy", "retention_anchor_accuracy",
        "gen_confidence_mean", "gen_confidence_std",
    ),
    "candidate_data_stats": (
        "token_length_mean", "token_length_std", "pre_update_nll", "mean_confidence",
        "verifier_pass_rate", "difficulty_proxy", "embedding_diversity", "duplicate_rate",
    ),
    "gradient_interference_stats": (
        "grad_norm", "grad_cosine_retention", "layerwise_grad_norms", "sign_agreement",
    ),
    "short_probe_dynamics": (
        "train_loss_slope", "anchor_loss_slope", "kl_drift_anchor",
        "lora_delta_norm", "val_loss_slope",
    ),
    "recursive_history": (
        "prev_accepted_value", "current_gen_index", "prev_target_change",
        "prev_ood_change", "prev_retention_change", "history_mask",
    ),
}
_LABEL_FIELDS = (
    "delta_target_h1", "delta_target_h2", "delta_ood_h2", "delta_retention_h2", "lhv2",
)
_HISTORY_VALUES = (
    "prev_accepted_value", "prev_target_change", "prev_ood_change", "prev_retention_change",
)
_DERIVED_KEYS = (
    "static_candidate_feature_dim", "static_state_feature_dim", "history_feature_dim",
)
_SEMANTIC_DECISIONS = (
    "Only D01/D02/D04/D05/D13/D15 and explicit D03/D10/D11 revisions are accepted; schema is not frozen.",
    "Task scorers, dataset identities and the 256-example anchor remain unresolved (D03).",
    "Generation caps need training-only measured calibration and review before collection (D03).",
    "Actual model placement/count/ordered manifest must still be supplied explicitly (D05).",
    "Generation/corrected-answer confidence formulas and std convention are unresolved.",
    "Near-duplicate detection, threshold and duplicate-rate denominator are unresolved.",
    "Token lengths, NLL reduction, verifier attempts, difficulty and embeddings need definitions.",
    "Probe budget/optimizer policy, gradients, slopes, KL and adapter delta need definitions.",
    "Compute/storage and branch-retention approval remain unresolved (D14).",
)


class ContractError(ValueError):
    """Invalid schema, configuration, payload or availability metadata."""


class UnresolvedContractError(ContractError):
    """Structural validation needs an explicit dimension or layer manifest."""


def accuracy_deltas(parent, endpoint):
    for values in (parent, endpoint):
        _mapping_keys(values, METRICS, where="accuracy metrics")
        for value in values.values():
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise ContractError("Accuracies must be finite fractions in [0, 1]")
    return np.array([endpoint[key] - parent[key] for key in METRICS], dtype=np.float32)


def weighted_utility(deltas):
    values = np.asarray(deltas)
    if values.shape != (3,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ContractError("Utility requires three finite target/OOD/retention deltas")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.float32(values.astype(np.float64) @ UTILITY_WEIGHTS)
    if not np.isfinite(result):
        raise ContractError("Nonfinite utility after float32 conversion")
    return result


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ContractError("YAML merge keys are unsupported; declare fields explicitly")
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ContractError("Schema mapping keys must be strings")
        if key in result:
            raise ContractError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping,
)


def _mapping_keys(value, required, optional=(), *, where):
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{where}: expected a mapping with string keys")
    required = set(required)
    missing = required - set(value)
    extra = set(value) - required - set(optional)
    if missing or extra:
        raise ContractError(f"{where}: missing fields {sorted(missing)}; unknown fields {sorted(extra)}")


def _integer(value, *, where, minimum=0):
    if type(value) is not int or value < minimum:
        raise ContractError(f"{where}: expected an integer >= {minimum}")


def _layer_order(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ContractError("layer_order: expected a nonempty explicit list/tuple")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ContractError("layer_order: identities must be nonempty strings")
    if len(set(value)) != len(value):
        raise ContractError("layer_order: duplicate layer identities")
    return tuple(value)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    shape: Tuple[Union[int, str], ...]
    description: Optional[str] = None


@dataclass(frozen=True)
class GroupSpec:
    name: str
    scope: str
    fields: Tuple[FieldSpec, ...]


@dataclass(frozen=True)
class ObservationAvailability:
    """Caller-asserted logical availability, not evidence verified by this module.

    Source must be applied_h1 for every history value, including accepted utility.
    available_at_state is the first recursive state where the value was observed.
    Offline simulated_h2 labels are intentionally not an accepted source.
    """
    source: str
    available_at_state: int


@dataclass(frozen=True)
class HistoryEntry:
    update_round: int
    observations: Mapping
    transition_id: Optional[str] = None


@dataclass(frozen=True)
class TrajectoryIdentity:
    corpus_version: str
    group_id: str
    trajectory_id: str
    root_sha256: str
    evaluation_sha256: str


@dataclass(frozen=True)
class AppliedTransition:
    identity: TrajectoryIdentity
    transition_id: str
    from_state: int
    to_state: int
    candidate_id: int
    parent_sha256: str
    endpoint_sha256: str
    parent_accuracy: Mapping
    endpoint_accuracy: Mapping
    available_at_state: int
    observed_at: int
    record_type: str = "applied_h1"
    successful: bool = True


@dataclass(frozen=True)
class HistoryContext:
    """Explicit caller policy/provenance for one unbatched history window.

    entries are aligned with mask slots; padding uses None. update_round r is an
    applied update from state r to r+1 under this supplied indexing convention.
    ledger is a complete as-of applied lineage, not a simulated branch label file.
    observed_at/cutoff are integer logical observation ticks, separate from state.
    Checks verify consistency of supplied evidence, not independently its truth.
    """
    ranking_state: int
    padding_side: str
    order: str
    entries: Tuple[Optional[HistoryEntry], ...]
    identity: Optional[TrajectoryIdentity] = None
    ranking_parent_sha256: Optional[str] = None
    cutoff: Optional[int] = None
    ledger: Tuple[AppliedTransition, ...] = ()


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ContractError("Expected a lowercase SHA-256 identity")


def _applied_window(context, history_len):
    if not isinstance(context.identity, TrajectoryIdentity):
        raise ContractError("Explicit trajectory identity is required")
    identity = context.identity
    for name in ("corpus_version", "group_id", "trajectory_id"):
        if not isinstance(getattr(identity, name), str) or not getattr(identity, name).strip():
            raise ContractError("Trajectory identity strings must be nonempty")
    _sha(identity.root_sha256)
    _sha(identity.evaluation_sha256)
    _sha(context.ranking_parent_sha256)
    _integer(context.cutoff, where="observation cutoff")
    _integer(context.ranking_state, where="ranking_state")
    if context.ranking_state > CORE_DECISIONS:
        raise ContractError("ranking_state exceeds T=5")
    if not isinstance(context.ledger, (tuple, list)) or len(context.ledger) != context.ranking_state:
        raise ContractError("Missing expected applied observations; ledger must cover states 0..t-1")
    parent = identity.root_sha256
    previous = None
    seen = set()
    for state, transition in enumerate(context.ledger):
        if not isinstance(transition, AppliedTransition) or transition.identity != identity:
            raise ContractError("Applied-transition provenance/identity mismatch")
        if (transition.record_type != "applied_h1" or transition.successful is not True
                or type(transition.from_state) is not int or type(transition.to_state) is not int
                or transition.from_state != state or transition.to_state != state + 1):
            raise ContractError("Expected a successful applied_h1 transition, not a simulated branch")
        if (not isinstance(transition.transition_id, str) or not transition.transition_id.strip()
                or transition.transition_id in seen):
            raise ContractError("Missing or duplicate applied transition ID")
        seen.add(transition.transition_id)
        _integer(transition.candidate_id, where="selected candidate")
        if transition.candidate_id >= 3:
            raise ContractError("Expected one of K=3 applied candidates")
        _sha(transition.parent_sha256)
        _sha(transition.endpoint_sha256)
        if transition.parent_sha256 != parent:
            raise ContractError("Applied parent/endpoint chain mismatch")
        accuracy_deltas(transition.parent_accuracy, transition.endpoint_accuracy)
        if previous is not None and transition.parent_accuracy != previous:
            raise ContractError("Applied accuracy endpoints do not form one lineage")
        _integer(transition.available_at_state, where="available_at_state")
        _integer(transition.observed_at, where="observed_at")
        if (not state + 1 <= transition.available_at_state <= context.ranking_state
                or transition.observed_at > context.cutoff):
            raise ContractError("Applied observation unavailable at ranking cutoff")
        parent, previous = transition.endpoint_sha256, transition.endpoint_accuracy
    if parent != context.ranking_parent_sha256:
        raise ContractError("Ranking parent does not match the applied lineage")
    return context.ledger[-history_len:]


def build_applied_history(ledger, *, identity, ranking_state, ranking_parent_sha256, cutoff):
    """Pure D10 collator. Never invent/pad missing successfully applied results."""
    context = HistoryContext(ranking_state, "left", "oldest_first", (), identity,
                             ranking_parent_sha256, cutoff, tuple(ledger))
    window = _applied_window(context, 3)
    values = {name: np.zeros(3, np.float32) for name in (*_HISTORY_VALUES, "history_mask")}
    values["current_gen_index"] = np.array([ranking_state], np.int32)
    entries = [None] * 3
    for slot, transition in enumerate(window, start=3-len(window)):
        deltas = accuracy_deltas(transition.parent_accuracy, transition.endpoint_accuracy)
        for name, value in zip(_HISTORY_VALUES, (weighted_utility(deltas), *deltas)):
            values[name][slot] = value
        values["history_mask"][slot] = 1
        entries[slot] = HistoryEntry(transition.from_state, {
            name: ObservationAvailability("applied_h1", transition.available_at_state)
            for name in _HISTORY_VALUES
        }, transition.transition_id)
    return values, HistoryContext(ranking_state, "left", "oldest_first", tuple(entries),
                                 identity, ranking_parent_sha256, cutoff, tuple(ledger))


@dataclass(frozen=True)
class ValidationResult:
    unresolved_definitions: Tuple[str, ...]
    structural_only: bool = True


@dataclass(frozen=True)
class FeatureContract:
    schema_version: str
    frozen_on: Optional[str]
    groups: Tuple[GroupSpec, ...]
    dimensions: Mapping
    layer_order: Optional[Tuple[str, ...]]
    history_len: int
    probe_steps: int
    source_sha256: str
    schema_fingerprint: str
    fingerprint: str

    @property
    def input_groups(self):
        return tuple(group for group in self.groups if group.name != "labels")

    @property
    def label_group(self):
        return next(group for group in self.groups if group.name == "labels")

    def resolved_shape(self, field):
        shape = []
        for size in field.shape:
            if isinstance(size, str):
                if size not in self.dimensions:
                    raise UnresolvedContractError(f"Unresolved dimension: {size}; supply explicit configuration")
                size = self.dimensions[size]
            shape.append(size)
        return tuple(shape)

    def dimension_summary(self):
        widths = {}
        for group in self.groups:
            try:
                widths[group.name] = sum(
                    prod(self.resolved_shape(field)) for field in group.fields
                    if not (group.name == "recursive_history" and field.name == "current_gen_index")
                )
            except UnresolvedContractError:
                widths[group.name] = None
        candidates = [widths[name] for name in (
            "candidate_data_stats", "gradient_interference_stats", "short_probe_dynamics",
        )]
        return {
            "static_state_feature_dim": widths["current_state_performance"],
            "static_candidate_feature_dim": None if None in candidates else sum(candidates),
            "history_feature_dim": widths["recursive_history"],
        }

    def unresolved_definitions(self):
        issues = []
        for name in ("num_lora_layers", "history_len"):
            if name not in self.dimensions:
                issues.append(f"Unresolved dimension {name}: supply explicit configuration.")
        if self.layer_order is None:
            issues.append("Unresolved layer_order: supply an explicit ordered layer manifest.")
        return tuple(issues) + _SEMANTIC_DECISIONS

    def _validate_groups(self, values, groups, *, where):
        _mapping_keys(values, (group.name for group in groups), where=where)
        for group in groups:
            fields = values[group.name]
            _mapping_keys(fields, (field.name for field in group.fields), where=group.name)
            for field in group.fields:
                value = fields[field.name]
                path = f"{group.name}.{field.name}"
                if type(value) is not np.ndarray:
                    raise ContractError(f"{path}: expected a NumPy ndarray; no implicit conversion")
                if value.dtype != np.dtype(field.dtype):
                    raise ContractError(f"{path}: expected dtype {field.dtype}, got {value.dtype}")
                shape = self.resolved_shape(field)
                if value.shape != shape:
                    raise ContractError(f"{path}: expected shape {shape}, got {value.shape}")
                if not np.isfinite(value).all():
                    raise ContractError(f"{path}: values must be finite, including padding")

    def validate_inputs(self, values, *, layer_order, history_context):
        """Check strict unbatched arrays and as-of history; do not compute features.

        Labels belong to validate_labels(), never this payload. Caller-supplied
        provenance is required even for empty history. Success is structural only.
        """
        if isinstance(values, Mapping):
            forbidden = {"labels", *_LABEL_FIELDS}
            if forbidden.intersection(values):
                raise ContractError("Label fields are forbidden in forecasting inputs")
            for group in values.values():
                if isinstance(group, Mapping) and forbidden.intersection(group):
                    raise ContractError("Label fields are forbidden in forecasting inputs")
        if self.layer_order is None:
            raise UnresolvedContractError("Unresolved layer_order: supply an explicit ordered layer manifest")
        if _layer_order(layer_order) != self.layer_order:
            raise ContractError("Payload layer_order does not match the configured layer manifest")
        self._validate_groups(values, self.input_groups, where="inputs")
        for name in ("target_val_accuracy", "ood_proxy_accuracy", "retention_anchor_accuracy"):
            if np.any((values["current_state_performance"][name] < 0)
                      | (values["current_state_performance"][name] > 1)):
                raise ContractError(f"{name}: accuracy must be a fraction in [0, 1]")
        self._validate_history(values["recursive_history"], history_context)
        return ValidationResult(self.unresolved_definitions())

    def validate_labels(self, values):
        """Validate the separate offline label payload; never expose it as inputs."""
        self._validate_groups(values, (self.label_group,), where="labels payload")
        labels = values["labels"]
        if any(np.any(np.abs(value) > 1) for value in labels.values()):
            raise ContractError("Accuracy delta labels must be in [-1, 1]")
        expected = weighted_utility([labels[f"delta_{name}_h2"][0] for name in METRICS])
        if labels["lhv2"][0] != expected:
            raise ContractError("lhv2 must equal the agreed weighted H2 deltas, never H1")
        return ValidationResult(self.unresolved_definitions())

    def _validate_history(self, history, context):
        if not isinstance(context, HistoryContext):
            raise ContractError("history_context: explicit policy and availability metadata are required")
        _integer(context.ranking_state, where="ranking_state")
        window = _applied_window(context, self.history_len)
        if int(history["current_gen_index"][0]) != context.ranking_state:
            raise ContractError("current_gen_index must equal the as-of ranking_state")
        if (not isinstance(context.padding_side, str) or not isinstance(context.order, str)
                or context.padding_side != "left" or context.order != "oldest_first"):
            raise ContractError("history_context: supply padding_side and history order explicitly")
        if not isinstance(context.entries, (tuple, list)) or len(context.entries) != self.history_len:
            raise ContractError("history_context: entries must align with all history slots")
        mask = history["history_mask"]
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ContractError("history_mask must contain only 0.0 or 1.0")
        real = int(mask.sum())
        if real != len(window):
            raise ContractError("history_mask cannot pad missing expected applied observations")
        expected = np.zeros(self.history_len, dtype=np.float32)
        if real:
            expected[-real:] = 1
        if not np.array_equal(mask, expected):
            raise ContractError("history_mask must be contiguous and match the explicit padding_side")
        for name in _HISTORY_VALUES:
            if np.any(history[name][mask == 0] != 0):
                raise ContractError(f"{name}: padded values must be zero")
        rounds = []
        expected_rounds = [row.from_state for row in window]
        for index, entry in enumerate(context.entries):
            if mask[index] == 0:
                if entry is not None:
                    raise ContractError("Padded history slots must have no provenance entry")
                continue
            if not isinstance(entry, HistoryEntry):
                raise ContractError("Real history slots require HistoryEntry provenance")
            _integer(entry.update_round, where="history update_round")
            if entry.update_round >= context.ranking_state:
                raise ContractError("History update must be applied before the ranking state")
            rounds.append(entry.update_round)
            transition = window[len(rounds)-1]
            if entry.update_round != transition.from_state or entry.transition_id != transition.transition_id:
                raise ContractError("History window/provenance must match the last applied transitions")
            _mapping_keys(entry.observations, _HISTORY_VALUES, where="history observations")
            for name, observation in entry.observations.items():
                if not isinstance(observation, ObservationAvailability):
                    raise ContractError(f"{name}: missing ObservationAvailability provenance")
                if not isinstance(observation.source, str):
                    raise ContractError(f"{name}: history source must be a string")
                _integer(observation.available_at_state, where=f"{name}.available_at_state")
                if observation.source != "applied_h1":
                    raise ContractError(f"{name}: unsupported history source {observation.source!r}")
                earliest = entry.update_round + 1
                if not earliest <= observation.available_at_state <= context.ranking_state:
                    raise ContractError(f"{name}: outcome unavailable at ranking or before its endpoint")
                if observation.available_at_state != transition.available_at_state:
                    raise ContractError("History observation availability does not match applied provenance")
            deltas = accuracy_deltas(transition.parent_accuracy, transition.endpoint_accuracy)
            for name, expected_value in zip(_HISTORY_VALUES, (weighted_utility(deltas), *deltas)):
                if history[name][index] != expected_value:
                    raise ContractError(f"{name}: value does not match applied H1 accuracy evidence")
        if len(set(rounds)) != len(rounds):
            raise ContractError("History contains duplicate update rounds")
        if rounds != expected_rounds:
            raise ContractError("History entries do not match the explicit history order")


def load_feature_contract(path, *, dimensions=None, layer_order=None):
    """Load the supplied draft format safely; never infer dimensions or freeze it.

    dimensions must explicitly bind num_lora_layers and history_len for input
    validation. Missing bindings remain reportable; history_len must match the
    schema declaration. Derived prose is retained in the fingerprint, not executed.
    """
    raw = Path(path).read_bytes()
    try:
        document = yaml.load(raw.decode("utf-8"), Loader=_UniqueSafeLoader)
    except (yaml.YAMLError, UnicodeError) as error:
        raise ContractError(f"Invalid safe YAML schema: {error}") from error
    required = ("schema_version", "frozen_on", *_INPUT_FIELDS, "labels", "derived")
    _mapping_keys(document, required, where="schema")
    if document["schema_version"] != SEMANTIC_VERSION:
        raise ContractError(f"Unsupported schema_version; expected {SEMANTIC_VERSION}")
    frozen_on = document["frozen_on"]
    if frozen_on is not None:
        try:
            frozen_on = date.fromisoformat(str(frozen_on)).isoformat()
        except ValueError as error:
            raise ContractError("frozen_on must be null or an ISO date, not approval") from error
    _mapping_keys(document["derived"], _DERIVED_KEYS, where="derived")
    if any(not isinstance(value, str) for value in document["derived"].values()):
        raise ContractError("derived: expected descriptive strings, not executable definitions")
    groups = []
    for name, group in document.items():
        if name not in (*_INPUT_FIELDS, "labels"):
            continue
        controls = ("history_len",) if name == "recursive_history" else ()
        if name == "short_probe_dynamics":
            controls = ("probe_steps",)
        _mapping_keys(group, ("scope", "fields", *controls), where=name)
        expected_scope = "per_state" if name in ("current_state_performance", "recursive_history") else "per_candidate"
        if group["scope"] != expected_scope:
            raise ContractError(f"{name}: expected scope {expected_scope}")
        if not isinstance(group["fields"], list) or not group["fields"]:
            raise ContractError(f"{name}.fields: expected a nonempty ordered list")
        fields = []
        names = []
        for declaration in group["fields"]:
            _mapping_keys(declaration, ("name", "dtype", "shape"), ("description",), where=f"{name}.field")
            field_name = declaration["name"]
            if not isinstance(field_name, str) or not field_name:
                raise ContractError(f"{name}: field name must be a nonempty string")
            dtype = "int32" if field_name == "current_gen_index" else "float32"
            if declaration["dtype"] != dtype:
                raise ContractError(f"{name}.{field_name}: expected dtype {dtype}")
            shape = [1]
            if field_name == "layerwise_grad_norms":
                shape = ["num_lora_layers"]
            elif name == "recursive_history" and field_name != "current_gen_index":
                shape = ["history_len"]
            declared_shape = declaration["shape"]
            if (not isinstance(declared_shape, list) or declared_shape != shape
                    or any(type(actual) is not type(expected) for actual, expected in zip(declared_shape, shape))):
                raise ContractError(f"{name}.{field_name}: unsupported shape {declared_shape!r}; expected {shape}")
            description = declaration.get("description")
            if description is not None and not isinstance(description, str):
                raise ContractError(f"{name}.{field_name}: description must be a string")
            names.append(field_name)
            fields.append(FieldSpec(field_name, dtype, tuple(declared_shape), description))
        expected_fields = _LABEL_FIELDS if name == "labels" else _INPUT_FIELDS[name]
        if len(names) != len(set(names)) or set(names) != set(expected_fields):
            raise ContractError(f"{name}: duplicate, missing or unknown schema fields")
        groups.append(GroupSpec(name, expected_scope, tuple(fields)))
    history_len = document["recursive_history"]["history_len"]
    probe_steps = document["short_probe_dynamics"]["probe_steps"]
    _integer(history_len, where="history_len", minimum=1)
    if history_len != 3:
        raise ContractError("D10 requires history_len=3")
    _integer(probe_steps, where="probe_steps", minimum=1)
    dimensions = {} if dimensions is None else dimensions
    _mapping_keys(dimensions, (), ("num_lora_layers", "history_len"), where="dimensions")
    bindings = dict(dimensions)
    for name, value in bindings.items():
        _integer(value, where=name, minimum=1)
    if "history_len" in bindings and bindings["history_len"] != history_len:
        raise ContractError("Configured history_len does not match the schema")
    order = None if layer_order is None else _layer_order(layer_order)
    if order is not None and "num_lora_layers" in bindings and len(order) != bindings["num_lora_layers"]:
        raise ContractError("layer_order length must equal explicit num_lora_layers")
    signature = {
        "schema_version": document["schema_version"], "frozen_on": frozen_on,
        "groups": [
            {"name": group.name, "scope": group.scope, "fields": [field.__dict__ for field in group.fields]}
            for group in groups
        ],
        "history_len": history_len, "probe_steps": probe_steps, "derived": document["derived"],
    }
    schema_fingerprint = _fingerprint(signature)
    fingerprint = _fingerprint({"schema": schema_fingerprint, "dimensions": bindings, "layer_order": order})
    return FeatureContract(
        document["schema_version"], frozen_on, tuple(groups), MappingProxyType(bindings),
        order, history_len, probe_steps, hashlib.sha256(raw).hexdigest(), schema_fingerprint, fingerprint,
    )
