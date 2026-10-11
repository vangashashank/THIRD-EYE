"""Pure accepted Sprint 2 planning/evaluation helpers; no training or model imports."""
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .feature_contract import CORE_DECISIONS, ContractError, _integer


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ContractError("Expected a nonempty identity string")
    return value


def _digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ContractError("Plan metadata must be finite JSON data") from error
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class LineageRecipe:
    trajectory_id: str
    adapted_root_id: str
    prefix_ids: tuple
    source_families: tuple


def lineage_components(recipes):
    """Planned dependencies, not inferred statistical independence or measured labels."""
    recipes = tuple(recipes)
    by_id, owners, edges = {}, {}, defaultdict(set)
    for row in recipes:
        if not isinstance(row, LineageRecipe):
            raise ContractError("Expected LineageRecipe metadata")
        identity = _text(row.trajectory_id)
        if identity in by_id:
            raise ContractError("Duplicate trajectory ID")
        by_id[identity] = row
        if not isinstance(row.prefix_ids, tuple) or not isinstance(row.source_families, tuple):
            raise ContractError("Lineage dependencies must be explicit tuples")
        dependencies = [("ancestor", _text(row.adapted_root_id))]
        dependencies += [("ancestor", _text(value)) for value in row.prefix_ids]
        dependencies += [("source", _text(value)) for value in row.source_families]
        for token in dependencies:
            if token in owners:
                other = owners[token]
                edges[identity].add(other)
                edges[other].add(identity)
            else:
                owners[token] = identity
    unseen, components = set(by_id), []
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(edges[current] - component)
        unseen -= component
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def assign_lineage_splits(recipes, *, split_seed=20261010):
    _integer(split_seed, where="split_seed")
    components = lineage_components(recipes)
    if len(components) != 12:
        raise ContractError("The accepted preliminary 8/2/2 design needs 12 independent lineage groups")
    groups = {"group_" + _digest(component): component for component in components}
    ordered = sorted(groups, key=lambda group: (_digest([split_seed, group]), group))
    splits = {group: "train" if index < 8 else "validation" if index < 10 else "test"
              for index, group in enumerate(ordered)}
    membership = {trajectory: group for group, members in groups.items() for trajectory in members}
    return MappingProxyType(membership), MappingProxyType(splits)


def verify_lineage_splits(recipes, membership, splits):
    for component in lineage_components(recipes):
        if any(trajectory not in membership for trajectory in component):
            raise ContractError("Missing trajectory split membership")
        group_ids = {membership[trajectory] for trajectory in component}
        if len(group_ids) != 1 or any(group not in splits for group in group_ids):
            raise ContractError("Shared roots/prefixes/source families cross lineage groups")
        if any(splits[group] not in ("train", "validation", "test") for group in group_ids):
            raise ContractError("Unknown split")


@dataclass(frozen=True)
class LoraModule:
    block_index: int
    projection: str
    qualified_name: str


def canonical_lora_manifest(modules, *, placement_order):
    placement_order = tuple(placement_order)
    if not placement_order or len(set(placement_order)) != len(placement_order):
        raise ContractError("Supply a unique explicit placement order")
    for value in placement_order:
        _text(value)
    modules = tuple(modules)
    if not modules:
        raise ContractError("No actual adapted modules supplied")
    seen, positions = set(), set()
    for module in modules:
        if not isinstance(module, LoraModule):
            raise ContractError("Expected explicit module metadata, not guessed name parsing")
        _integer(module.block_index, where="block_index")
        _text(module.qualified_name)
        if (module.projection not in placement_order or module.qualified_name in seen
                or (module.block_index, module.projection) in positions):
            raise ContractError("Unknown/duplicate adapted module")
        seen.add(module.qualified_name)
        positions.add((module.block_index, module.projection))
    ordered = sorted(modules, key=lambda row: (row.block_index, placement_order.index(row.projection)))
    return len(ordered), tuple(row.qualified_name for row in ordered)


@dataclass(frozen=True)
class ContinuationPlan:
    group_id: str
    state: int
    recipe_json: str
    prompt_ids: tuple
    prompt_seed: int
    generation_seed: int
    training_seed: int
    correction_seed: int
    sha256: str


def _stream_seed(master_seed, group_id, state, stream):
    return int(_digest([master_seed, group_id, state, stream])[:15], 16)


def build_continuation_plan(*, group_id, state, recipe, prompt_pool, prompt_count, master_seed):
    """Exactly one standardized sibling draw; no candidate or outcome parameter.

    Each stream uses local identity-keyed RNG, not ambient/process RNG. Recipe,
    source prompts and common seeds are sealed before child outcomes exist.
    """
    _text(group_id)
    _integer(state, where="state")
    if state >= CORE_DECISIONS:
        raise ContractError("Expected one of T=5 decisions")
    _integer(master_seed, where="master_seed")
    _integer(prompt_count, where="prompt_count", minimum=1)
    if not isinstance(recipe, dict) or not recipe:
        raise ContractError("Supply an explicit continuation recipe")
    _digest(recipe)
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if not isinstance(prompt_pool, (list, tuple)):
        raise ContractError("Supply an explicit list/tuple of source prompt IDs")
    pool = tuple(sorted(_text(value) for value in prompt_pool))
    if len(set(pool)) != len(pool) or prompt_count > len(pool):
        raise ContractError("Prompt pool must be unique and cover the shared draw")
    seeds = [_stream_seed(master_seed, group_id, state, stream)
             for stream in ("source_draw", "generation", "training", "correction")]
    prompts = tuple(random.Random(seeds[0]).sample(pool, prompt_count))
    digest = _digest([group_id, state, encoded, prompts, seeds])
    return ContinuationPlan(group_id, state, encoded, prompts, *seeds, digest)


def prompt_processing_seed(plan, prompt_id, *, stream):
    if prompt_id not in plan.prompt_ids or stream not in ("generation", "correction"):
        raise ContractError("Unknown prompt/processing stream")
    seed = plan.generation_seed if stream == "generation" else plan.correction_seed
    return int(_digest([plan.sha256, seed, prompt_id, stream])[:15], 16)


@dataclass(frozen=True)
class PilotStatePlan:
    candidates: tuple
    continuation: ContinuationPlan
    overlaps: tuple
    excluded_h1_prompt_ids: tuple


def build_pilot_state_plan(*, group_id, state, prompt_pool, candidate_master_seed,
                           continuation_master_seed, recipe):
    """Accepted first pilot: 3 x 32 H1 draws; H2 excludes their union, no redraw.

    Candidate seeds include (group,state,candidate); the shared continuation
    seed/draw includes ONLY (group,state). No outcomes or failure batches enter.
    """
    _integer(candidate_master_seed, where="candidate_master_seed")
    candidates = []
    for candidate in range(3):
        seed = int(_digest([candidate_master_seed, group_id, state, candidate, "h1_candidate"])[:15], 16)
        candidates.append(build_continuation_plan(group_id=group_id, state=state,
            recipe={"candidate_id": candidate, "execution": recipe}, prompt_pool=prompt_pool,
            prompt_count=32, master_seed=seed))
    excluded = set(p for plan in candidates for p in plan.prompt_ids)
    remaining = sorted(set(prompt_pool)-excluded)
    if len(remaining) < 32:
        raise ContractError("Insufficient H2 prompts after excluding all H1 source draws; stop, never redraw")
    continuation = build_continuation_plan(group_id=group_id, state=state, recipe=recipe,
        prompt_pool=remaining, prompt_count=32, master_seed=continuation_master_seed)
    overlaps = tuple((i, j, tuple(sorted(set(candidates[i].prompt_ids) & set(candidates[j].prompt_ids))))
                     for i in range(3) for j in range(i))
    return PilotStatePlan(tuple(candidates), continuation, overlaps, tuple(sorted(excluded)))


@dataclass(frozen=True)
class ChildFailureBatch:
    candidate_id: int
    plan_sha256: str
    examples: tuple
    training_seed: int


def bind_child_failure_batch(plan, *, candidate_id, corrected_failures):
    _integer(candidate_id, where="candidate_id")
    if candidate_id >= 3 or not isinstance(corrected_failures, dict):
        raise ContractError("Expected K=3 child and prompt-to-correction mapping")
    if not set(corrected_failures).issubset(plan.prompt_ids):
        raise ContractError("Child correction came from outside the shared source draw")
    examples = tuple((prompt, _text(corrected_failures[prompt]))
                     for prompt in plan.prompt_ids if prompt in corrected_failures)
    return ChildFailureBatch(candidate_id, plan.sha256, examples, plan.training_seed)


@dataclass(frozen=True)
class CandidateKey:
    group_id: str
    trajectory_id: str
    state: int
    candidate_id: int


def group_macro_weights(keys):
    keys = tuple(keys)
    if not keys or any(not isinstance(key, CandidateKey) for key in keys):
        raise ContractError("Expected CandidateKey identities")
    if len(set(keys)) != len(keys):
        raise ContractError("Expected unique complete candidate records")
    states = defaultdict(list)
    trajectories = defaultdict(set)
    groups = defaultdict(set)
    for index, key in enumerate(keys):
        if not isinstance(key, CandidateKey):
            raise ContractError("Expected CandidateKey identities")
        _text(key.group_id)
        _text(key.trajectory_id)
        _integer(key.state, where="state")
        _integer(key.candidate_id, where="candidate_id")
        if key.state >= CORE_DECISIONS or key.candidate_id >= 3:
            raise ContractError("Invalid T=5/K=3 identity")
        state = (key.group_id, key.trajectory_id, key.state)
        states[state].append(index)
        trajectories[state[:2]].add(key.state)
        groups[key.group_id].add(key.trajectory_id)
    weights = np.zeros(len(keys), dtype=np.float64)
    for (group, trajectory, state), indices in states.items():
        if {keys[index].candidate_id for index in indices} != {0, 1, 2}:
            raise ContractError("Missing expected candidates; do not silently reduce K")
        weights[indices] = 1 / (len(groups) * len(groups[group]) * len(trajectories[group, trajectory]) * 3)
    return weights


def _matrix(values, rows):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0
            or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
        raise ContractError("Expected finite numeric records with shape [rows, channels]")
    return values.astype(np.float64)


@dataclass(frozen=True)
class TrainingStatistics:
    mean: np.ndarray
    scale: np.ndarray
    observed_weight: np.ndarray

    def transform(self, values, *, observed=None):
        values = _matrix(values, len(values))
        if values.shape[1] != len(self.mean):
            raise ContractError("Normalizer channel shape mismatch")
        mask = _observation_mask(values, observed)
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.where(mask, (values - self.mean) / self.scale, 0.0)
        if not np.isfinite(result).all():
            raise ContractError("Nonfinite normalized values")
        return result


def _observation_mask(values, observed):
    mask = np.ones(values.shape, dtype=bool) if observed is None else np.asarray(observed)
    if mask.dtype != np.dtype(bool) or mask.shape != values.shape:
        raise ContractError("Observed-value mask must be aligned boolean data")
    if np.any(values[~mask] != 0):
        raise ContractError("Masked padding must be zero before normalization")
    return mask


def fit_training_statistics(values, keys, split_by_group, *, observed=None):
    keys = tuple(keys)
    weights = group_macro_weights(keys)
    if any(split_by_group.get(key.group_id) != "train" for key in keys):
        raise ContractError("Fit statistics/baselines on training groups only")
    values = _matrix(values, len(keys))
    mask = _observation_mask(values, observed)
    effective = weights[:, None] * mask
    denominator = effective.sum(0)
    mean = np.divide((effective * values).sum(0), denominator,
                     out=np.zeros(values.shape[1]), where=denominator > 0)
    variance = np.divide((effective * (values - mean)**2).sum(0), denominator,
                         out=np.zeros_like(mean), where=denominator > 0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1
    for array in (mean, scale, denominator):
        if not np.isfinite(array).all():
            raise ContractError("Nonfinite fitted statistics")
        array.setflags(write=False)
    return TrainingStatistics(mean, scale, denominator)


def prediction_errors(predictions, labels, keys):
    keys = tuple(keys)
    weights = group_macro_weights(keys)
    predicted, actual = _matrix(predictions, len(keys)), _matrix(labels, len(keys))
    if predicted.shape != actual.shape:
        raise ContractError("Prediction/label shape mismatch")
    with np.errstate(over="ignore", invalid="ignore"):
        differences = predicted - actual
        mse = (weights[:, None] * differences**2).sum(0)
        mae = (weights[:, None] * np.abs(differences)).sum(0)
    if not np.isfinite(mse).all() or not np.isfinite(mae).all():
        raise ContractError("Nonfinite prediction error")
    return {"mae": mae,
            "mse": mse, "rmse": np.sqrt(mse)}


def expected_selection_regret(utilities, ranking_scores=None):
    values = _matrix([utilities], 1)[0]
    if values.shape != (3,):
        raise ContractError("Regret requires the complete K=3 candidate set")
    scores = np.zeros(3) if ranking_scores is None else _matrix([ranking_scores], 1)[0]
    if scores.shape != (3,):
        raise ContractError("Ranking requires K=3 scores")
    tied = np.flatnonzero(scores == scores.max())
    return {"regret": float(values.max() - values[tied].mean()),
            "tie_size": len(tied), "selection_probabilities": np.isin(np.arange(3), tied) / len(tied)}
