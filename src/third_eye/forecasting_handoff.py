"""Opt-in Sprint 2 JSONL boundary, not a measured-feature producer or training hook.

Hashes bind supplied definitions/configuration, not their approval or truth.
No missing feature or observation is imputed. Sprint 1 is not imported.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .feature_contract import (
    AppliedTransition, ContractError, TrajectoryIdentity, accuracy_deltas,
    build_applied_history, load_feature_contract, weighted_utility, _integer, _sha,
)
from .forecasting.data import from_records, fixed_split, validate_records
from .forecasting.layout import CANDIDATE, HISTORY, STATE, FeatureGroup, FeatureLayout
from .sprint2_protocol import (
    LineageRecipe, _digest, build_continuation_plan, bind_child_failure_batch,
    group_macro_weights, CandidateKey, verify_lineage_splits,
    build_pilot_state_plan,
)


def raw_module_layout(contract):
    """Keep YAML order; relocate generation index, separate mask, never depth-bin."""
    if contract.layer_order is None:
        raise ContractError("Explicit LoRA module manifest required")
    groups = []
    for group in contract.input_groups:
        if group.name == "recursive_history":
            groups.append(FeatureGroup("generation_index", STATE, ("current_gen_index",)))
            channels = tuple(f.name for f in group.fields
                             if f.name not in ("current_gen_index", "history_mask"))
            groups.append(FeatureGroup(group.name, HISTORY, channels))
            continue
        columns = []
        for field in group.fields:
            shape = contract.resolved_shape(field)
            columns.extend((field.name,) if shape == (1,) and field.name != "layerwise_grad_norms"
                           else tuple("%s[%d]" % (field.name, i) for i in range(shape[0])))
        groups.append(FeatureGroup(group.name, STATE if group.scope == "per_state" else CANDIDATE,
                                   tuple(columns)))
    return FeatureLayout("sprint2-raw-module-" + contract.fingerprint[:16], tuple(groups),
                         history_len=contract.history_len, h1_heads=("target", "ood", "retention"))


def make_handoff_spec(contract, bindings, *, synthetic_only):
    if type(synthetic_only) is not bool:
        raise ContractError("Explicit boolean synthetic evidence tag required")
    if bindings["definitions"]["utility_weights"] != [0.50, 0.25, 0.25]:
        raise ContractError("Handoff definitions must retain the accepted utility weights")
    layout = raw_module_layout(contract)
    spec = {
        "format": "third-eye-sprint2-handoff-v1", "synthetic_only": synthetic_only,
        "schema_version": contract.schema_version, "schema_source_sha256": contract.source_sha256,
        "schema_fingerprint": contract.schema_fingerprint, "contract_fingerprint": contract.fingerprint,
        "dimensions": dict(contract.dimensions), "layer_order": list(contract.layer_order),
        "layout": layout.to_dict(), "layout_fingerprint": layout.fingerprint(),
        "unresolved_definitions": list(contract.unresolved_definitions()), "bindings": bindings,
    }
    spec["semantic_hash"] = _digest(spec)
    return spec


def load_handoff_spec(path, *, schema_path=None):
    spec = json.loads(Path(path).read_text())
    schema_path = schema_path or Path(__file__).resolve().parents[2] / "configs/feature_schema.yaml"
    contract = load_feature_contract(schema_path, dimensions=spec["dimensions"], layer_order=spec["layer_order"])
    expected = make_handoff_spec(contract, spec["bindings"], synthetic_only=spec["synthetic_only"])
    if spec != expected or type(spec["synthetic_only"]) is not bool:
        raise ContractError("Handoff schema/layout/definition hash mismatch")
    return spec, contract, raw_module_layout(contract)


def continuation_for(spec, group, state):
    binding = spec["bindings"]["continuation"]
    if binding.get("exclude_h1") is True:
        return build_pilot_state_plan(group_id=group, state=state, recipe=binding["recipe"],
            prompt_pool=binding["prompt_pools"][group], candidate_master_seed=binding["candidate_master_seed"],
            continuation_master_seed=binding["master_seed"]).continuation
    return build_continuation_plan(group_id=group, state=state, recipe=binding["recipe"],
        prompt_pool=binding["prompt_pools"][group], prompt_count=binding["prompt_count"],
        master_seed=binding["master_seed"])


def export_candidate(contract, spec, *, inputs, history_context, model, task, candidate_id,
                     ranking_accuracy, h1, h2, corrected_failures, plan_sealed_at, zero_gradient_diagnostics):
    """Export supplied arrays and observed accuracy endpoints; no feature extraction.

    H1/H2 dictionaries include accuracy, endpoint_id, parent/endpoint SHA-256 and observed_at.
    Their observations occur AFTER the recorded ranking cutoff, never in history.
    """
    contract.validate_inputs(inputs, layer_order=contract.layer_order, history_context=history_context)
    _integer(candidate_id, where="candidate_id")
    if candidate_id >= 3:
        raise ContractError("K=3 candidate ID required")
    identity = history_context.identity
    projected = {key: float(inputs["current_state_performance"][name][0]) for key, name in zip(
        ("target", "ood", "retention"),
        ("target_val_accuracy", "ood_proxy_accuracy", "retention_anchor_accuracy"))}
    _check_accuracy_projection(projected, ranking_accuracy)
    parent = ranking_accuracy
    labels = {}
    for horizon, measurement in (("h1", h1), ("h2", h2)):
        deltas = accuracy_deltas(parent, measurement["accuracy"])
        labels.update({"delta_%s_%s" % (key, horizon): float(value)
                       for key, value in zip(("target", "ood", "retention"), deltas)})
    labels["lhv2"] = float(weighted_utility([labels["delta_%s_h2" % key]
                                           for key in ("target", "ood", "retention")]))
    layout = raw_module_layout(contract)
    state, candidate = {}, {}
    for group in contract.input_groups:
        if group.name == "recursive_history":
            continue
        dest = state if group.scope == "per_state" else candidate
        for field in group.fields:
            value = inputs[group.name][field.name]
            if field.name == "layerwise_grad_norms":
                dest.update({"%s[%d]" % (field.name, i): float(v) for i, v in enumerate(value)})
            else:
                dest[field.name] = float(value[0])
    history = inputs["recursive_history"]
    state["current_gen_index"] = int(history["current_gen_index"][0])
    plan = continuation_for(spec, identity.group_id, history_context.ranking_state)
    batch = bind_child_failure_batch(plan, candidate_id=candidate_id, corrected_failures=corrected_failures)
    return _json_form({
        "model": model, "task": task, "trajectory_id": identity.trajectory_id,
        "group_id": identity.group_id, "state_id": history_context.ranking_state,
        "candidate_id": candidate_id, "state": state, "candidate": candidate,
        "history": {"values": {name: history[name].tolist() for name in layout.history_channels},
                    "mask": history["history_mask"].tolist()},
        "labels": labels, "semantic_hash": spec["semantic_hash"],
        "provenance": {
            "synthetic": spec["synthetic_only"], "identity": asdict(identity),
            "ranking_parent_sha256": history_context.ranking_parent_sha256,
            "ranking_accuracy": ranking_accuracy,
            "cutoff": history_context.cutoff, "ledger": [asdict(row) for row in history_context.ledger],
            "h1": h1, "h2": h2, "continuation": asdict(plan),
            "plan_sealed_at": plan_sealed_at, "child_batch": asdict(batch),
            "zero_gradient_diagnostics": zero_gradient_diagnostics,
        },
    })


def _json_form(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _check_accuracy_projection(projected, measured):
    accuracy_deltas(measured, measured)
    if any(projected[k] != float(np.float32(measured[k])) for k in ("target", "ood", "retention")):
        raise ContractError("Input accuracy differs from float32 projection of recorded measurement")


def _source_inputs(record, contract):
    values = {}
    for group in contract.input_groups:
        fields = {}
        for field in group.fields:
            if group.name == "recursive_history":
                raw = ([record["state"]["current_gen_index"]] if field.name == "current_gen_index"
                       else record["history"]["mask"] if field.name == "history_mask"
                       else record["history"]["values"][field.name])
            else:
                source = record["state"] if group.scope == "per_state" else record["candidate"]
                raw = ([source["%s[%d]" % (field.name, i)] for i in range(contract.resolved_shape(field)[0])]
                       if field.name == "layerwise_grad_norms" else [source[field.name]])
            if field.dtype == "int32":
                if any(type(v) is not int or not 0 <= v < 5 for v in raw):
                    raise ContractError("Generation index must preserve source int32/T=5 semantics")
            fields[field.name] = np.asarray(raw, dtype=field.dtype)
        values[group.name] = fields
    return values


def validate_handoff(records, spec, contract, layout, *, manifest):
    """Join supplied identities, plans and applied results, not just numeric shapes.

    Caller-asserted hashes/ticks cannot prove a run happened or that settings were
    sealed beforehand. Dataset/scorer/calibration bindings remain review blockers.
    """
    if spec != make_handoff_spec(contract, spec["bindings"], synthetic_only=spec["synthetic_only"]):
        raise ContractError("Handoff spec hash mismatch")
    if layout.fingerprint() != spec["layout_fingerprint"]:
        raise ContractError("Handoff layout order mismatch")
    validate_records(records, layout, expected_semantic_hash=spec["semantic_hash"])
    fixed_split(from_records(records, layout, validate=False), manifest)
    binding = spec["bindings"]["lineage"]
    if (set(binding["membership"].values()) != set(binding["splits"])
            or [list(binding["splits"].values()).count(name) for name in ("train", "validation", "test")]
            != [8, 2, 2]):
        raise ContractError("Spec must preserve the full preassigned 8/2/2 plan; restrict only the CLI manifest")
    recipes = [LineageRecipe(row["trajectory_id"], row["adapted_root_id"], tuple(row["prefix_ids"]),
                            tuple(row["source_families"])) for row in binding["recipes"]]
    if set(binding["membership"]) != {r.trajectory_id for r in recipes}:
        raise ContractError("Lineage recipes and membership differ")
    verify_lineage_splits(recipes, binding["membership"], binding["splits"])
    if set(binding["identities"]) != set(binding["membership"]):
        raise ContractError("Pinned identities and planned trajectories differ")
    actual_roots = [LineageRecipe(r.trajectory_id, r.adapted_root_id,
        r.prefix_ids + (binding["identities"][r.trajectory_id]["root_sha256"],), r.source_families)
        for r in recipes]
    verify_lineage_splits(actual_roots, binding["membership"], binding["splits"])
    _integer(binding["assigned_at"], where="split assignment tick")
    group_macro_weights([CandidateKey(r["group_id"], r["trajectory_id"], r["state_id"], r["candidate_id"])
                         for r in records])
    by_key = {(r["trajectory_id"], r["state_id"], r["candidate_id"]): r for r in records}
    siblings, trajectory_roots, evaluation_ids, endpoints = {}, {}, set(), set()
    for record in records:
        p = record["provenance"]
        if any(type(v) not in (int, float) for v in record["history"]["mask"]):
            raise ContractError("History mask must be numeric, not boolean")
        lhv = record["labels"]["lhv2"]
        if type(lhv) not in (int, float) or not np.isfinite(lhv):
            raise ContractError("LHV2 must be a finite numeric label, not boolean")
        if type(p["synthetic"]) is not bool or p["synthetic"] != spec["synthetic_only"]:
            raise ContractError("Synthetic evidence tag mismatch")
        identity = TrajectoryIdentity(**p["identity"])
        if (identity.trajectory_id != record["trajectory_id"] or identity.group_id != record["group_id"]
                or binding["membership"].get(identity.trajectory_id) != identity.group_id):
            raise ContractError("Record identity/membership mismatch")
        if p["identity"] != binding["identities"][identity.trajectory_id]:
            raise ContractError("Record identity differs from pinned corpus/root/evaluation identity")
        if record["model"] != spec["bindings"]["model"] or record["task"] != spec["bindings"]["task"]:
            raise ContractError("Record model/task differs from configured binding")
        split = binding["splits"][identity.group_id]
        if manifest[identity.group_id] != ("val" if split == "validation" else split):
            raise ContractError("Collection split differs from preassigned lineage split")
        previous = trajectory_roots.setdefault(identity.trajectory_id, identity)
        if identity != previous:
            raise ContractError("Trajectory identity changed")
        evaluation_ids.add(identity.evaluation_sha256)
        ledger = []
        for raw in p["ledger"]:
            row = dict(raw)
            row["identity"] = TrajectoryIdentity(**row["identity"])
            ledger.append(AppliedTransition(**row))
        _, context = build_applied_history(ledger, identity=identity,
            ranking_state=record["state_id"], ranking_parent_sha256=p["ranking_parent_sha256"], cutoff=p["cutoff"])
        inputs = _source_inputs(record, contract)
        contract.validate_inputs(inputs, layer_order=contract.layer_order, history_context=context)
        projected = {key: float(inputs["current_state_performance"][name][0]) for key, name in zip(
            ("target", "ood", "retention"),
            ("target_val_accuracy", "ood_proxy_accuracy", "retention_anchor_accuracy"))}
        parent = p["ranking_accuracy"]
        _check_accuracy_projection(projected, parent)
        if ledger and parent != ledger[-1].endpoint_accuracy:
            raise ContractError("Ranking accuracies differ from observed applied endpoint")
        for row in ledger:
            prior = by_key.get((identity.trajectory_id, row.from_state, row.candidate_id))
            if prior is None:
                raise ContractError("Missing selected candidate record for applied-history join")
            measured = prior["provenance"]["h1"]
            if (row.parent_sha256 != measured["parent_sha256"]
                    or row.endpoint_sha256 != measured["endpoint_sha256"]
                    or row.observed_at != measured["observed_at"]
                    or row.parent_accuracy != prior["provenance"]["ranking_accuracy"]
                    or row.endpoint_accuracy != measured["accuracy"]):
                raise ContractError("Applied history does not match selected candidate's H1 observation")
        _integer(p["plan_sealed_at"], where="plan seal tick")
        if not binding["assigned_at"] <= p["plan_sealed_at"] <= context.cutoff:
            raise ContractError("Split/continuation not sealed before ranking cutoff")
        plan = continuation_for(spec, identity.group_id, record["state_id"])
        if p["continuation"] != _json_form(asdict(plan)):
            raise ContractError("Sibling continuation differs from state-keyed shared draw")
        child = p["child_batch"]
        batch = bind_child_failure_batch(plan, candidate_id=record["candidate_id"],
                                        corrected_failures=dict(child["examples"]))
        if child != _json_form(asdict(batch)):
            raise ContractError("Child batch differs from permitted failure subset / RNG stream")
        expected_parent, tick = context.ranking_parent_sha256, context.cutoff
        for horizon in ("h1", "h2"):
            measurement = p[horizon]
            _sha(measurement["parent_sha256"])
            _sha(measurement["endpoint_sha256"])
            _integer(measurement["observed_at"], where="label observation tick")
            if measurement["parent_sha256"] != expected_parent or measurement["observed_at"] <= tick:
                raise ContractError("H1/H2 endpoint chain or observation ordering mismatch")
            endpoint_id = measurement["endpoint_id"]
            if not isinstance(endpoint_id, str) or not endpoint_id.strip() or endpoint_id in endpoints:
                raise ContractError("Duplicate descendant identity")
            endpoints.add(endpoint_id)
            deltas = accuracy_deltas(parent, measurement["accuracy"])
            for key, delta in zip(("target", "ood", "retention"), deltas):
                if record["labels"]["delta_%s_%s" % (key, horizon)] != float(delta):
                    raise ContractError("Label differs from endpoint-minus-ranking-parent accuracy")
            expected_parent, tick = measurement["endpoint_sha256"], measurement["observed_at"]
        contract.validate_labels({"labels": {field.name: np.array([record["labels"][field.name]], np.float32)
                                  for field in contract.label_group.fields}})
        if record["labels"]["lhv2"] != float(weighted_utility(
                [record["labels"]["delta_%s_h2" % k] for k in ("target", "ood", "retention")])):
            raise ContractError("Serialized LHV2 must match H2 deltas exactly")
        diagnostics = p["zero_gradient_diagnostics"]
        fractions = ("zero_in_a", "zero_in_b", "zero_in_both")
        if set(diagnostics) != {"candidate_zero_norm", "retention_zero_norm", *fractions}:
            raise ContractError("Explicit zero-gradient diagnostics required")
        if (any(type(diagnostics[k]) is not bool for k in ("candidate_zero_norm", "retention_zero_norm"))
                or any(type(diagnostics[k]) not in (int, float) or not np.isfinite(diagnostics[k])
                       or not 0 <= diagnostics[k] <= 1 for k in fractions)):
            raise ContractError("Invalid zero-gradient diagnostics")
        a, b, both = (diagnostics[k] for k in fractions)
        if not max(0, a+b-1) <= both <= min(a, b):
            raise ContractError("Inconsistent zero-sign intersection fractions")
        shared = (identity, p["ranking_parent_sha256"], parent, p["cutoff"], p["ledger"],
                  p["continuation"], p["plan_sealed_at"])
        key = (identity.trajectory_id, record["state_id"])
        if siblings.setdefault(key, shared) != shared:
            raise ContractError("Siblings disagree on ranking/history/continuation provenance")
    if len(evaluation_ids) != 1:
        raise ContractError("This same-task handoff must use one fixed evaluation identity")
