"""Fabricated integration fixtures ONLY; no extracted research features or labels.

Writes Mahitha's candidate JSONL, bare group-split manifest and an opt-in raw
module spec. All numbers, identities, ticks and child batches are synthetic.
No training/model/scorer/calibration is executed. Existing outputs are refused.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.third_eye.feature_contract import (
    AppliedTransition, TrajectoryIdentity, build_applied_history, load_feature_contract,
)
from src.third_eye.forecasting_handoff import (
    continuation_for, export_candidate, make_handoff_spec, raw_module_layout, validate_handoff,
)
from src.third_eye.sprint2_protocol import LineageRecipe, assign_lineage_splits, _digest


ROOT = Path(__file__).resolve().parents[1]


def make_fixture(states=2):
    if type(states) is not int or not 1 <= states <= 5:
        raise ValueError("Synthetic fixture must use 1..5 T=5 decision states")
    order = ("synthetic.block.2.q_proj", "synthetic.block.10.q_proj")
    contract = load_feature_contract(ROOT / "configs/feature_schema.yaml",
        dimensions={"num_lora_layers": 2, "history_len": 3}, layer_order=order)
    recipes = [LineageRecipe("synthetic-trajectory-%02d" % i, "synthetic-root-%02d" % i,
                              (), ("synthetic-source-family-%02d" % i,)) for i in range(12)]
    membership, splits = assign_lineage_splits(recipes)
    pools = {group: ["synthetic-%s-prompt-%d" % (group, i) for i in range(4)] for group in splits}
    evaluation = _digest("synthetic-fixed-diagnostics")
    identities = {r.trajectory_id: asdict(TrajectoryIdentity("synthetic-handoff-v1", membership[r.trajectory_id],
                  r.trajectory_id, _digest(r.adapted_root_id), evaluation)) for r in recipes}
    bindings = {
        "evidence": "synthetic fixtures, not extracted research features",
        "model": "synthetic-model-NOT-Qwen", "task": "synthetic-task",
        "lineage": {"recipes": [asdict(r) for r in recipes], "membership": dict(membership),
                    "splits": dict(splits), "assigned_at": 0, "identities": identities},
        "continuation": {"recipe": {"name": "synthetic-only-recipe", "steps": 1},
                         "prompt_pools": pools, "prompt_count": 2, "master_seed": 19},
        "definitions": {"values": "arbitrary synthetic values; no D06-D09/D12 approval",
                        "delta": "accuracy(endpoint)-accuracy(ranking_parent)",
                        "utility_weights": [0.50, 0.25, 0.25]},
        "generation_calibration": {"status": "unbound_not_run", "tasks": [], "caps": {}},
    }
    spec = make_handoff_spec(contract, bindings, synthetic_only=True)
    # Exercise all three manifest names without claiming a completed 8/2/2 corpus.
    selected = [next(r for r in recipes if splits[membership[r.trajectory_id]] == name)
                for name in ("train", "validation", "test")]
    manifest = {membership[r.trajectory_id]: "val" if splits[membership[r.trajectory_id]] == "validation"
                else splits[membership[r.trajectory_id]] for r in selected}
    records = []
    for recipe in selected:
        trajectory = recipe.trajectory_id
        identity = TrajectoryIdentity("synthetic-handoff-v1", membership[trajectory], trajectory,
                                      _digest(recipe.adapted_root_id), evaluation)
        parent_hash, parent_accuracy, ledger = identity.root_sha256, dict(target=.5, ood=.5, retention=.5), []
        for state in range(states):
            history, context = build_applied_history(ledger, identity=identity, ranking_state=state,
                ranking_parent_sha256=parent_hash, cutoff=state * 100 + 10)
            chosen = None
            for candidate_id in range(3):
                inputs = {g.name: {f.name: np.full(contract.resolved_shape(f), .125, dtype=f.dtype)
                                  for f in g.fields} for g in contract.input_groups}
                inputs["recursive_history"] = history
                for key, name in zip(("target", "ood", "retention"),
                    ("target_val_accuracy", "ood_proxy_accuracy", "retention_anchor_accuracy")):
                    inputs["current_state_performance"][name][0] = parent_accuracy[key]
                inputs["gradient_interference_stats"]["layerwise_grad_norms"][:] = [3, 4]
                inputs["gradient_interference_stats"]["grad_norm"][:] = 5
                inputs["candidate_data_stats"]["pre_update_nll"][:] = 1 + candidate_id
                change = ([0, 0, 0] if state == 1 and candidate_id == 0 else
                          [[1/32, 0, -1/32], [-1/32, 1/32, 0], [0, -1/32, 1/32]][candidate_id])
                endpoint = dict(zip(parent_accuracy, [parent_accuracy[k] + d
                                for k, d in zip(parent_accuracy, change)]))
                h1 = {"endpoint_id": "%s/s%d/c%d/h1" % (trajectory, state, candidate_id),
                      "parent_sha256": parent_hash, "endpoint_sha256": _digest([trajectory, state, candidate_id, "h1"]),
                      "accuracy": endpoint, "observed_at": state * 100 + 20}
                h2 = {"endpoint_id": "%s/s%d/c%d/h2" % (trajectory, state, candidate_id),
                      "parent_sha256": h1["endpoint_sha256"],
                      "endpoint_sha256": _digest([trajectory, state, candidate_id, "h2"]),
                      "accuracy": {k: endpoint[k] + ([0, -1/32, 1/32][candidate_id] if k == "target" else 0)
                                   for k in endpoint}, "observed_at": state * 100 + 30}
                plan = continuation_for(spec, identity.group_id, state)
                corrected = {plan.prompt_ids[candidate_id % 2]: "synthetic corrected answer %d" % candidate_id}
                records.append(export_candidate(contract, spec, inputs=inputs, history_context=context,
                    model="synthetic-model-NOT-Qwen", task="synthetic-task", candidate_id=candidate_id,
                    ranking_accuracy=parent_accuracy, h1=h1, h2=h2, corrected_failures=corrected,
                    plan_sealed_at=state * 100 + 5,
                    zero_gradient_diagnostics={"candidate_zero_norm": False, "retention_zero_norm": False,
                                               "zero_in_a": 0.0, "zero_in_b": 0.0, "zero_in_both": 0.0}))
                if candidate_id == 0:
                    chosen = h1
            ledger.append(AppliedTransition(identity, "synthetic-applied-%s-%d" % (trajectory, state),
                state, state+1, 0, parent_hash, chosen["endpoint_sha256"], parent_accuracy,
                chosen["accuracy"], state+1, chosen["observed_at"]))
            parent_hash, parent_accuracy = chosen["endpoint_sha256"], chosen["accuracy"]
    spec = json.loads(json.dumps(spec))
    validate_handoff(records, spec, contract, raw_module_layout(contract), manifest=manifest)
    return spec, records, manifest


def write_fixture(out, states=2):
    spec, records, manifest = make_fixture(states)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (("handoff_spec.json", spec), ("split_manifest.json", manifest)):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    (out / "records.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--states", type=int, default=2)
    args = parser.parse_args()
    out = write_fixture(args.out, args.states)
    print("Synthetic fixture ONLY:", out, "(%d records; NOT measured features or forecasting evidence)"
          % (args.states * 9))


if __name__ == "__main__":
    main()
