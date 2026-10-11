"""Small measured readiness check, NOT collection, labels, or forecasting evaluation."""
import time
from pathlib import Path

from .feature_contract import ContractError
from .sprint2_collection import file_sha, write_json
from .sprint2_protocol import build_pilot_state_plan
from .sprint2_runtime import restored_training_state


def operation_counts(groups=1):
    if type(groups) is not int or groups < 1:
        raise ContractError("Positive integer lineage-group count required")
    return {"groups": groups, "states": 5*groups, "records": 15*groups,
            "full_updates": 960*groups, "probe_updates": 225*groups,
            "optimizer_updates": 1185*groups, "endpoint_scoring_generations": 11200*groups,
            "endpoint_adapters": 30*groups}


def measure_backend(backend, trajectory, group, output, *, synchronize=lambda: None, clock=time.monotonic):
    """Use candidate-0's sealed 32-source draw; one approved 32-step update.

    Four pinned examples per diagnostic role are throughput samples ONLY, never
    substitutes for collection's 32/32/256. No H2 branch, labels, promotion or
    history is produced. Finally restores the complete root/training state.
    The output directory must already be new; partial evidence is retained.
    """
    import torch

    output = Path(output)
    backend.output_path = backend.state_output_path = output
    timings = {}

    def timed(name, fn):
        backend.check_limits()
        synchronize()
        start = clock()
        result = fn()
        synchronize()
        timings[name] = clock()-start
        return result

    timed("root_load_seconds", lambda: backend.load_parent(trajectory, None))
    lineage = backend.spec["bindings"]["lineage"]
    parent = backend.parent_sha()
    if parent != lineage["identities"][trajectory]["root_sha256"]:
        raise ContractError("Actual benchmark root differs from audited lineage identity")
    binding = backend.config["continuation"]
    plan = build_pilot_state_plan(group_id=group, state=0,
        prompt_pool=binding["prompt_pools"][group], recipe=binding["recipe"],
        candidate_master_seed=binding["candidate_master_seed"],
        continuation_master_seed=binding["master_seed"]).candidates[0]
    write_json(output/"plan.json", {"purpose": "throughput_only_not_collection", "trajectory_id": trajectory,
        "group_id": group, "root_sha256": parent, "source_plan_sha256": plan.sha256,
        "source_prompt_ids": list(plan.prompt_ids), "update": backend.config["update"]})
    synchronize()
    start = clock()
    snapshot_context = restored_training_state(backend.model)
    with snapshot_context as snapshot:
        synchronize()
        timings["snapshot_seconds"] = clock()-start
        generated = []
        for role in ("target", "ood", "retention"):
            rows = backend._rows(role)[:4]
            if len(rows) != 4:
                raise ContractError("Need four actual diagnostic examples per throughput role")
            attempts = timed(role+"_generation_seconds", lambda: [backend._generate(r) for r in rows])
            generated.extend({"role": role, **a} for a in attempts)
        examples, corrections, source_attempts = timed("failure_batch_seconds", lambda: backend._failure_batch(group, plan))
        write_json(output/"generations.json", {"diagnostic_samples": generated, "source_attempts": source_attempts,
                   "verified_corrections": corrections})
        losses = timed("update_seconds", lambda: backend._update(examples, plan.training_seed))
        if len(losses) != backend.config["update"]["steps"]:
            raise ContractError("Benchmark update count differs from pinned settings")
        endpoint = backend.parent_sha()

        def save_adapter():
            backend.check_limits()
            with (output/"adapter_endpoint.pt").open("xb") as f:
                torch.save({"purpose": "throughput_only_not_collection", "semantic_hash": backend.spec["semantic_hash"],
                    "parameters": {n: p.detach().cpu().clone() for n, p in backend.model.named_parameters() if p.requires_grad}}, f)
        timed("adapter_save_seconds", save_adapter)
        write_json(output/"update.json", {"step_losses": losses})
        synchronize()
        restore_start = clock()
    synchronize()
    timings["restore_seconds"] = clock()-restore_start
    if backend.parent_sha() != parent:
        raise ContractError("Throughput check did not restore its parent")
    count = operation_counts()
    # This intentionally excludes feature forwards, probes, H2 corrections,
    # all-state restore overhead and initialization: it is NOT a fit prediction.
    scoring = 35*sum(backend.config["diagnostic_counts"][r]*timings[r+"_generation_seconds"]/4
                    for r in ("target", "ood", "retention"))
    estimate = count["full_updates"]*timings["update_seconds"]/len(losses) + scoring
    report = {"purpose": "throughput_only_not_collection", "synthetic_only": backend.spec["synthetic_only"],
        "trajectory_id": trajectory, "group_id": group, "semantic_hash": backend.spec["semantic_hash"],
        "parent_sha256": parent, "temporary_endpoint_sha256": endpoint,
        "restoration_verified": True, "full_restoration_fingerprint": snapshot.before,
        "timings": timings, "optimizer_updates_measured": len(losses),
        "diagnostic_generations_measured": 12,
        "diagnostic_generated_tokens": sum(a["generated_tokens"] for a in generated),
        "diagnostic_truncated": sum(a["truncated"] for a in generated),
        "source_generations_measured": sum(len(v) for v in source_attempts.values()),
        "source_generated_tokens": sum(a["generated_tokens"] for rows in source_attempts.values() for a in rows),
        "adapter_file_bytes": (output/"adapter_endpoint.pt").stat().st_size,
        "adapter_file_sha256": file_sha(output/"adapter_endpoint.pt"),
        "measured_output_bytes_before_report": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "one_group_operation_counts": count,
        "partial_workload_extrapolation_seconds": estimate,
        "feasibility_established": False,
        "limitations": "Only four fixed prompts/role and one root update: not representative across states, tasks or sequence lengths. "
                       "Excludes initialization, features/gradients/embeddings, 15-step probes, H2 correction generation, "
                       "all-state restoration and additional storage. Probe cost is not measured by the full-update rate. "
                       "No Gate 1, complete records, accuracy deltas, or two-hour/5-GiB feasibility claim."}
    write_json(output/"report.json", report)
    return report
