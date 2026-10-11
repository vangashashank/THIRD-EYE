"""Prepare a small real throughput check; execution needs separate authorization/allocation."""
import argparse
import json
import os
import socket
import time
from pathlib import Path

from src.third_eye.feature_contract import ContractError
from src.third_eye.sprint2_collection import binding_blockers, inspect_bindings, load_config, write_json
from src.third_eye.sprint2_throughput import measure_backend, operation_counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/sprint2_collection.yaml")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--out")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    try:
        config = load_config(args.config)
        blockers = binding_blockers(config)
        if blockers:
            print(json.dumps({"status": "blocked_before_model_load", "blockers": blockers}, indent=2))
            return 2
        base = Path(args.config).resolve().parent
        spec, _, _, populations = inspect_bindings(config, base)
        lineage = spec["bindings"]["lineage"]
        group = lineage["membership"][args.trajectory]
        if lineage["splits"][group] != "train":
            raise ContractError("Throughput readiness check must use an assigned training group")
        if not args.execute:
            print(json.dumps({"status": "offline_only_no_measurements", "purpose": "throughput_only_not_collection",
                "source_prompts": 32, "diagnostic_samples": 12, "optimizer_updates": config["update"]["steps"],
                "pilot_operation_counts": operation_counts()}, indent=2))
            return 0
        if not args.out or not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("dgx-login"):
            raise ContractError("Execution needs a new --out and separately authorized existing compute allocation")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractError("Expected one allocated CUDA/MIG device")
        props = torch.cuda.get_device_properties(0)
        if config["retention"]["mig_profile"] not in props.name or props.total_memory != config["retention"]["device_total_memory_bytes"]:
            raise ContractError("MIG profile/memory differs from the inspected configuration")
        start = time.monotonic()
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"sealed.json", {"config": config, "spec": spec, "purpose": "throughput_only_not_collection"})
        from src.third_eye.sprint2_backend import HFCollectionBackend
        try:
            backend = HFCollectionBackend(config, base, spec, populations)
            backend.deadline = start+config["retention"]["max_device_seconds"]-120
            torch.cuda.reset_peak_memory_stats()
            report = measure_backend(backend, args.trajectory, group, output, synchronize=torch.cuda.synchronize)
        except BaseException as error:
            write_json(output/"interrupted.json", {"error_type": type(error).__name__, "message": str(error),
                "elapsed_seconds": time.monotonic()-start, "partial_evidence_retained": True})
            raise
        write_json(output/"resources.json", {"elapsed_seconds_including_model_load": time.monotonic()-start,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "job_id": os.environ["SLURM_JOB_ID"],
            "device_name": props.name, "device_total_memory_bytes": props.total_memory})
        print(json.dumps(report, indent=2))
        return 0
    except (ContractError, OSError, ValueError, KeyError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
