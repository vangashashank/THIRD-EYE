"""Offline preflight/dry-run by default; --execute requires an existing allocation."""
import argparse
import json
import os
import socket
import hashlib
import importlib.metadata
import platform
import time
from pathlib import Path

from src.third_eye.feature_contract import ContractError
from src.third_eye.sprint2_collection import (RunStore, inspect_bindings, load_config, binding_blockers,
    prepare_spec, write_json)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/sprint2_collection.yaml")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--trajectories", nargs="+", help="One assigned train trajectory first, six only after review")
    ap.add_argument("--prepare-spec", action="store_true", help="Seal supplied complete bindings; never infers agreements")
    args = ap.parse_args(argv)
    if sum((args.preflight, args.dry_run, args.execute)) > 1:
        ap.error("Choose only one mode")
    try:
        config = load_config(args.config)
        missing = binding_blockers(config)
        if missing:
            print(json.dumps({"status": "blocked_before_model_load", "blockers": missing}, indent=2))
            return 2
        if args.prepare_spec:
            if args.execute or args.resume or args.out:
                raise ContractError("Prepare-spec only, no execution/resume/output run")
            base = Path(args.config).resolve().parent
            spec = prepare_spec(config, base, Path(__file__).resolve().parents[1]/"configs/feature_schema.yaml")
            inspect_bindings(config, base, prepared_spec=spec)
            write_json(base/config["handoff_spec"], spec)
            print("Sealed supplied bindings (not measurements or research readiness):", base/config["handoff_spec"])
            return 0
        spec, contract, layout, populations = inspect_bindings(config, Path(args.config).resolve().parent)
        trajectories = args.trajectories or []
        lineage = spec["bindings"]["lineage"]
        if args.execute or args.dry_run:
            if not trajectories or len(trajectories) not in (1, 6) or len(set(trajectories)) != len(trajectories):
                raise ContractError("Choose one assigned train T=5 trajectory, or reviewed six-trajectory batch")
            groups = {lineage["membership"][t] for t in trajectories}
            if len(groups) != len(trajectories) or any(lineage["splits"][g] != "train" for g in groups):
                raise ContractError("Initial batch needs distinct preassigned training groups")
        if not args.execute:
            print(json.dumps({"status": "offline_bindings_valid_no_measurements",
                "states": 5*len(trajectories), "records": 15*len(trajectories),
                "full_updates": 30*len(trajectories), "temporary_probes": 15*len(trajectories),
                "semantic_hash": spec["semantic_hash"], "synthetic": False}, indent=2))
            return 0
        if not args.out or not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("dgx-login"):
            raise ContractError("Execution needs --out and a separately authorized existing compute allocation")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractError("Expected exactly one allocated CUDA/MIG device")
        props = torch.cuda.get_device_properties(0)
        if (config["retention"]["mig_profile"] not in props.name
                or props.total_memory != config["retention"]["device_total_memory_bytes"]):
            raise ContractError("Allocated MIG name/profile/memory differs from inspected configuration; stop and review fit")
        from src.third_eye.sprint2_backend import HFCollectionBackend
        execution_started = time.monotonic()
        store = RunStore(args.out, config, spec, resume=args.resume)
        if not args.resume:
            project = Path(__file__).resolve().parents[1]
            sources = list((project/"src").rglob("*.py"))+list((project/"scripts").glob("*.py"))
            write_json(store.path/"environment.json", {"python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")},
                "job_id": os.environ["SLURM_JOB_ID"], "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(),
                "device_total_memory_bytes": props.total_memory,
                "source_hashes": {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
        else:
            from src.third_eye.sprint2_collection import read_json
            saved = read_json(store.path/"environment.json")
            project = Path(__file__).resolve().parents[1]
            if any(hashlib.sha256((project/p).read_bytes()).hexdigest() != sha for p, sha in saved["source_hashes"].items()):
                raise ContractError("Resume source changed; retain prior run and version the corpus")
        backend = HFCollectionBackend(config, Path(args.config).resolve().parent, spec, populations)
        backend.execution_started = execution_started
        backend.deadline = execution_started + config["retention"]["max_device_seconds"] - 120
        backend.output_path = store.path
        out = store.collect(config, contract, backend, trajectories)
        print("Complete measured records (not forecasting performance):", out)
        return 0
    except (ContractError, ValueError, OSError, KeyError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
