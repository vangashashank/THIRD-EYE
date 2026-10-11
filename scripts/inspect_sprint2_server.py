"""Read-only university readiness inventory; stdlib only, no model/job execution.

Can be sent over SSH stdin without installing or writing server files. Reports
metadata/checksums, not environment variables, credentials, prompt contents or logs.
Login-node GPU visibility is NOT evidence of an available student allocation.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


MODEL = "models--Qwen--Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(argv, cwd=None):
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=30, check=False,
                                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        return {"command": argv, "returncode": result.returncode, "stdout": result.stdout[:20000],
                "stderr": result.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "unavailable": str(error)}


def source_hashes(project):
    paths = list((project/"src").rglob("*.py"))+list((project/"scripts").rglob("*.py"))
    for pattern in ("*.sh", "*.sbatch", "*.slurm"):
        paths += list((project/"scripts").rglob(pattern))+list(project.glob(pattern))
    paths += list((project/"configs").glob("*.yaml"))
    return {str(p.relative_to(project)): sha(p) for p in sorted(paths) if p.is_file()}


def cached_model(project):
    homes = [project/"hf_cache", Path.home()/".cache/huggingface"]
    if os.environ.get("HF_HOME"):
        homes.insert(0, Path(os.environ["HF_HOME"]))
    result = []
    for home in dict.fromkeys(homes):
        snapshot = home/"hub"/MODEL/"snapshots"/REVISION
        item = {"cache_home": str(home), "pinned_snapshot_exists": snapshot.is_dir(), "revision": REVISION}
        if snapshot.is_dir():
            weights = list(snapshot.glob("*.safetensors"))
            item["weight_files"] = [{"name": p.name, "resolved_exists": p.is_file(),
                                     "bytes": p.stat().st_size if p.is_file() else None} for p in sorted(weights)]
            config = snapshot/"config.json"
            if config.is_file():
                data = json.loads(config.read_text())
                item["model_config"] = {k: data.get(k) for k in ("model_type", "num_hidden_layers", "hidden_size", "torch_dtype")}
                item["model_config_sha256"] = sha(config)
            metadata = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]
            item["tokenizer_files"] = {name: sha(snapshot/name) for name in metadata if (snapshot/name).is_file()}
            config = snapshot/"tokenizer_config.json"
            if config.is_file():
                template = json.loads(config.read_text()).get("chat_template")
                if isinstance(template, str):
                    item["declared_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
            item["note"] = "Cached metadata only: verify the actual loaded tokenizer template and LoRA tensor manifest before collection."
        result.append(item)
    return result


def manifest_metadata(project):
    rows = []
    for path in sorted((project/"data").rglob("*.json"))[:128]:
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                continue
            rows.append({"path": str(path.relative_to(project)), "sha256": sha(path),
                **{k: data.get(k) for k in ("task_id", "dataset_role", "split", "dataset_revision")},
                "prompt_count": len(data.get("prompts", [])),
                "group_ids": sorted({r["group_id"] for r in data.get("prompts", []) if "group_id" in r}),
                "source_families": sorted({r["source_family"] for r in data.get("prompts", []) if "source_family" in r})})
        except (OSError, ValueError, TypeError, KeyError) as error:
            rows.append({"path": str(path.relative_to(project)), "error": str(error)})
    return rows


def inspect(project):
    project = Path(project).resolve()
    if not project.is_dir():
        raise ValueError("Server project directory not found: " + str(project))
    return {"purpose": "read_only_readiness_inventory", "project": str(project),
        "commands": [command(["hostname"]), command(["date", "-u", "+%FT%TZ"]),
            command(["nvidia-smi", "-L"]), command(["sinfo", "-o", "%N|%P|%t|%G"]),
            command(["squeue", "--me", "-o", "%i|%j|%T|%M|%l|%R|%b"]),
            command(["sacct", "-j", "25765,26946", "--format=JobID,State,Elapsed,AllocTRES,MaxRSS", "-P"]),
            command(["quota", "-s"]), command(["df", "-h", str(project)]), command(["du", "-sx", "-B1", str(project)]),
            command(["du", "-sx", "-B1", str(Path.home())]),
            command(["git", "rev-parse", "HEAD"], project), command(["git", "status", "--short"], project)],
        "cached_models": cached_model(project), "data_manifests": manifest_metadata(project),
        "dataset_caches": [{"path": str(p), "exists": p.is_dir(),
            "entries": sorted(c.name for c in p.iterdir())[:128] if p.is_dir() else []}
            for p in dict.fromkeys([project/"hf_cache/datasets", Path.home()/".cache/huggingface/datasets"]
                + ([Path(os.environ["HF_DATASETS_CACHE"])] if os.environ.get("HF_DATASETS_CACHE") else []))],
        "run_directories": sorted(p.name for p in (project/"runs").iterdir() if p.is_dir()) if (project/"runs").is_dir() else [],
        "adapter_configs": sorted(str(p.relative_to(project)) for p in project.glob("runs/**/adapter_config.json"))[:128],
        "source_hashes": source_hashes(project),
        "limitations": "No allocation claimed. nvidia-smi may be unavailable/limited on login nodes. "
                       "df free space is not quota; quota failure leaves headroom unverified. "
                       "No logs, adapters, datasets or server-only scripts were changed; no base-model files copied."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--compare", help="Compare a previously captured server report with local source hashes only")
    args = ap.parse_args(argv)
    if args.compare:
        server = json.loads(Path(args.compare).read_text())["source_hashes"]
        local = source_hashes(Path(args.project).resolve())
        report = {"different": sorted(p for p in local.keys() & server.keys() if local[p] != server[p]),
            "local_only": sorted(local.keys()-server.keys()), "server_only_preserve": sorted(server.keys()-local.keys()),
            "action": "Read-only comparison; stage reviewed code in a NEW server snapshot. Never replace server-only files or outputs."}
    else:
        report = inspect(args.project)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
