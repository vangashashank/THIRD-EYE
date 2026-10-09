"""Bounded, scheduler-only tiny-data pipeline verification, not an accuracy study."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from scripts.fit_direct_forecaster import fit_forecaster
from src.data.dataset import TextDataset
from src.evaluation.consequence_evaluator import evaluate_loss
from src.models.lora_model import attach_lora
from src.models.model_loader import load_model_and_tokenizer
from src.third_eye.candidate_generator import CandidateGenerator
from src.third_eye.checkpoint_manager import CheckpointManager
from src.third_eye.consequence_generator import ConsequenceGenerator
from src.third_eye.history_store import HistoryStore, METRICS
from src.third_eye.run_workspace import create_trajectory_workspace
from src.third_eye.state_manager import StateManager
from src.training.checkpoint import save_adapter, load_adapter, adapter_state_sha256
from src.training.trainer import train_lora


# Small arithmetic fixtures are deliberately not a research dataset.
TRAIN_TEXTS = ["Question: What is 2 + 3?\nAnswer: 5", "Question: What is 8 - 3?\nAnswer: 5"]
EVAL_TEXTS = {
    "target": ["Question: What is 6 + 2?\nAnswer: 8", "Question: What is 9 - 4?\nAnswer: 5"],
    "ood": ["Question: What is 3 * 4?\nAnswer: 12", "Question: What is 5 * 2?\nAnswer: 10"],
    "retention": ["Question: What is 7 + 4?\nAnswer: 11", "Question: What is 12 - 5?\nAnswer: 7"],
}


def write_json(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def probe(model, tokenizer, device):
    model.eval()
    inputs = tokenizer("Question: What is 6 + 2?\nAnswer:", return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs).logits[:, -1, :].float().cpu()


def evaluate(model, tokenizer, device, settings):
    return {key: evaluate_loss(model, TextDataset(texts, tokenizer, settings["max_length"]),
                               device, batch_size=settings["batch_size"])
            for key, texts in EVAL_TEXTS.items()}


def inspect_adapter(path, model_name, load_options, settings):
    base, tokenizer, device = load_model_and_tokenizer(model_name, **load_options)
    model = load_adapter(base, path, device)
    result = evaluate(model, tokenizer, device, settings), probe(model, tokenizer, device)
    digest = adapter_state_sha256(model)
    del model, base, tokenizer
    clear_memory()
    return *result, digest


def compare_predictions(a, b):
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    return {"max_abs_logit_difference": (a - b).abs().max().item(), "rtol": 1e-4, "atol": 1e-4,
            "argmax_equal": bool(torch.equal(a.argmax(-1), b.argmax(-1)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname() == "dgx-login01":
        raise RuntimeError("This pilot must run in a Slurm compute allocation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one allocated CUDA GPU/MIG device")
    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    torch.cuda.reset_peak_memory_stats()
    job_id = os.environ["SLURM_JOB_ID"]
    run_id = f"sprint1_qlora_{job_id}"
    paths = create_trajectory_workspace(run_id, "trajectory_000", args.run_root)
    meta = paths["metadata"]
    config_bytes = Path(args.config).read_bytes()
    config = yaml.safe_load(config_bytes)
    write_json(meta / "run_config.json", config)
    settings = config["training"]
    model_name = config["model"]["name"]
    options = {key: config["model"][key] for key in ("quantization", "revision", "local_files_only")}
    assert options["quantization"] == "nf4" and options["local_files_only"]
    source = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(source)): file_sha(path)
                     for directory in ("src", "scripts", "configs", "tests")
                     for path in sorted((source / directory).rglob("*"))
                     if path.is_file() and "__pycache__" not in path.parts}
    write_json(meta / "provenance.json", {
        "source_commit": args.source_commit, "source_sha256": source_hashes,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_revision": options["revision"], "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "gpu_capacity_bytes": torch.cuda.get_device_properties(0).total_memory,
        "job_id": job_id, "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "start_utc": start_utc, "smoke_test_only": True,
    })
    random.seed(42)
    torch.manual_seed(42)
    base, tokenizer, device = load_model_and_tokenizer(model_name, **options)
    assert base.is_loaded_in_4bit
    assert any(getattr(layer, "compute_dtype", None) == torch.bfloat16 for layer in base.modules())
    model = attach_lora(base, **config["lora"])
    model.print_trainable_parameters()
    train_lora(model, TextDataset(TRAIN_TEXTS, tokenizer, settings["max_length"]), device,
               epochs=settings["epochs"], batch_size=settings["batch_size"],
               learning_rate=settings["learning_rate"])
    parent_training = model.last_training_metrics
    original_probe = probe(model, tokenizer, device)
    original_digest = adapter_state_sha256(model)
    parent_source = paths["root"] / "parent_adapter"
    save_adapter(model, tokenizer, parent_source)
    checkpoints = CheckpointManager(paths["checkpoints"])
    parent = checkpoints.save_parent(parent_source, 0)
    parent_file_hash = file_sha(parent / "adapter_model.safetensors")
    del model, base, tokenizer
    clear_memory()
    parent_losses, parent_probe, reloaded_digest = inspect_adapter(parent, model_name, options, settings)
    assert original_digest == reloaded_digest
    reload_check = compare_predictions(original_probe, parent_probe)
    write_json(meta / "parent_metrics.json", {"record_type": "pre_update_parent_evaluation",
               "state_id": 0, "losses": parent_losses, "training": parent_training,
               "adapter_state_sha256": original_digest, "evaluation": settings})
    candidates = CandidateGenerator(checkpoints, k_candidates=3,
        metadata_file=meta / "candidate_metadata.jsonl", model_load_options=options)
    candidate_paths = candidates.generate_candidates(0, TRAIN_TEXTS, **settings,
        temporary_root=paths["candidate_work"], run_id=run_id, trajectory_id="trajectory_000")
    consequences = ConsequenceGenerator(meta / "consequence_labels.jsonl",
                                        paths["trajectory_cache"], model_load_options=options)
    labels, probes = [], []
    for i, candidate_path in enumerate(candidate_paths):
        consequence = consequences.generate_for_candidate(0, i, candidate_path, TRAIN_TEXTS,
                                                          EVAL_TEXTS["target"], **settings)
        clear_memory()
        t1, t1_probe, _ = inspect_adapter(candidate_path, model_name, options, settings)
        t2, _, _ = inspect_adapter(consequence["t2_checkpoint"], model_name, options, settings)
        assert abs(t1["target"] - consequence["t1_validation_loss"]) < 1e-4
        assert abs(t2["target"] - consequence["t2_validation_loss"]) < 1e-4
        row = {"state_id": 0, "candidate_id": i, "horizon": 2, "smoke_test_only": True,
               "parent_losses": parent_losses, "t1_losses": t1, "t2_losses": t2,
               "label_order": list(METRICS),
               "labels": [parent_losses[key] - t2[key] for key in METRICS],
               "evaluation": {"batch_size": settings["batch_size"], "max_length": settings["max_length"]},
               "evaluation_data_sha256": {key: hashlib.sha256(json.dumps(texts).encode()).hexdigest()
                                          for key, texts in EVAL_TEXTS.items()},
               "evaluation_examples_per_set": 2}
        assert all(torch.isfinite(torch.tensor(row["labels"])))
        labels.append(row)
        probes.append(t1_probe)
    with (meta / "multiconsequence_labels.jsonl").open("x") as file:
        file.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in labels))
    assert file_sha(parent / "adapter_model.safetensors") == parent_file_hash
    rollback = checkpoints.rollback_to_parent(0)
    _, rollback_probe, rollback_digest = inspect_adapter(rollback, model_name, options, settings)
    assert rollback_digest == original_digest
    rollback_check = compare_predictions(parent_probe, rollback_probe)
    # Promotion and history use actual t+1 observations, never simulated t+2 outcomes.
    selected = min(range(3), key=lambda i: labels[i]["t1_losses"]["target"])
    promoted = StateManager(checkpoints, meta / "candidate_metadata.jsonl").promote_candidate(0, selected)
    promoted_losses, promoted_probe, _ = inspect_adapter(promoted, model_name, options, settings)
    assert promoted_losses == labels[selected]["t1_losses"]
    promotion_check = compare_predictions(probes[selected], promoted_probe)
    history = HistoryStore(meta / "history.jsonl")
    assert history.features_before(0) == []
    transition = history.record_applied_transition(from_state=0, selected_candidate_id=selected,
                        before_losses=parent_losses, after_losses=promoted_losses)
    assert history.features_before(0) == []
    assert history.features_before(1) == [transition["loss_reductions"]]
    forecaster = fit_forecaster(labels_path=meta / "multiconsequence_labels.jsonl",
        proposals_path=meta / "candidate_proposals.jsonl", parent_metrics_path=meta / "parent_metrics.json",
        output_path=paths["forecasting"] / "forecaster.pth")
    torch.cuda.synchronize()
    summary = {"run_id": run_id, "job_id": job_id, "smoke_test_only": True,
        "source_commit": args.source_commit, "start_utc": start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(), "k": 3, "horizon": 2,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "parent_training": parent_training, "settings": config,
        "adapter_save_reload": reload_check, "rollback": rollback_check,
        "promotion": {**promotion_check, "selected_candidate": selected, "selection_metric": "t1_target_loss"},
        "parent_unchanged": True, "history": {"applied_transitions": 1, "simulated_t2_excluded": True},
        "forecaster": forecaster, "status": "PASSED"}
    write_json(meta / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print("SPRINT 1 NF4/BF16 PIPELINE PILOT PASSED; NOT VALIDATED FORECASTING ACCURACY")


if __name__ == "__main__":
    main()
