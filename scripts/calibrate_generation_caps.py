"""Prepare/run training-only generation calibration, never training or H2 labeling.

Default is offline validation only. --execute additionally requires an existing
Slurm CUDA allocation, explicit cached base model, and actual task manifests.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import socket
from pathlib import Path

from src.third_eye.feature_contract import ContractError
from src.third_eye.generation_calibration import (
    CompletionMeasurement, final_answer, load_calibration_config,
    summarize_calibration, verify_answer, generation_prompt,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recheck-cap", type=int,
                        help="Explicit training-only rerun cap, e.g. 1024; same prompt draw, new output")
    args = parser.parse_args(argv)
    try:
        config, tasks = load_calibration_config(args.config)
    except (ContractError, OSError, ValueError) as error:
        parser.error(str(error))
    if args.recheck_cap is not None:
        if args.recheck_cap < 1:
            parser.error("--recheck-cap must be positive")
        config["initial_cap"] = args.recheck_cap
    if not args.execute:
        print(json.dumps({"status": "validated_offline_only", "tasks": [
            {"task_id": task["task_id"], "manifest_sha256": task["manifest_sha256"],
             "prompt_ids": [row["prompt_id"] for row in task["prompts"]],
             "answer_rule": task["answer_rule"]} for task in tasks]}, indent=2))
        return
    if not args.output or not os.environ.get("SLURM_JOB_ID"):
        parser.error("--execute needs --output and an already authorized Slurm compute allocation")
    if socket.gethostname().split(".")[0] == "dgx-login01":
        parser.error("Never execute generation on the university login node")
    if Path(args.output).exists():
        parser.error("Refusing to overwrite existing calibration output")
    # Model/runtime imports occur only after offline manifest and execution gates.
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from src.models.model_loader import load_model_and_tokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Expected exactly one allocated CUDA/MIG device; no login-node generation")
    model_config = config["model"]
    model, tokenizer, device = load_model_and_tokenizer(
        model_config["name"], quantization=model_config["quantization"],
        revision=model_config["revision"], local_files_only=True,
    )
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    with (output / "measurements.jsonl").open("x") as log:
        for task in tasks:
            records = []
            rule = task["answer_rule"]
            for row in task["prompts"]:
                prompt = generation_prompt(tokenizer, row["prompt"], rule)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
                input_length = inputs["input_ids"].shape[1]
                cap = config["initial_cap"]
                if input_length + cap > model.config.max_position_embeddings:
                    raise ContractError("Training prompt plus generation cap exceeds context capacity")

                class AnswerStop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        generated = tokenizer.decode(input_ids[0, input_length:], skip_special_tokens=True)
                        done = final_answer(generated, rule) is not None
                        return torch.tensor([done], device=input_ids.device, dtype=torch.bool)

                with torch.inference_mode():
                    tokens = model.generate(**inputs, do_sample=False, num_beams=1,
                                            max_new_tokens=cap,
                                            stopping_criteria=StoppingCriteriaList([AnswerStop()]))
                generated_ids = tokens[0, input_length:]
                eos_ids = model.generation_config.eos_token_id
                eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                at_eos = len(generated_ids) > 0 and int(generated_ids[-1]) in eos_ids
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                answer = final_answer(text, rule, at_eos=at_eos)
                reason = "final_answer" if answer is not None else "eos" if at_eos else "max_new_tokens"
                measurement = CompletionMeasurement(task["task_id"], row["prompt_id"], row["group_id"],
                    "train", len(generated_ids), verify_answer(answer, row["reference"], rule),
                    reason, reason == "max_new_tokens", cap)
                records.append(measurement)
                entry = {**asdict(measurement), "completion_sha256": hashlib.sha256(text.encode()).hexdigest()}
                log.write(json.dumps(entry, allow_nan=False) + "\n")
                log.flush()
            summaries.append(summarize_calibration(records, minimum_correct=config["minimum_correct"],
                                                   target_prompts=config["prompts_per_task"]))
    report = {"version": config["version"], "model": model_config,
              "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
              "job_id": os.environ["SLURM_JOB_ID"], "generation": {"greedy": True, "thinking": False},
              "effective_initial_cap": config["initial_cap"],
              "minimum_correct": config["minimum_correct"], "target_prompts": config["prompts_per_task"],
              "recheck_cap": args.recheck_cap,
              "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              "tasks": [{"task_id": task["task_id"], "answer_rule": task["answer_rule"],
                         "manifest_sha256": task["manifest_sha256"]} for task in tasks],
              "summaries": summaries, "caps_status": "proposed_review_required"}
    with (output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
