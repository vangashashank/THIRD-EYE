"""Offline calibration analysis and explicit answer rules; no model execution."""
import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml

from .feature_contract import ContractError, _UniqueSafeLoader, _integer, _mapping_keys


ANSWER_RULES = {
    "final_numeric_line_v1": r"[+-]?(?:\d+/[1-9]\d*|\d+(?:\.\d+)?|\.\d+)",
    "final_choice_line_v1": r"[A-D]",
}


def generation_prompt(tokenizer, prompt, rule):
    if rule not in ANSWER_RULES:
        raise ContractError("Unsupported explicit answer rule")
    suffix = ("an exact integer, decimal or fraction" if rule == "final_numeric_line_v1"
              else "one uppercase A, B, C or D")
    messages = [{"role": "user", "content": prompt +
                 f"\nFinish with a single line: Final answer: VALUE, where VALUE is {suffix}."}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def final_answer(text, rule, *, at_eos=False):
    """Complete Final answer: VALUE line outside <think>, newline or EOS terminated.

    Numeric rule is exact rational equality; choice rule is exactly uppercase A-D.
    No arbitrary executable verifier/regex from prompt data. Unknown task rules fail.
    """
    if rule not in ANSWER_RULES or not isinstance(text, str):
        raise ContractError("Task needs an explicit supported final-answer rule")
    # The final answer is accepted only outside model reasoning spans.
    parts = text.split("</think>")
    visible = parts[-1] if len(parts) > 1 else text
    if "<think>" in visible:
        return None
    if len(re.findall(r"(?:^|\n)Final answer:", visible)) != 1:
        return None
    ending = r"(?:\r?\n|$)" if at_eos else r"\r?\n"
    matches = list(re.finditer(r"(?:^|\n)Final answer: (" + ANSWER_RULES[rule] + r")" + ending, visible))
    if len(matches) != 1:
        return None
    return matches[0].group(1)


def verify_answer(answer, reference, rule):
    if rule not in ANSWER_RULES or not isinstance(reference, str):
        raise ContractError("Explicit task rule and string reference are required")
    if not re.fullmatch(ANSWER_RULES[rule], reference):
        raise ContractError("Reference does not satisfy the declared task answer format")
    if answer is None:
        return False
    if not re.fullmatch(ANSWER_RULES[rule], answer):
        raise ContractError("Parsed answer does not satisfy task rule")
    return Fraction(answer) == Fraction(reference) if rule == "final_numeric_line_v1" else answer == reference


@dataclass(frozen=True)
class CompletionMeasurement:
    task_id: str
    prompt_id: str
    group_id: str
    split: str
    generated_tokens: int
    correct: bool
    stopping_reason: str
    truncated: bool
    initial_cap: int


def summarize_calibration(measurements, *, minimum_correct=30, target_prompts=50):
    """Empirical nearest-rank p99 proposal, never a reliable population percentile.

    Censoring or insufficient evidence produces no cap. Thirty correct completions
    is a conservative pilot screening proposal, not a statistically reliable p99.
    All-prompt truncation, including wrong answers, is reported without filtering.
    """
    _integer(minimum_correct, where="minimum_correct", minimum=1)
    _integer(target_prompts, where="target_prompts", minimum=1)
    rows = tuple(measurements)
    if not rows:
        raise ContractError("No actual calibration measurements")
    if not isinstance(rows[0], CompletionMeasurement):
        raise ContractError("Expected recorded CompletionMeasurement")
    seen, task, cap = set(), rows[0].task_id, rows[0].initial_cap
    for row in rows:
        if not isinstance(row, CompletionMeasurement):
            raise ContractError("Expected recorded CompletionMeasurement")
        _integer(row.generated_tokens, where="generated_tokens")
        _integer(row.initial_cap, where="initial_cap", minimum=1)
        if (row.task_id != task or row.initial_cap != cap or row.split != "train"
                or any(not isinstance(value, str) or not value.strip()
                       for value in (row.task_id, row.prompt_id, row.group_id)) or row.prompt_id in seen
                or type(row.correct) is not bool or type(row.truncated) is not bool
                or row.generated_tokens > cap):
            raise ContractError("Invalid, duplicate, mixed-task/cap or held-out calibration record")
        seen.add(row.prompt_id)
        if row.stopping_reason not in ("final_answer", "eos", "max_new_tokens"):
            raise ContractError("Unknown stopping reason")
        if row.truncated != (row.stopping_reason == "max_new_tokens"):
            raise ContractError("Truncation and stopping reason disagree")
        if row.truncated and (row.generated_tokens != cap or row.correct):
            raise ContractError("Capped incomplete measurements cannot be correct completions")
        if row.correct and row.generated_tokens == 0:
            raise ContractError("Correct completion cannot have zero generated tokens")
    correct = sorted(row.generated_tokens for row in rows if row.correct and not row.truncated)
    truncated = sum(row.truncated for row in rows)
    warnings = []
    if len(rows) < target_prompts:
        warnings.append("insufficient_training_prompts")
    if len(correct) < minimum_correct:
        warnings.append("insufficient_correct_completions")
    if truncated:
        warnings.append("initial_cap_censored_measurements")
    if truncated / len(rows) > 0.02:
        warnings.append("truncation_above_2_percent")
    proposal = None
    if not warnings:
        proposal = correct[math.ceil(0.99 * len(correct))-1]
    projected = None if proposal is None else sum(row.generated_tokens > proposal for row in rows) / len(rows)
    if projected is not None and projected > 0.02:
        warnings.append("proposed_cap_all_prompt_truncation_above_2_percent")
    return {
        "task_id": task, "prompts": len(rows), "correct_completions": len(correct),
        "generated_lengths": [row.generated_tokens for row in rows],
        "initial_cap": cap, "truncated_prompts": truncated,
        "all_prompt_truncation_rate": truncated / len(rows),
        "empirical_correct_p99_cap_proposal": proposal,
        "projected_all_prompt_truncation_rate": projected,
        "warnings": warnings, "caps_status": "proposed_review_required",
        "recheck": ("Increase the initial cap on the SAME preselected training prompts in a new output directory; "
                    "repeat until uncensored, then recheck the proposed fixed cap on all prompts. "
                    "Never choose/revise caps from held-out results.") if truncated or (projected is not None and projected > .02) else None,
        "population_p99_reliable": False,
        "sample_limitation": "Approximately 50 prompts cannot establish a reliable population 99th percentile.",
    }


def load_calibration_config(path):
    path = Path(path).resolve()
    try:
        config = yaml.load(path.read_text(), Loader=_UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ContractError("Invalid safe calibration YAML") from error
    _mapping_keys(config, ("version", "model", "initial_cap", "prompts_per_task", "minimum_correct",
                          "selection_seed", "training_group_ids", "tasks"), where="calibration config")
    if config["version"] != "sprint2-cap-calibration-v1":
        raise ContractError("Unsupported calibration configuration version")
    _mapping_keys(config["model"], ("name", "revision", "quantization", "local_files_only"), where="model")
    if (not isinstance(config["model"]["name"], str) or not config["model"]["name"]
            or not isinstance(config["model"]["revision"], str) or not config["model"]["revision"]
            or config["model"]["quantization"] not in (None, "nf4")
            or config["model"]["local_files_only"] is not True):
        raise ContractError("Use an explicit cached base-model revision, never an adapter/download")
    for key in ("initial_cap", "prompts_per_task", "minimum_correct"):
        _integer(config[key], where=key, minimum=1)
    _integer(config["selection_seed"], where="selection_seed")
    if not isinstance(config["tasks"], list) or not config["tasks"]:
        raise ContractError("Unresolved task IDs/training manifests: bind tasks before calibration")
    groups = config["training_group_ids"]
    if (not isinstance(groups, list) or not groups
            or any(not isinstance(group, str) or not group.strip() for group in groups)
            or len(set(groups)) != len(groups)):
        raise ContractError("Supply explicit forecaster training_group_ids before calibration")
    tasks, seen = [], set()
    for task in config["tasks"]:
        _mapping_keys(task, ("task_id", "training_manifest", "answer_rule"), where="task")
        task_id = task["task_id"]
        if (not isinstance(task_id, str) or not task_id or task_id in seen
                or task["answer_rule"] not in ANSWER_RULES):
            raise ContractError("Unique actual task IDs and explicit supported answer rules required")
        seen.add(task_id)
        if not isinstance(task["training_manifest"], str) or not task["training_manifest"]:
            raise ContractError("Actual training manifest path is required")
        manifest_path = path.parent / task["training_manifest"]
        manifest_bytes = manifest_path.read_bytes()
        def unique_json(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ContractError(f"Duplicate manifest JSON key: {key}")
                result[key] = value
            return result
        manifest = json.loads(manifest_bytes, object_pairs_hook=unique_json)
        _mapping_keys(manifest, ("task_id", "split", "dataset_role", "prompts"), where="training manifest")
        if (manifest["task_id"] != task_id or manifest["split"] != "train"
                or manifest["dataset_role"] != "lm_training" or not isinstance(manifest["prompts"], list)):
            raise ContractError("Calibration must use explicit training prompts, not held-out/evaluation data")
        rows, ids = [], set()
        for row in manifest["prompts"]:
            _mapping_keys(row, ("prompt_id", "group_id", "source_family", "prompt", "reference"), where="prompt")
            if any(not isinstance(value, str) or not value.strip() for value in row.values()):
                raise ContractError("Training prompt identities/text/reference must be nonempty strings")
            if row["prompt_id"] in ids:
                raise ContractError("Duplicate training prompt ID")
            if row["group_id"] not in groups:
                raise ContractError("Calibration prompt belongs to a held-out/unknown lineage group")
            verify_answer(None, row["reference"], task["answer_rule"])
            ids.add(row["prompt_id"])
            rows.append(row)
        if len(rows) < config["prompts_per_task"]:
            raise ContractError("Need approximately 50 real training prompts per task, not smoke fixtures")
        def selection_key(row):
            value = [config["selection_seed"], task_id, row["prompt_id"]]
            return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
        rows = sorted(rows, key=selection_key)[:config["prompts_per_task"]]
        tasks.append({**task, "prompts": rows,
                      "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
    return config, tasks
