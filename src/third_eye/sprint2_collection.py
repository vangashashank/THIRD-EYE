"""Fail-fast bindings and crash-resumable, immutable complete-state collection.

This module never loads a model by itself. Backend execution is explicit. Files
are append-only/exclusive; interrupted attempts are retained, never overwritten.
"""
import hashlib
import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml

from .feature_contract import (AppliedTransition, ContractError, TrajectoryIdentity,
    _UniqueSafeLoader, build_applied_history, load_feature_contract)
from .forecasting_handoff import (continuation_for, export_candidate, load_handoff_spec,
    make_handoff_spec, raw_module_layout, validate_handoff)
from .generation_calibration import ANSWER_RULES, verify_answer
from .sprint2_protocol import (_digest, LineageRecipe, canonical_lora_manifest, LoraModule,
    verify_lineage_splits)


def read_json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ContractError("Duplicate JSON key: " + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ContractError("Nonfinite JSON: " + v)))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def config_hash(config):
    return _digest({k: v for k, v in config.items() if k != "handoff_spec"})


def implementation_hashes():
    project = Path(__file__).resolve().parents[2]
    paths = list((project/"src").rglob("*.py")) + list((project/"scripts").glob("*.py"))
    return {str(p.relative_to(project)): file_sha(p) for p in sorted(paths)}


def load_config(path):
    config = yaml.load(Path(path).read_text(), Loader=_UniqueSafeLoader)
    if not isinstance(config, dict) or config.get("version") != "sprint2-collection-v1":
        raise ContractError("Unsupported collection config")
    return config


def prepare_spec(config, base, schema_path):
    """Seal explicit provided bindings; not a definition approval or root generator."""
    missing = binding_blockers(config)
    if missing:
        raise ContractError("Cannot seal unresolved bindings: " + "; ".join(missing))
    base = Path(base)
    contract = load_feature_contract(schema_path, dimensions={"num_lora_layers": len(config["module_manifest"]),
        "history_len": 3}, layer_order=[r["qualified_name"] for r in config["module_manifest"]])
    paths = list(config["datasets"].values()) + list(config["adaptation_manifests"].values()) + config["calibration_reports"]
    paths += [str(Path(p).parent/"measurements.jsonl") for p in config["calibration_reports"]]
    bindings = {"model": config["model"]["name"], "task": config["task"],
        "lineage": read_json(base/config["lineage_plan"]), "continuation": config["continuation"],
        "definitions": {**config["definitions"], "utility_weights": [.5, .25, .25]},
        "generation_calibration": {"reports": config["calibration_reports"], "caps": config["generation_caps"]},
        "task_rules": config["task_rules"],
        "selection": {"policy": config["selection_policy"], "seed": config["selection_seed"]},
        "collection": {"config_hash": config_hash(config), "files": {p: file_sha(base/p) for p in paths},
                       "source_hashes": implementation_hashes()}}
    return make_handoff_spec(contract, bindings, synthetic_only=False)


def validate_source_batch(attempts, corrections, plan, parent_sha, bindings):
    """Join recorded attempts to sealed sources/RNGs; never substitute references."""
    from .generation_calibration import final_answer
    from .sprint2_measurements import score_completions
    from .sprint2_protocol import prompt_processing_seed
    original, corrected = attempts["original"], attempts["corrections"]
    if [a["prompt_id"] for a in original] != list(plan.prompt_ids):
        raise ContractError("Actual H1/H2 source prompts differ from the sealed draw")
    failed, passed = [], {}
    references = {a["prompt_id"]: a["reference"] for a in original}
    for stream, rows in (("generation", original), ("correction", corrected)):
        for row in rows:
            task, prompt = row["task_id"], row["prompt_id"]
            rule = bindings["task_rules"][task]
            if (row["cap"] != bindings["generation_calibration"]["caps"][task]
                    or row["scorer"] != rule or row["parent_sha256"] != parent_sha
                    or row["seed"] != prompt_processing_seed(plan, prompt, stream=stream)
                    or row["reference"] != references[prompt]):
                raise ContractError("Source measurement scorer/cap/parent/RNG binding differs")
            result = score_completions([row], rule)["attempts"][0]
            expected_truncated = result["parsed"] is None and not row["at_eos"] and row["generated_tokens"] == row["cap"]
            if (type(row["correct"]) is not bool or row["correct"] != result["correct"]
                    or row["truncated"] != expected_truncated
                    or (result["parsed"] is None and not row["at_eos"] and row["generated_tokens"] < row["cap"])):
                raise ContractError("Source correctness/truncation is not a completed verified attempt")
            if stream == "generation" and not row["correct"]:
                failed.append(prompt)
            if stream == "correction" and row["correct"]:
                passed[prompt] = final_answer(row["text"], rule, at_eos=row["at_eos"])
    if [a["prompt_id"] for a in corrected] != failed or not passed or corrections != passed:
        raise ContractError("Training batch must contain exactly verified corrections of failed source attempts")


def validate_measured_records(ds_records, spec):
    """Recorded counts/provenance consistency, not independent proof of measurement."""
    if spec["synthetic_only"]:
        raise ContractError("Synthetic records are not real measurements")
    selections = {}
    for record in ds_records:
        p = record["provenance"]
        evidence = p["measurement_evidence"]
        if (evidence["runtime"]["restoration_verified"] is not True
                or evidence["features"]["probe"]["restoration_verified"] is not True
                or evidence["features"]["probe"]["temporary_probe_not_h1_or_h2"] is not True):
            raise ContractError("Missing verified full-update/probe restoration evidence")
        selected = evidence["selected_before_labels"]
        if type(selected) is not int or not 0 <= selected < 3:
            raise ContractError("Missing applied selection identity")
        key = (record["trajectory_id"], record["state_id"])
        choice = (selected, evidence["selection_policy"])
        if selections.setdefault(key, choice) != choice:
            raise ContractError("Siblings disagree on pre-label collection selection")
        selection = spec["bindings"]["selection"]
        if (selection["policy"] != "uniform_state_seed_v1" or choice[1] != selection["policy"]
                or selected != choose_candidate(selection["policy"], selection["seed"],
                                                record["group_id"], record["state_id"], [])):
            raise ContractError("Applied selection differs from the outcome-blind sealed policy")
        from .sprint2_protocol import build_pilot_state_plan
        binding = spec["bindings"]["continuation"]
        draws = build_pilot_state_plan(group_id=record["group_id"], state=record["state_id"],
            prompt_pool=binding["prompt_pools"][record["group_id"]], recipe=binding["recipe"],
            candidate_master_seed=binding["candidate_master_seed"], continuation_master_seed=binding["master_seed"])
        features = evidence["features"]
        plan = draws.candidates[record["candidate_id"]]
        if (features["source_plan"] != plan.sha256 or features["source_prompt_ids"] != list(plan.prompt_ids)
                or features["excluded_h1_prompt_ids"] != list(draws.excluded_h1_prompt_ids)
                or features["sibling_source_overlaps"] != json.loads(json.dumps(draws.overlaps))
                or features["definitions"] != {k: v for k, v in spec["bindings"]["definitions"].items()
                                               if k != "utility_weights"}):
            raise ContractError("Feature sources/overlaps/definitions differ from sealed candidate draw")
        validate_source_batch(features["attempts"], features["corrections"], plan,
                              p["ranking_parent_sha256"], spec["bindings"])
        if [r["prompt_id"] for r in features["serialized"]] != list(features["corrections"]):
            raise ContractError("Serialized H1 batch differs from verified corrections")
        validate_source_batch(evidence["runtime"]["continuation_attempts"], dict(p["child_batch"]["examples"]),
                              draws.continuation, p["h1"]["endpoint_sha256"], spec["bindings"])
        for scoring, accuracy in ((evidence["parent"]["scoring"], p["ranking_accuracy"]),
                                   (p["h1"]["scoring"], p["h1"]["accuracy"]),
                                   (p["h2"]["scoring"], p["h2"]["accuracy"])):
            for role in ("target", "ood", "retention"):
                row = scoring[role]
                attempts = row["attempts"]
                from .sprint2_measurements import score_completions
                recomputed = score_completions(attempts, row["scorer"])
                if (type(row["total"]) is not int or row["total"] < 1 or row["total"] != len(attempts)
                        or row["correct"] != sum(a["correct"] for a in attempts)
                        or row["parse_failures"] != sum(a["parse_failed"] for a in attempts)
                        or row["truncated"] != sum(a["truncated"] for a in attempts)
                        or row["accuracy"] != row["correct"]/row["total"]
                        or row["accuracy"] != accuracy[role]
                        or any(row[k] != recomputed[k] for k in recomputed)):
                    raise ContractError("Accuracy/count/parse/truncation provenance mismatch")
                parent_role = evidence["parent"]["scoring"][role]
                if [(a["prompt_id"], a["reference"], a["cap"]) for a in attempts] != [
                        (a["prompt_id"], a["reference"], a["cap"]) for a in parent_role["attempts"]]:
                    raise ContractError("Endpoint scoring populations/caps differ from common parent")
                if row.get("manifest_sha256") != parent_role.get("manifest_sha256"):
                    raise ContractError("Endpoint scoring source manifest differs")
        evaluation = _digest({role: evidence["parent"]["scoring"][role]["manifest_sha256"]
                              for role in ("target", "ood", "retention")})
        if evaluation != p["identity"]["evaluation_sha256"]:
            raise ContractError("Actual scoring manifests differ from pinned evaluation identity")
    for record in ds_records:
        for row in record["provenance"]["ledger"]:
            if selections[record["trajectory_id"], row["from_state"]][0] != row["candidate_id"]:
                raise ContractError("Applied ledger differs from pre-label selected H1 identity")


def binding_blockers(config):
    required = ("task", "corpus_version", "handoff_spec", "module_manifest", "lineage_plan",
        "roots", "datasets", "adaptation_manifests", "task_rules", "generation_caps",
        "calibration_reports", "chat_template_sha256", "definitions", "candidate_recipes",
        "continuation", "selection_policy", "update", "probe", "retention")
    missing = ["unbound " + k for k in required if not config.get(k)]
    if type(config.get("selection_seed")) is not int:
        missing.append("unbound selection_seed")
    for decision in ("D06", "D07", "D08", "D09", "D12", "D14"):
        reference = config.get("definition_agreements", {}).get(decision)
        if not isinstance(reference, str) or not reference.strip():
            missing.append("awaiting explicit " + decision + " agreement")
    if not config.get("definition_agreements", {}).get("D12_measurements"):
        missing.append("awaiting explicit D12 probe-reduction agreement (budget is approved separately)")
    for key in ("mig_profile", "device_total_memory_bytes"):
        if not config.get("retention", {}).get(key):
            missing.append("unbound actual retention." + key)
    for role in ("target", "ood", "retention", "confidence", "anchor_probe", "val_probe", "retention_gradient", "final"):
        if not config.get("datasets", {}).get(role):
            missing.append("unbound datasets." + role)
    for key in ("prompt_pools", "recipe"):
        if not config.get("continuation", {}).get(key):
            missing.append("unbound continuation." + key)
    for key in ("confidence", "difficulty", "embedding", "duplicate", "gradient", "probe", "correction",
                "answer_span", "nll_reduction", "cosine_zero_epsilon", "duplicate_threshold", "duplicate_tolerance"):
        if key not in config.get("definitions", {}):
            missing.append("unbound definitions." + key)
    if config.get("caps_frozen_before_collection") is not True:
        missing.append("unreviewed caps_frozen_before_collection (caps only, not schema freeze)")
    if not config.get("caps_review_reference"):
        missing.append("unbound caps_review_reference")
    return missing


def inspect_bindings(config, base, *, prepared_spec=None):
    """Offline audit. No data/model download, no source-family independence invented."""
    missing = binding_blockers(config)
    if missing:
        raise ContractError("Collection blocked:\n- " + "\n- ".join(missing))
    base = Path(base)
    if prepared_spec is None:
        spec, contract, layout = load_handoff_spec(base/config["handoff_spec"])
    else:
        spec = prepared_spec
        contract = load_feature_contract(Path(__file__).resolve().parents[2]/"configs/feature_schema.yaml",
            dimensions=spec["dimensions"], layer_order=spec["layer_order"])
        if spec != make_handoff_spec(contract, spec["bindings"], synthetic_only=False):
            raise ContractError("Prepared spec hash mismatch")
        layout = raw_module_layout(contract)
    if spec["synthetic_only"]:
        raise ContractError("Real collector refuses synthetic specs")
    if config["model"]["local_files_only"] is not True:
        raise ContractError("Cached pinned model only")
    manifest = config["module_manifest"]
    count, order = canonical_lora_manifest([LoraModule(r["block_index"], r["projection"], r["qualified_name"])
        for r in manifest], placement_order=config["placement_order"])
    if (tuple(r["qualified_name"] for r in manifest) != order or contract.layer_order != order
            or contract.dimensions["num_lora_layers"] != count):
        raise ContractError("Explicit raw-module order/count mismatch")
    lineage = read_json(base/config["lineage_plan"])
    if lineage != spec["bindings"]["lineage"]:
        raise ContractError("Preassigned lineage plan differs from handoff")
    recipes = [LineageRecipe(r["trajectory_id"], r["adapted_root_id"], tuple(r["prefix_ids"]),
                            tuple(r["source_families"])) for r in lineage["recipes"]]
    verify_lineage_splits(recipes, lineage["membership"], lineage["splits"])
    if [list(lineage["splits"].values()).count(s) for s in ("train", "validation", "test")] != [8, 2, 2]:
        raise ContractError("Preserve accepted 12-group 8/2/2 plan")
    if set(config["roots"]) != set(lineage["membership"]):
        raise ContractError("Supply all planned roots, not invented independent replicas")
    roles = {"target", "ood", "retention", "confidence", "anchor_probe", "val_probe", "retention_gradient", "final"}
    if set(config["datasets"]) != roles or set(config["adaptation_manifests"]) != set(lineage["splits"]):
        raise ContractError("Bind all LM roles and each group's adaptation source pool")
    files, populations, tasks = {}, {}, set()
    for role, path in {**config["datasets"], **{"adaptation:"+g: p for g, p in config["adaptation_manifests"].items()}}.items():
        source = base/path
        data = read_json(source)
        expected_role = "lm_training" if role.startswith("adaptation:") else role
        if data.get("dataset_role") != expected_role or not data.get("prompts"):
            raise ContractError("Empty/wrong-role actual manifest: " + role)
        task = data["task_id"]
        tasks.add(task)
        if task not in config["task_rules"] or config["task_rules"][task] not in ANSWER_RULES:
            raise ContractError("Actual task requires a supported deterministic scorer")
        ids = set()
        families = set()
        for row in data["prompts"]:
            if any(not isinstance(row.get(k), str) or not row[k].strip()
                   for k in ("prompt_id", "source_family", "prompt", "reference")) or row["prompt_id"] in ids:
                raise ContractError("Invalid/duplicate actual prompt identities")
            verify_answer(None, row["reference"], config["task_rules"][task])
            ids.add(row["prompt_id"])
            families.add(row["source_family"])
            if role.startswith("adaptation:"):
                group = role.split(":", 1)[1]
                allowed = {f for r in recipes if lineage["membership"][r.trajectory_id] == group for f in r.source_families}
                if row.get("group_id") != group or row["source_family"] not in allowed:
                    raise ContractError("Adaptation source differs from preassigned lineage families")
                split = lineage["splits"][group]
                if data.get("split") != ("val" if split == "validation" else split):
                    raise ContractError("Adaptation forecaster split differs from plan")
        populations[role] = {"data": data, "ids": ids, "families": families, "manifest_sha256": file_sha(source)}
        files[path] = file_sha(source)
    if len(populations["retention"]["ids"]) != 256:
        raise ContractError("D03 requires the fixed 256-example retention anchor")
    approved_counts = {"target": 32, "ood": 32, "retention": 256,
                       "anchor_probe": 32, "retention_gradient": 32, "val_probe": 32}
    if config["diagnostic_counts"] != approved_counts:
        raise ContractError("Pilot diagnostic counts must remain the approved 32/32/256/subset32")
    for role, count in approved_counts.items():
        if len(populations[role]["ids"]) != count:
            raise ContractError("Pilot diagnostic subset count differs: " + role)
    for subset, parent in (("anchor_probe", "retention"), ("retention_gradient", "retention"), ("val_probe", "target")):
        outer = {r["prompt_id"]: r for r in populations[parent]["data"]["prompts"]}
        if (populations[subset]["data"]["task_id"] != populations[parent]["data"]["task_id"]
                or any(outer.get(r["prompt_id"]) != r for r in populations[subset]["data"]["prompts"])):
            raise ContractError("Probe/gradient subset must contain the exact pinned diagnostic examples")
    evaluation = _digest({role: populations[role]["manifest_sha256"] for role in ("target", "ood", "retention")})
    if any(v["evaluation_sha256"] != evaluation or v["corpus_version"] != config["corpus_version"]
           for v in lineage["identities"].values()):
        raise ContractError("Pinned corpus/evaluation identity differs from actual diagnostic manifests")
    allowed_pairs = {frozenset(p) for p in config["allowed_diagnostic_overlaps"]}
    for a, pop in populations.items():
        for b, other in populations.items():
            if a >= b:
                continue
            overlap = bool(pop["families"] & other["families"] or pop["ids"] & other["ids"])
            if overlap and (a == "final" or b == "final" or a.startswith("adaptation:")
                            or b.startswith("adaptation:") or frozenset((a, b)) not in allowed_pairs):
                raise ContractError("Undeclared/cross-role source-family leakage: %s/%s" % (a, b))
    for path in config["calibration_reports"]:
        report = read_json(base/path)
        if report["model"] != config["model"] or report.get("generation") != {"greedy": True, "thinking": False}:
            raise ContractError("Calibration base revision/generation mismatch")
        files[path] = file_sha(base/path)
        from dataclasses import fields
        from .generation_calibration import CompletionMeasurement, summarize_calibration
        measurement_path = (base/path).parent/"measurements.jsonl"
        names = [f.name for f in fields(CompletionMeasurement)]
        rows = [json.loads(line) for line in measurement_path.read_text().splitlines() if line.strip()]
        train_groups = {g for g, split in lineage["splits"].items() if split == "train"}
        source_ids = {(role.split(":", 1)[1], r["prompt_id"], pop["data"]["task_id"]) for role, pop in populations.items()
                      if role.startswith("adaptation:") and role.split(":", 1)[1] in train_groups
                      for r in pop["data"]["prompts"]}
        if any(r["group_id"] not in train_groups or (r["group_id"], r["prompt_id"], r["task_id"]) not in source_ids for r in rows):
            raise ContractError("Calibration used held-out/unknown source prompts")
        files[str(Path(path).parent/"measurements.jsonl")] = file_sha(measurement_path)
        if any(t["answer_rule"] != config["task_rules"].get(t["task_id"]) for t in report["tasks"]):
            raise ContractError("Calibration scorer binding differs from collection")
        for summary in report["summaries"]:
            task = summary["task_id"]
            actual = [CompletionMeasurement(**{k: r[k] for k in names}) for r in rows if r["task_id"] == task]
            recalculated = summarize_calibration(actual, minimum_correct=report.get("minimum_correct", 30),
                target_prompts=report.get("target_prompts", 50))
            if summary != recalculated:
                raise ContractError("Calibration summary differs from actual recorded measurements")
            if (summary["empirical_correct_p99_cap_proposal"] is None or summary["warnings"]
                    or config["generation_caps"].get(task) != summary["empirical_correct_p99_cap_proposal"]):
                raise ContractError("Measured caps need uncensored evidence and a reviewed fixed proposal")
        if report.get("chat_template_sha256") != config["chat_template_sha256"]:
            raise ContractError("Calibration chat-template hash mismatch")
    covered = {s["task_id"] for path in config["calibration_reports"] for s in read_json(base/path)["summaries"]}
    if set(config["generation_caps"]) != tasks or covered != tasks:
        raise ContractError("One fixed measured generation cap per actual task required")
    if config.get("caps_frozen_before_collection") is not True or not config.get("caps_review_reference"):
        raise ContractError("Caps must be reviewed and fixed before collection; this is not a feature-schema freeze")
    if config["selection_policy"] != "uniform_state_seed_v1":
        raise ContractError("Approved pilot selects uniformly before all features/outcomes, not using NLL or diagnostics")
    if len(config["candidate_recipes"]) != 3 or config["continuation"] != spec["bindings"]["continuation"]:
        raise ContractError("K=3 and shared continuation bindings required")
    if config["candidate_recipes"] != [{"candidate_id": i, "prompt_count": 32} for i in range(3)]:
        raise ContractError("Pilot requires three candidate-keyed 32-prompt draws, same procedure/count")
    continuation = config["continuation"]
    if continuation.get("exclude_h1") is not True or continuation["prompt_count"] != 32:
        raise ContractError("Pilot H2 requires one 32-prompt draw excluding the union of H1 draws")
    if set(continuation["prompt_pools"]) != set(lineage["splits"]):
        raise ContractError("Bind a continuation source pool for every planned group")
    for group, pool in continuation["prompt_pools"].items():
        source_ids = populations["adaptation:"+group]["ids"]
        if (not isinstance(pool, list) or len(pool) != len(set(pool)) or set(pool) != source_ids
                or any(r["prompt_count"] > len(source_ids) for r in config["candidate_recipes"])):
            raise ContractError("Candidate/continuation draw exceeds actual assigned source pool")
        for state in range(5):
            continuation_for(spec, group, state)  # Validate all sealed draws before any outcome.
    update, probe = config["update"], config["probe"]
    if set(update) != {"steps", "learning_rate", "weight_decay", "max_length", "batch_size",
                       "gradient_accumulation", "cycling", "scheduler", "betas", "eps"}:
        raise ContractError("Bind the supported fresh-AdamW update recipe exactly")
    if (update["steps"] != 32 or update["batch_size"] != 1 or update["gradient_accumulation"] != 1
            or update["cycling"] != "sealed_prompt_order_v1" or update["scheduler"] != "none"):
        raise ContractError("Pilot uses 32 optimizer updates, batch 1/accumulation 1, sealed-order cycling and no scheduler")
    if (any(type(update[k]) is not int or update[k] < 1 for k in ("steps", "max_length"))
            or any(type(update[k]) not in (int, float) or not math.isfinite(update[k]) or update[k] < 0
                   for k in ("learning_rate", "weight_decay")) or update["learning_rate"] == 0):
        raise ContractError("Invalid full-update settings")
    if (set(probe) != {"steps", "cadence", "optimizer", "scheduler"} or type(probe["steps"]) is not int
            or probe["steps"] < 2 or not isinstance(probe["cadence"], list)
            or len(probe["cadence"]) < 2 or probe["cadence"][0] != 0
            or probe["cadence"][-1] != probe["steps"] or probe["cadence"] != sorted(set(probe["cadence"]))
            or any(type(v) is not int or not 0 <= v <= probe["steps"] for v in probe["cadence"])
            or set(probe["optimizer"]) != {"lr", "weight_decay", "betas", "eps"}
            or any(type(probe["optimizer"][k]) not in (int, float) or not math.isfinite(probe["optimizer"][k])
                   or probe["optimizer"][k] < 0 for k in ("lr", "weight_decay", "eps"))
            or probe["optimizer"]["lr"] == 0):
        raise ContractError("Invalid explicit probe recipe/cadence")
    if probe["steps"] != 15 or probe["cadence"] != [0, 5, 10, 15] or probe["scheduler"] != "none":
        raise ContractError("Pilot probe is 15 optimizer updates with fixed 0/5/10/15 diagnostics")
    expected_optimizer = {"lr": update["learning_rate"], "weight_decay": update["weight_decay"],
                          "betas": update["betas"], "eps": update["eps"]}
    if (probe["optimizer"] != expected_optimizer or update["learning_rate"] != .0001
            or update["weight_decay"] != .01 or update["betas"] != [.9, .999] or update["eps"] != 1e-8):
        raise ContractError("Pilot reuses pinned supported AdamW hyperparameters")
    expected_continuation_recipe = {"update": update, "generation": {"greedy": True, "thinking": False},
        "correction": config["definitions"]["correction"], "empty_failure_batch": "stop_v1"}
    if continuation["recipe"] != expected_continuation_recipe:
        raise ContractError("Sealed continuation recipe must match the actually executed update/generation/correction")
    if config["retention"].get("policy") != "retain_all_v1":
        raise ContractError("No automatic artifact pruning; bind approved retain_all_v1")
    for key in ("max_device_seconds", "max_output_bytes"):
        if type(config["retention"].get(key)) is not int or config["retention"][key] < 1:
            raise ContractError("Positive explicit D14 resource caps required")
    if (config["retention"]["max_device_seconds"] > 7200
            or config["retention"]["max_output_bytes"] > 5*1024**3
            or not isinstance(config["retention"]["mig_profile"], str)
            or type(config["retention"]["device_total_memory_bytes"]) is not int
            or config["retention"]["device_total_memory_bytes"] < 1):
        raise ContractError("Pilot ceilings cannot expand without review; bind measured MIG profile/memory")
    definitions = config["definitions"]
    expected = {"confidence": "answer_mean_logprob_v1", "difficulty": "original_failure_fraction_v1",
        "embedding": "frozen_base_mean_nonspecial_v1", "duplicate": "nfc_word_trigram_components_v1",
        "gradient": "eval_unscaled_lora_ab_v1", "probe": "ols_cadence_parent_post_answer_kl_v2",
        "correction": "parent_self_correction_v1", "answer_span": "final_value_boundary_whitespace_v1"}
    if any(definitions.get(k) != v for k, v in expected.items()):
        raise ContractError("Unsupported/unbound feature definitions; these versions remain proposals until agreement")
    if definitions.get("nll_reduction") not in ("equal_answer_v1", "token_weighted_v1"):
        raise ContractError("Pin the NLL reduction, no silent choice")
    if (type(definitions.get("cosine_zero_epsilon")) not in (int, float)
            or not math.isfinite(definitions["cosine_zero_epsilon"]) or definitions["cosine_zero_epsilon"] < 0
            or type(definitions.get("duplicate_threshold")) not in (int, float)
            or not 0 < definitions["duplicate_threshold"] <= 1
            or type(definitions.get("duplicate_tolerance")) not in (int, float)
            or not 0 <= definitions["duplicate_tolerance"] < definitions["duplicate_threshold"]):
        raise ContractError("Bind finite explicit cosine/duplicate thresholds")
    actual_roots = [LineageRecipe(r.trajectory_id, r.adapted_root_id,
        r.prefix_ids + (lineage["identities"][r.trajectory_id]["root_sha256"],), r.source_families) for r in recipes]
    verify_lineage_splits(actual_roots, lineage["membership"], lineage["splits"])
    for trajectory, root in config["roots"].items():
        path = base/root["path"]
        for name, sha in root["files"].items():
            if Path(name).name != name or file_sha(path/name) != sha:
                raise ContractError("Root artifact checksum/path mismatch")
        if "adapter_model.safetensors" not in root["files"] or "adapter_config.json" not in root["files"]:
            raise ContractError("Actual cached root adapter and config checksums required")
        adapter = read_json(path/"adapter_config.json")
        expected_adapter = {"r": config["lora"]["r"], "lora_alpha": config["lora"]["alpha"],
                            "lora_dropout": config["lora"]["dropout"]}
        if (any(adapter.get(k) != v for k, v in expected_adapter.items())
                or set(adapter.get("target_modules", [])) != set(config["lora"]["target_modules"])
                or adapter.get("base_model_name_or_path") != config["model"]["name"]
                or adapter.get("bias") != "none"):
            raise ContractError("Root LoRA recipe differs from pinned collection configuration")
    if spec["bindings"].get("collection") != {"config_hash": config_hash(config), "files": files,
                                             "source_hashes": implementation_hashes()}:
        raise ContractError("Bind immutable collection configuration and source-file hashes in the handoff spec")
    if spec["bindings"]["model"] != config["model"]["name"] or spec["bindings"]["task"] != config["task"]:
        raise ContractError("Model/task record binding differs from collection")
    if spec["bindings"]["definitions"] != {**config["definitions"], "utility_weights": [.5, .25, .25]}:
        raise ContractError("Definition hash configuration differs from collection")
    if (spec["bindings"].get("task_rules") != config["task_rules"]
            or spec["bindings"].get("selection") != {"policy": config["selection_policy"], "seed": config["selection_seed"]}):
        raise ContractError("Scorer/selection definitions differ from collection")
    return spec, contract, layout, populations


def choose_candidate(policy, seed, group, state, inputs):
    """Only pre-selection inputs accepted; no outcome/label parameter."""
    if policy == "uniform_state_seed_v1":
        return random.Random(int(_digest([policy, seed, group, state]), 16)).randrange(3)
    if policy == "highest_pre_update_nll_v1":
        scores = [float(row["candidate_data_stats"]["pre_update_nll"][0]) for row in inputs]
        tied = [i for i, score in enumerate(scores) if score == max(scores)]
        return random.Random(int(_digest([policy, seed, group, state]), 16)).choice(tied)
    raise ContractError("Unbound collection selection policy")


class RunStore:
    def __init__(self, path, config, spec, *, resume=False):
        self.path, self.spec = Path(path), spec
        if resume:
            saved = read_json(self.path/"sealed.json")
            if saved != {"config": config, "spec": spec}:
                raise ContractError("Resume config/definition/source hash mismatch")
        else:
            self.path.mkdir(parents=True, exist_ok=False)
            write_json(self.path/"sealed.json", {"config": config, "spec": spec})
            write_json(self.path/"lineage_plan.json", spec["bindings"]["lineage"])
        self.lock = self.path/"collector.lock"

    def event(self, kind, **values):
        with (self.path/"events.jsonl").open("a") as f:
            f.write(json.dumps({"event": kind, "time_ns": time.time_ns(), **values}, allow_nan=False)+"\n")
            f.flush()
            os.fsync(f.fileno())

    def completed(self):
        bundles = []
        for path in sorted(self.path.glob("commits/*.json")):
            bundle = read_json(path)
            if bundle["semantic_hash"] != self.spec["semantic_hash"]:
                raise ContractError("Committed state definition/config hash differs")
            for name, sha in bundle["checksums"].items():
                source = (self.path/name).resolve()
                if self.path.resolve() not in source.parents or file_sha(source) != sha:
                    raise ContractError("Committed artifact checksum/path mismatch")
            bundles.append(bundle)
        return sorted(bundles, key=lambda b: (b["trajectory_id"], b["state"]))

    def spent_seconds(self):
        path = self.path/"events.jsonl"
        if not path.exists():
            return 0.0
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        starts = {r["session_id"] for r in rows if r["event"] == "collection_started"}
        finished = {r["session_id"]: r["elapsed_seconds"] for r in rows if r["event"] == "collection_finished"}
        if starts != set(finished):
            raise ContractError("Unclosed execution session: review elapsed allocation time before resume; never reset its budget")
        values = list(finished.values())
        if any(type(v) not in (int, float) or v < 0 or not math.isfinite(v) for v in values):
            raise ContractError("Invalid recorded execution duration")
        return sum(values)

    def collect(self, config, contract, backend, trajectories):
        fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        started, session = getattr(backend, "execution_started", time.monotonic()), uuid.uuid4().hex
        logged = False
        try:
            os.write(fd, str(os.getpid()).encode())
            old_seconds = self.spent_seconds()
            self.event("collection_started", trajectories=trajectories, session_id=session)
            logged = True
            if getattr(backend, "deadline", None) is not None:
                backend.deadline = min(backend.deadline, started+config["retention"]["max_device_seconds"]-old_seconds-120)
            return self._collect(config, contract, backend, trajectories, started, old_seconds)
        except BaseException as error:
            self.event("collection_stopped", exception=type(error).__name__, detail=str(error))
            raise
        finally:
            try:
                if logged:
                    self.event("collection_finished", session_id=session, elapsed_seconds=time.monotonic()-started)
            finally:
                os.close(fd)
                self.lock.unlink()  # Only our process lock; no artifact deletion.

    def _collect(self, config, contract, backend, trajectories, started, old_seconds):
        spec, all_bundles = self.spec, self.completed()
        if all_bundles:
            existing = [r for b in all_bundles for r in read_json(self.path/b["records_path"])]
            validate_handoff(existing, spec, contract, raw_module_layout(contract), manifest=restricted_manifest(spec, existing))
        for trajectory in trajectories:
            identity = TrajectoryIdentity(**spec["bindings"]["lineage"]["identities"][trajectory])
            if spec["bindings"]["lineage"]["splits"][identity.group_id] != "train":
                raise ContractError("Initial 15/90-row collection is train-group only")
            complete = [b for b in all_bundles if b["trajectory_id"] == trajectory]
            if [b["state"] for b in complete] != list(range(len(complete))):
                raise ContractError("Resume has a missing applied state")
            ledger, parent_sha, parent_accuracy = [], identity.root_sha256, None
            for b in complete:
                for raw in b["applied"]:
                    row = dict(raw)
                    row["identity"] = TrajectoryIdentity(**row["identity"])
                    ledger.append(AppliedTransition(**row))
                parent_sha, parent_accuracy = b["selected_h1"]["endpoint_sha256"], b["selected_h1"]["accuracy"]
            backend.load_parent(trajectory, self.path/complete[-1]["selected_snapshot"] if complete else None)
            if backend.parent_sha() != parent_sha:
                raise ContractError("Reloaded actual parent does not match pinned/applied hash")
            for state in range(len(complete), 5):
                state_started = time.monotonic()
                bytes_used = sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())
                if (old_seconds + time.monotonic()-started >= config["retention"]["max_device_seconds"]
                        or bytes_used >= config["retention"]["max_output_bytes"]):
                    raise ContractError("D14 resource cap reached before next state; no artifacts removed")
                attempt = self.path/"attempts"/uuid.uuid4().hex
                attempt.mkdir(parents=True, exist_ok=False)
                if hasattr(backend, "state_output_path"):
                    backend.state_output_path = attempt
                plan = continuation_for(spec, identity.group_id, state)
                selected = (choose_candidate(config["selection_policy"], config["selection_seed"],
                    identity.group_id, state, []) if config["selection_policy"] == "uniform_state_seed_v1" else None)
                draws = None
                if spec["bindings"]["continuation"].get("exclude_h1") is True:
                    from .sprint2_protocol import build_pilot_state_plan
                    binding = spec["bindings"]["continuation"]
                    draws = asdict(build_pilot_state_plan(group_id=identity.group_id, state=state,
                        prompt_pool=binding["prompt_pools"][identity.group_id], recipe=binding["recipe"],
                        candidate_master_seed=binding["candidate_master_seed"], continuation_master_seed=binding["master_seed"]))
                # Persist outcome-blind plans before generating any candidate results.
                tick = state*100 + spec["bindings"]["lineage"]["assigned_at"] + 10
                write_json(attempt/"plan.json", {"plan": asdict(plan), "sealed_at": tick,
                    "candidate_recipes": config["candidate_recipes"], "selection_policy": config["selection_policy"],
                    "selected_before_features_and_labels": selected, "source_draws": draws})
                self.event("state_plan_sealed", trajectory=trajectory, state=state, plan_sha256=plan.sha256)
                ranking, state_features, state_evidence = backend.measure_parent()
                if parent_accuracy is not None and ranking != parent_accuracy:
                    raise ContractError("Re-scored applied parent differs from recorded H1")
                history, context = build_applied_history(ledger, identity=identity, ranking_state=state,
                    ranking_parent_sha256=parent_sha, cutoff=tick+1)
                inputs, evidence, batches, diagnostics = [], [], [], []
                for candidate in range(3):
                    features, proof, batch, zero = backend.prepare_candidate(identity.group_id, state, candidate)
                    row = {"current_state_performance": state_features, **features, "recursive_history": history}
                    contract.validate_inputs(row, layer_order=contract.layer_order, history_context=context)
                    if backend.parent_sha() != parent_sha:
                        raise ContractError("Feature/probe mutated the common parent")
                    inputs.append(row)
                    evidence.append(proof)
                    batches.append(batch)
                    diagnostics.append(zero)
                if selected is None:
                    selected = choose_candidate(config["selection_policy"], config["selection_seed"], identity.group_id, state, inputs)
                self.event("selected_before_full_labels", trajectory=trajectory, state=state, candidate=selected)
                write_json(attempt/"ranking.json", {"parent": state_evidence, "candidates": evidence,
                    "selected_before_labels": selected, "cutoff": tick+1})
                records, selected_h1, selected_snapshot = [], None, None
                for candidate in range(3):
                    h1, h2, corrected, snapshot, runtime = backend.run_sibling(
                        batches[candidate], plan, candidate, attempt/str(candidate))
                    h1.update(endpoint_id=f"{trajectory}/s{state}/c{candidate}/h1", observed_at=tick+2)
                    h2.update(endpoint_id=f"{trajectory}/s{state}/c{candidate}/h2", observed_at=tick+3)
                    record = export_candidate(contract, spec, inputs=inputs[candidate], history_context=context,
                        model=spec["bindings"]["model"], task=spec["bindings"]["task"], candidate_id=candidate,
                        ranking_accuracy=ranking, h1=h1, h2=h2, corrected_failures=corrected,
                        plan_sealed_at=tick, zero_gradient_diagnostics=diagnostics[candidate])
                    record["provenance"]["measurement_evidence"] = {"parent": state_evidence,
                        "features": evidence[candidate], "runtime": runtime,
                        "selection_policy": config["selection_policy"], "selected_before_labels": selected}
                    records.append(record)
                    if backend.parent_sha() != parent_sha:
                        raise ContractError("Sibling did not restore the common parent")
                    if candidate == selected:
                        selected_h1, selected_snapshot = h1, str(snapshot.relative_to(self.path))
                transition = AppliedTransition(identity, f"{trajectory}/applied/{state}", state, state+1,
                    selected, parent_sha, selected_h1["endpoint_sha256"], ranking, selected_h1["accuracy"],
                    state+1, selected_h1["observed_at"])
                write_json(attempt/"records.json", records)
                prior = [r for b in all_bundles for r in read_json(self.path/b["records_path"])]
                manifest = restricted_manifest(spec, prior+records)
                validate_handoff(prior+records, spec, contract, raw_module_layout(contract), manifest=manifest)
                if not spec["synthetic_only"]:
                    validate_measured_records(prior+records, spec)
                checksums = {str(p.relative_to(self.path)): file_sha(p) for p in attempt.rglob("*") if p.is_file()}
                bundle = {"trajectory_id": trajectory, "state": state, "applied": [asdict(transition)],
                    "selected_h1": selected_h1, "selected_snapshot": selected_snapshot,
                    "records_path": str((attempt/"records.json").relative_to(self.path)), "checksums": checksums,
                    "elapsed_seconds": time.monotonic()-state_started, "semantic_hash": spec["semantic_hash"]}
                commits = self.path/"commits"
                commits.mkdir(exist_ok=True)
                write_json(commits/(_digest([trajectory, state])+".json"), bundle)
                self.event("state_committed", trajectory=trajectory, state=state, selected=selected)
                all_bundles.append(bundle)
                ledger.append(transition)
                parent_sha, parent_accuracy = selected_h1["endpoint_sha256"], selected_h1["accuracy"]
                backend.load_parent(trajectory, self.path/selected_snapshot)
                if backend.parent_sha() != parent_sha:
                    raise ContractError("Selected applied H1 reload mismatch")
                current_bytes = sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())
                if (old_seconds + time.monotonic()-started > config["retention"]["max_device_seconds"]
                        or current_bytes > config["retention"]["max_output_bytes"]):
                    raise ContractError("Resource cap exceeded within completed state; stopped, outputs retained")
        return self.export(contract)

    def export(self, contract):
        records = [r for b in self.completed() for r in read_json(self.path/b["records_path"])]
        manifest = restricted_manifest(self.spec, records)
        validate_handoff(records, self.spec, contract, raw_module_layout(contract), manifest=manifest)
        if not self.spec["synthetic_only"]:
            validate_measured_records(records, self.spec)
        out = self.path/"exports"/uuid.uuid4().hex
        out.mkdir(parents=True, exist_ok=False)
        with (out/"meta.jsonl").open("x") as f:
            for row in records:
                f.write(json.dumps(row, allow_nan=False)+"\n")
        write_json(out/"manifest.json", manifest)
        write_json(out/"spec.json", self.spec)
        write_json(out/"checksums.json", {p.name: file_sha(p) for p in out.iterdir() if p.is_file()})
        return out


def restricted_manifest(spec, records):
    splits = spec["bindings"]["lineage"]["splits"]
    return {g: "val" if splits[g] == "validation" else splits[g] for g in sorted({r["group_id"] for r in records})}
