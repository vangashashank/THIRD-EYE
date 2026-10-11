"""Explicitly SYNTHETIC journal/backend fixtures, never real records or accuracy."""
import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.build_synthetic_handoff import make_fixture
from scripts.collect_sprint2 import main
from src.third_eye.feature_contract import ContractError, load_feature_contract
from src.third_eye.forecasting_handoff import _source_inputs, raw_module_layout, validate_handoff
from src.third_eye.sprint2_collection import (RunStore, binding_blockers, choose_candidate, config_hash,
    file_sha, load_config, read_json, restricted_manifest, write_json)
from src.third_eye.sprint2_collection import inspect_bindings, prepare_spec
from src.third_eye.generation_calibration import CompletionMeasurement, summarize_calibration
from src.third_eye.sprint2_protocol import _digest


ROOT = Path(__file__).resolve().parents[1]


class SyntheticJournalBackend:
    """Copies fabricated handoff fixtures; NO model, features or labels measured."""
    def __init__(self, rows, contract, *, failure_state=None):
        self.rows = {(r["state_id"], r["candidate_id"]): r for r in rows}
        self.contract, self.failure_state = contract, failure_state
        self.state, self.parent = 0, rows[0]["provenance"]["ranking_parent_sha256"]
    def load_parent(self, trajectory, snapshot):
        self.state = 0 if snapshot is None else read_json(snapshot)["next_state"]
        self.parent = (self.rows[0, 0]["provenance"]["ranking_parent_sha256"] if snapshot is None
                       else read_json(snapshot)["parent_sha256"])
    def parent_sha(self):
        return self.parent
    def measure_parent(self):
        r = self.rows[self.state, 0]
        inputs = _source_inputs(r, self.contract)
        return r["provenance"]["ranking_accuracy"], inputs["current_state_performance"], {"synthetic_only": True}
    def prepare_candidate(self, group, state, candidate):
        if state == self.failure_state:
            raise RuntimeError("synthetic injected interruption, not a GPU job")
        inputs = _source_inputs(self.rows[state, candidate], self.contract)
        inputs["candidate_data_stats"]["pre_update_nll"][0] = 4 if candidate == 0 else 1
        features = {k: v for k, v in inputs.items() if k not in ("recursive_history", "current_state_performance")}
        return features, {"synthetic_only": True}, state, self.rows[state, candidate]["provenance"]["zero_gradient_diagnostics"]
    def run_sibling(self, batch, plan, candidate, out):
        out.mkdir()
        p = self.rows[batch, candidate]["provenance"]
        snapshot = out/"SYNTHETIC_NOT_WEIGHTS.json"
        write_json(snapshot, {"next_state": batch+1, "parent_sha256": p["h1"]["endpoint_sha256"]})
        return copy.deepcopy(p["h1"]), copy.deepcopy(p["h2"]), dict(p["child_batch"]["examples"]), snapshot, {"synthetic_only": True}


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.spec, rows, _ = make_fixture(states=5)
        self.contract = load_feature_contract(ROOT/"configs/feature_schema.yaml", dimensions=self.spec["dimensions"],
                                             layer_order=self.spec["layer_order"])
        group = next(g for g, s in self.spec["bindings"]["lineage"]["splits"].items() if s == "train"
                     and any(r["group_id"] == g for r in rows))
        self.rows = [r for r in rows if r["group_id"] == group]
        self.trajectory = self.rows[0]["trajectory_id"]
        self.config = {"selection_policy": "highest_pre_update_nll_v1", "selection_seed": 1,
            "candidate_recipes": [{"synthetic_only": True}]*3,
            "retention": {"max_device_seconds": 1000, "max_output_bytes": 100000000}}

    def test_offline_unbound_preflight_does_not_load_model(self):
        config = load_config(ROOT/"configs/sprint2_collection.yaml")
        missing = binding_blockers(config)
        self.assertIn("awaiting explicit D06 agreement", missing)
        self.assertIn("unbound task", missing)
        self.assertEqual(main(["--config", str(ROOT/"configs/sprint2_collection.yaml"), "--preflight"]), 2)

    def test_selection_input_only_exact_ties_and_config_hash(self):
        inputs = [_source_inputs(self.rows[i], self.contract) for i in range(3)]
        self.assertEqual(choose_candidate("highest_pre_update_nll_v1", 1, "g", 0, inputs), 2)
        before = config_hash(self.config)
        self.assertEqual(before, config_hash({**self.config, "handoff_spec": "other-path.json"}))
        self.assertNotEqual(before, config_hash({**self.config, "selection_seed": 2}))
        with self.assertRaises(ContractError):
            choose_candidate("unapproved_h2_oracle", 1, "g", 0, inputs)

    def test_complete_t5_resume_and_restricted_bare_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"synthetic-journal"
            store = RunStore(path, self.config, self.spec)
            broken = SyntheticJournalBackend(self.rows, self.contract, failure_state=2)
            with self.assertRaises(RuntimeError):
                store.collect(self.config, self.contract, broken, [self.trajectory])
            self.assertEqual(len(store.completed()), 2)
            self.assertGreater(store.spent_seconds(), 0)
            self.assertFalse(store.lock.exists())
            attempts_before = {p.name for p in (path/"attempts").iterdir()}
            resumed = RunStore(path, self.config, self.spec, resume=True)
            out = resumed.collect(self.config, self.contract, SyntheticJournalBackend(self.rows, self.contract), [self.trajectory])
            records = [__import__("json").loads(line) for line in (out/"meta.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 15)
            self.assertEqual([sum(r["history"]["mask"]) for r in records if r["candidate_id"] == 0], [0, 1, 2, 3, 3])
            manifest = read_json(out/"manifest.json")
            self.assertEqual(manifest, {self.rows[0]["group_id"]: "train"})
            self.assertEqual(len(read_json(path/"lineage_plan.json")["splits"]), 12)
            self.assertTrue(attempts_before <= {p.name for p in (path/"attempts").iterdir()})
            validate_handoff(records, self.spec, self.contract, raw_module_layout(self.contract), manifest=manifest)
            with self.assertRaises(FileExistsError):
                RunStore(path, self.config, self.spec)
            with self.assertRaises(ContractError):
                RunStore(path, {**self.config, "selection_seed": 4}, self.spec, resume=True)
            one = resumed.completed()[0]
            artifact = path/one["records_path"]
            with artifact.open("a") as f:
                f.write("tampered SYNTHETIC fixture")
            with self.assertRaises(ContractError):
                resumed.completed()

    def test_resource_guard_preserves_outputs(self):
        config = {**self.config, "retention": {"max_device_seconds": 100, "max_output_bytes": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp)/"fixture", config, self.spec)
            with self.assertRaises(ContractError):
                store.collect(config, self.contract, SyntheticJournalBackend(self.rows, self.contract), [self.trajectory])
            self.assertTrue((store.path/"sealed.json").exists())
            self.assertFalse(store.completed())

    def test_failed_attempt_time_counts_and_unclosed_session_blocks_resume(self):
        config = {**self.config, "retention": {"max_device_seconds": 1000, "max_output_bytes": 100000000}}
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp)/"SYNTHETIC-time", config, self.spec)
            store.event("collection_started", session_id="SYNTHETIC-failed")
            store.event("collection_finished", session_id="SYNTHETIC-failed", elapsed_seconds=1001.)
            with self.assertRaises(ContractError):
                store.collect(config, self.contract, SyntheticJournalBackend(self.rows, self.contract), [self.trajectory])
            self.assertFalse(store.completed())
            store.event("collection_started", session_id="SYNTHETIC-hard-kill")
            with self.assertRaisesRegex(ContractError, "Unclosed execution session"):
                store.spent_seconds()

    def test_uniform_id_is_sealed_before_features_even_on_failure(self):
        config = {**self.config, "selection_policy": "uniform_state_seed_v1"}
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp)/"SYNTHETIC-sealed", config, self.spec)
            with self.assertRaises(RuntimeError):
                store.collect(config, self.contract, SyntheticJournalBackend(self.rows, self.contract, failure_state=0), [self.trajectory])
            attempt = next((store.path/"attempts").iterdir())
            sealed = read_json(attempt/"plan.json")
            expected = choose_candidate("uniform_state_seed_v1", 1, self.rows[0]["group_id"], 0, [])
            self.assertEqual(sealed["selected_before_features_and_labels"], expected)
            self.assertFalse((attempt/"ranking.json").exists())
            self.assertFalse(store.completed())

    def test_fully_bound_shape_fixture_and_leakage_guards_without_model(self):
        # ALL bindings/results below are fabricated for parser tests in a temp
        # directory. Passing real-mode preflight is not proof of real evidence.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = load_config(ROOT/"configs/sprint2_collection.yaml")
            config["retention"].update(mig_profile="SYNTHETIC-NOT-HARDWARE", device_total_memory_bytes=1)
            config.update(task="SYNTHETIC", corpus_version="SYNTHETIC-v1", handoff_spec="SYNTHETIC-spec.json",
                task_rules={"SYNTHETIC": "final_numeric_line_v1"}, generation_caps={"SYNTHETIC": 4},
                chat_template_sha256="0"*64, lineage_plan="SYNTHETIC-lineage.json",
                caps_frozen_before_collection=True, caps_review_reference="SYNTHETIC fixture, NOT approval")
            config["definition_agreements"].update({k: "SYNTHETIC fixture, NOT agreement"
                for k in ("D06", "D07", "D08", "D09", "D12_measurements")})
            config["definitions"] = dict(confidence="answer_mean_logprob_v1", difficulty="original_failure_fraction_v1",
                embedding="frozen_base_mean_nonspecial_v1", duplicate="nfc_word_trigram_components_v1",
                gradient="eval_unscaled_lora_ab_v1", probe="ols_cadence_parent_post_answer_kl_v2",
                correction="parent_self_correction_v1", answer_span="final_value_boundary_whitespace_v1",
                nll_reduction="equal_answer_v1", cosine_zero_epsilon=0, duplicate_threshold=.9, duplicate_tolerance=0)
            config["module_manifest"] = [{"block_index": n, "projection": "q_proj", "qualified_name": name,
                "a_name": name+".a", "a_shape": [2, 1], "b_name": name+".b", "b_shape": [1, 2]}
                for n, name in zip((2, 10), self.spec["layer_order"])]
            lineage = copy.deepcopy(self.spec["bindings"]["lineage"])
            groups = lineage["splits"]
            def rows(name, count, family):
                return [{"prompt_id": name+"-%d" % i, "source_family": family,
                         "prompt": "SYNTHETIC placeholder", "reference": "2"} for i in range(count)]
            role_rows = {role: rows(role, count, "SYNTHETIC-"+role) for role, count in
                         (("target", 32), ("ood", 32), ("retention", 256), ("confidence", 32), ("final", 32))}
            role_rows.update(anchor_probe=role_rows["retention"][:32], retention_gradient=role_rows["retention"][:32],
                             val_probe=role_rows["target"])
            for role, prompts in role_rows.items():
                path = "SYNTHETIC-"+role+".json"
                write_json(base/path, {"task_id": "SYNTHETIC", "dataset_role": role, "prompts": prompts})
                config["datasets"][role] = path
            evaluation = _digest({r: file_sha(base/config["datasets"][r]) for r in ("target", "ood", "retention")})
            for identity in lineage["identities"].values():
                identity.update(corpus_version=config["corpus_version"], evaluation_sha256=evaluation)
            for recipe in lineage["recipes"]:
                trajectory = recipe["trajectory_id"]
                group = lineage["membership"][trajectory]
                prompts = [{**r, "group_id": group} for r in rows(group, 128, recipe["source_families"][0])]
                path = "SYNTHETIC-"+group+".json"
                write_json(base/path, {"task_id": "SYNTHETIC", "dataset_role": "lm_training",
                    "split": "val" if groups[group] == "validation" else groups[group], "prompts": prompts})
                config["adaptation_manifests"][group] = path
                config["continuation"]["prompt_pools"][group] = [r["prompt_id"] for r in prompts]
                root = base/trajectory
                root.mkdir()
                # JSON deliberately isn't usable weights; preflight only audits
                # hashes/metadata. The backend must still load/audit real tensors.
                write_json(root/"adapter_model.safetensors", {"SYNTHETIC_NOT_WEIGHTS": trajectory})
                write_json(root/"adapter_config.json", {"r": 8, "lora_alpha": 16, "lora_dropout": .05,
                    "target_modules": ["q_proj", "v_proj"], "base_model_name_or_path": config["model"]["name"], "bias": "none"})
                config["roots"][trajectory] = {"path": trajectory,
                    "files": {p.name: file_sha(p) for p in root.iterdir()}}
            write_json(base/config["lineage_plan"], lineage)
            group = next(g for g, split in groups.items() if split == "train")
            measured = [CompletionMeasurement("SYNTHETIC", p, group, "train", 4, True, "final_answer", False, 512)
                        for p in config["continuation"]["prompt_pools"][group][:50]]
            calibration = base/"SYNTHETIC-calibration"
            calibration.mkdir()
            # Absolute report path exercises consistent checksum keys.
            path = str(calibration/"report.json")
            config["calibration_reports"] = [path]
            write_json(calibration/"report.json", {"model": config["model"], "generation": {"greedy": True, "thinking": False},
                "chat_template_sha256": config["chat_template_sha256"], "minimum_correct": 30, "target_prompts": 50,
                "tasks": [{"task_id": "SYNTHETIC", "answer_rule": "final_numeric_line_v1"}],
                "summaries": [summarize_calibration(measured)]})
            (calibration/"measurements.jsonl").write_text("".join(json.dumps(asdict(r))+"\n" for r in measured))
            config["continuation"]["recipe"] = {"update": config["update"], "generation": {"greedy": True, "thinking": False},
                "correction": config["definitions"]["correction"], "empty_failure_batch": "stop_v1"}
            spec = prepare_spec(config, base, ROOT/"configs/feature_schema.yaml")
            inspect_bindings(config, base, prepared_spec=spec)
            for change in ("diagnostic_count", "source_leak", "wrong_subset", "scorer_mismatch", "outcome_selection"):
                changed = copy.deepcopy(config)
                if change == "diagnostic_count":
                    changed["diagnostic_counts"]["target"] = 16
                elif change == "source_leak":
                    changed["datasets"]["final"] = changed["datasets"]["target"]
                elif change == "wrong_subset":
                    changed["datasets"]["anchor_probe"] = changed["datasets"]["confidence"]
                elif change == "scorer_mismatch":
                    changed["task_rules"]["SYNTHETIC"] = "final_choice_line_v1"
                else:
                    changed["selection_policy"] = "highest_pre_update_nll_v1"
                with self.assertRaises(ContractError):
                    inspect_bindings(changed, base, prepared_spec=spec)


if __name__ == "__main__":
    unittest.main()
