"""Synthetic calibration fixtures, never measured generation or research prompts."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.calibrate_generation_caps import main
from src.third_eye.feature_contract import ContractError
from src.third_eye.generation_calibration import (
    CompletionMeasurement, final_answer, load_calibration_config,
    summarize_calibration, verify_answer,
)


NUMERIC = "final_numeric_line_v1"
CHOICE = "final_choice_line_v1"


def synthetic_measurements(n=50):
    return [CompletionMeasurement("synthetic-task", f"synthetic-prompt-{i}", "synthetic-train", "train",
                                  10+i, True, "final_answer", False, 512) for i in range(n)]


def synthetic_config(directory):
    directory = Path(directory)
    manifest = {"task_id": "synthetic-task", "split": "train", "dataset_role": "lm_training", "prompts": [
        {"prompt_id": f"synthetic-prompt-{i}", "group_id": "synthetic-train", "source_family": "synthetic-family",
         "prompt": "Synthetic fixture; not a research prompt", "reference": "1"} for i in range(55)]}
    config = {"version": "sprint2-cap-calibration-v1", "model": {"name": "synthetic-model-not-loaded",
              "revision": "synthetic-pinned-revision", "quantization": None, "local_files_only": True},
              "initial_cap": 512, "prompts_per_task": 50, "minimum_correct": 30, "selection_seed": 17,
              "training_group_ids": ["synthetic-train"], "tasks": [{"task_id": "synthetic-task",
              "training_manifest": "manifest.json", "answer_rule": NUMERIC}]}
    # Only temporary, explicitly synthetic fixtures are written by these tests.
    (directory / "manifest.json").write_text(json.dumps(manifest))
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config, manifest


class AnswerRuleTests(unittest.TestCase):
    def test_exact_numeric_and_choice_rules(self):
        self.assertEqual(final_answer("work\nFinal answer: -1/2\n", NUMERIC), "-1/2")
        self.assertTrue(verify_answer("-1/2", "-0.5", NUMERIC))
        self.assertEqual(final_answer("Final answer: C\n", CHOICE), "C")
        self.assertFalse(verify_answer("C", "A", CHOICE))
        for text in ("Final answer: c\n", "Final answer: E\n", "answer C", "Final answer: C because\n"):
            self.assertIsNone(final_answer(text, CHOICE))

    def test_newline_or_actual_eos_not_cap_terminates_answer(self):
        self.assertIsNone(final_answer("Final answer: 1", NUMERIC))
        self.assertEqual(final_answer("Final answer: 1", NUMERIC, at_eos=True), "1")
        self.assertEqual(final_answer("Final answer: 1\r\n", NUMERIC), "1")
        for text in ("Final answer: 1e2\n", "Final answer: 1/0\n", "Final answer: 1,000\n",
                     "Final answer: 1\nFinal answer: bad\n", "Final answer: 1\nFinal answer: 2\n"):
            self.assertIsNone(final_answer(text, NUMERIC))

    def test_reasoning_spans_and_bad_references_are_not_accepted(self):
        self.assertIsNone(final_answer("<think>Final answer: 1\n", NUMERIC))
        self.assertEqual(final_answer("<think>\nFinal answer: 2\n</think>\nFinal answer: 1\n", NUMERIC), "1")
        self.assertFalse(verify_answer(None, "1", NUMERIC))
        with self.assertRaises(ContractError):
            verify_answer("1", "one", NUMERIC)
        with self.assertRaises(ContractError):
            final_answer("Final answer: 1\n", "unagreed-task-scorer")


class CalibrationAnalysisTests(unittest.TestCase):
    def test_empirical_p99_not_population_reliability_or_frozen_cap(self):
        report = summarize_calibration(synthetic_measurements())
        self.assertEqual(report["empirical_correct_p99_cap_proposal"], 59)
        self.assertEqual(report["all_prompt_truncation_rate"], 0)
        self.assertEqual(report["caps_status"], "proposed_review_required")
        self.assertFalse(report["population_p99_reliable"])

    def test_insufficient_evidence_produces_no_percentile(self):
        for rows in (synthetic_measurements(20), [replace(row, correct=False) for row in synthetic_measurements()]):
            report = summarize_calibration(rows)
            self.assertIsNone(report["empirical_correct_p99_cap_proposal"])
            self.assertIn("insufficient_correct_completions", report["warnings"])

    def test_censoring_and_all_prompt_2_percent_threshold(self):
        for count in (1, 2):
            rows = synthetic_measurements()
            rows[:count] = [replace(row, correct=False, stopping_reason="max_new_tokens", truncated=True,
                                   generated_tokens=512) for row in rows[:count]]
            report = summarize_calibration(rows)
            self.assertEqual(report["all_prompt_truncation_rate"], count/50)
            self.assertIn("initial_cap_censored_measurements", report["warnings"])
            self.assertEqual("truncation_above_2_percent" in report["warnings"], count > 1)
            self.assertIsNone(report["empirical_correct_p99_cap_proposal"])

    def test_wrong_completions_are_in_projected_cap_truncation(self):
        rows = synthetic_measurements()
        rows[-2:] = [replace(row, correct=False, generated_tokens=400) for row in rows[-2:]]
        report = summarize_calibration(rows)
        self.assertEqual(report["empirical_correct_p99_cap_proposal"], 57)
        self.assertEqual(report["projected_all_prompt_truncation_rate"], .04)
        self.assertIn("proposed_cap_all_prompt_truncation_above_2_percent", report["warnings"])

    def test_malformed_mixed_or_heldout_records_fail(self):
        row = synthetic_measurements()[0]
        for bad in (replace(row, split="test"), replace(row, task_id="other"),
                    replace(row, initial_cap=256), replace(row, generated_tokens=513),
                    replace(row, truncated=True), replace(row, correct=1),
                    replace(row, stopping_reason="unknown"), replace(row, generated_tokens=0)):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                summarize_calibration([row, bad])
        with self.assertRaises(ContractError):
            summarize_calibration(["not a measurement"])


class CalibrationPreparationTests(unittest.TestCase):
    def test_valid_manifest_selection_is_training_only_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, manifest = synthetic_config(directory)
            config, tasks = load_calibration_config(path)
            self.assertEqual(config["initial_cap"], 512)
            self.assertEqual(len(tasks[0]["prompts"]), 50)
            ids = [row["prompt_id"] for row in tasks[0]["prompts"]]
            manifest["prompts"].reverse()
            (Path(directory) / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual([row["prompt_id"] for row in load_calibration_config(path)[1][0]["prompts"]], ids)

    def test_bad_roles_groups_counts_references_and_duplicate_ids_fail(self):
        cases = ("split", "role", "group", "short", "reference", "duplicate")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path, _, manifest = synthetic_config(directory)
                if case == "split":
                    manifest["split"] = "validation"
                elif case == "role":
                    manifest["dataset_role"] = "final_evaluation"
                elif case == "group":
                    manifest["prompts"][0]["group_id"] = "held-out"
                elif case == "short":
                    manifest["prompts"] = manifest["prompts"][:10]
                elif case == "reference":
                    manifest["prompts"][0]["reference"] = "one"
                else:
                    manifest["prompts"].append(manifest["prompts"][0])
                (Path(directory) / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ContractError):
                    load_calibration_config(path)

    def test_default_cli_only_validates_and_never_imports_model_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = synthetic_config(directory)
            with patch.dict("sys.modules", {"src.models.model_loader": None}), redirect_stdout(io.StringIO()) as output:
                main(["--config", str(path)])
            self.assertEqual(json.loads(output.getvalue())["status"], "validated_offline_only")
            self.assertEqual(sorted(item.name for item in Path(directory).iterdir()), ["config.yaml", "manifest.json"])

    def test_execute_requires_existing_allocation_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = synthetic_config(directory)
            with patch.dict("os.environ", {}, clear=True), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--config", str(path), "--output", str(Path(directory) / "output"), "--execute"])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((Path(directory) / "output").exists())

    def test_repository_config_is_explicitly_unbound_and_safe_yaml_rejects_code(self):
        with self.assertRaisesRegex(ContractError, "Unresolved task IDs"):
            load_calibration_config(Path(__file__).resolve().parents[1] / "configs/sprint2_generation_calibration.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text("!!python/object/apply:os.system ['echo NEVER']")
            with self.assertRaises(ContractError):
                load_calibration_config(path)

    def test_login_node_and_existing_output_are_rejected_before_model_import(self):
        for hostname, existing in (("dgx-login01", False), ("synthetic-compute", True)):
            with self.subTest(hostname=hostname), tempfile.TemporaryDirectory() as directory:
                path, _, _ = synthetic_config(directory)
                output = Path(directory) / "output"
                if existing:
                    output.mkdir()
                    (output / "untouched.txt").write_text("synthetic existing result")
                with patch.dict("os.environ", {"SLURM_JOB_ID": "synthetic-id"}), \
                        patch("socket.gethostname", return_value=hostname), \
                        patch.dict("sys.modules", {"src.models.model_loader": None}), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(["--config", str(path), "--output", str(output), "--execute"])
                self.assertEqual(error.exception.code, 2)
                if existing:
                    self.assertEqual((output / "untouched.txt").read_text(), "synthetic existing result")
                else:
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
