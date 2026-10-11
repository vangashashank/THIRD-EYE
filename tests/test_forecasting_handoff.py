"""Synthetic integration fixtures ONLY, never extracted research measurements."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.build_synthetic_handoff import ROOT, make_fixture, write_fixture
from scripts import run_forecasting_eval as cli
from src.third_eye.feature_contract import ContractError, load_feature_contract
from src.third_eye.forecasting.data import fixed_split, from_records
from src.third_eye.forecasting.layout import CANDIDATE, STATE
from src.third_eye.forecasting_handoff import load_handoff_spec, raw_module_layout, validate_handoff


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = make_fixture(states=5)

    def setUp(self):
        self.spec, self.records, self.manifest = copy.deepcopy(self.template)
        self.contract = load_feature_contract(ROOT / "configs/feature_schema.yaml",
            dimensions=self.spec["dimensions"], layer_order=self.spec["layer_order"])
        self.layout = raw_module_layout(self.contract)

    def validate(self):
        return validate_handoff(self.records, self.spec, self.contract, self.layout, manifest=self.manifest)

    def test_roundtrip_shapes_order_dtypes_and_all_history_lengths(self):
        self.validate()
        ds = from_records(self.records, self.layout)
        self.assertEqual((len(ds), ds.n_states), (45, 15))
        self.assertEqual(ds.state.shape, (45, 6))
        self.assertEqual(ds.cand.shape, (45, 18))
        self.assertEqual(ds.hist.shape, (45, 3, 4))
        self.assertEqual(ds.y_h1.shape, (45, 3))
        self.assertEqual(ds.y_h2.shape, (45, 3))
        for values in (ds.state, ds.cand, ds.hist, ds.hist_mask, ds.y_h1, ds.y_h2):
            self.assertEqual(values.dtype, np.float32)
        columns = self.layout.columns(CANDIDATE)
        start = columns.index("layerwise_grad_norms[0]")
        np.testing.assert_array_equal(ds.cand[:, start:start+2], np.tile([3, 4], (45, 1)))
        self.assertEqual(self.layout.columns(STATE)[-1], "current_gen_index")
        rows = self.records[:15:3]
        self.assertEqual([sum(r["history"]["mask"]) for r in rows], [0, 1, 2, 3, 3])
        self.assertTrue(all(r["provenance"]["synthetic"] for r in self.records))
        np.testing.assert_array_equal(ds.lhv2(), [r["labels"]["lhv2"] for r in self.records])

    def test_real_zero_observation_still_has_mask_one(self):
        history = self.records[6]["history"]  # S2, previous applied S1->S2 has zero changes
        self.assertEqual(history["mask"], [0, 1, 1])
        self.assertTrue(all(v[-1] == 0 for v in history["values"].values()))
        self.validate()

    def test_raw_layout_runs_all_three_cpu_models_without_fitting(self):
        import torch
        from src.third_eye.forecasting.models import build_model
        ds = from_records(self.records, self.layout)
        tensors = [torch.from_numpy(v) for v in (ds.state, ds.cand, ds.hist, ds.hist_mask)]
        with torch.no_grad():
            for kind in ("one_step", "direct", "dynamics"):
                out = build_model(kind, self.layout, hidden=8, dropout=0).eval()(*tensors)
                head = "h1" if kind == "one_step" else "h2"
                self.assertEqual(tuple(out[head].shape), (45, 3))
                self.assertTrue(torch.isfinite(out[head]).all())

    def test_fractional_accuracy_keeps_measured_precision_separate_from_float32_features(self):
        from src.third_eye.feature_contract import accuracy_deltas, weighted_utility
        # Use only S0 so changing its parent does not alter any downstream lineage.
        self.records = [r for r in self.records if r["state_id"] == 0]
        for row in self.records:
            parent = dict(target=3/5, ood=2/3, retention=4/5)
            row["provenance"]["ranking_accuracy"] = parent
            for key, col in zip(parent, ("target_val_accuracy", "ood_proxy_accuracy", "retention_anchor_accuracy")):
                row["state"][col] = float(np.float32(parent[key]))
            for horizon in ("h1", "h2"):
                deltas = accuracy_deltas(parent, row["provenance"][horizon]["accuracy"])
                row["labels"].update({"delta_%s_%s" % (k, horizon): float(v) for k, v in zip(parent, deltas)})
            row["labels"]["lhv2"] = float(weighted_utility([row["labels"]["delta_%s_h2" % k] for k in parent]))
        self.validate()

    def test_labels_cannot_change_input_arrays(self):
        before = from_records(self.records, self.layout)
        for row in self.records:
            row["labels"] = {k: .1 for k in row["labels"]}
        after = from_records(self.records, self.layout, validate=False)
        for key in ("state", "cand", "hist", "hist_mask"):
            np.testing.assert_array_equal(getattr(before, key), getattr(after, key))

    def test_missing_candidate_rejected_even_though_upstream_accepts_two(self):
        self.records.pop(2)
        with self.assertRaisesRegex(ContractError, "Missing expected candidates"):
            self.validate()

    def test_missing_field_and_malformed_history(self):
        del self.records[0]["candidate"]["pre_update_nll"]
        with self.assertRaisesRegex(ValueError, "columns differ"):
            self.validate()
        self.setUp()
        self.records[0]["history"]["mask"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "mask"):
            self.validate()

    def test_nonfinite_and_label_input_rejected(self):
        self.records[0]["candidate"]["pre_update_nll"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.validate()
        self.setUp()
        self.records[0]["state"]["delta_target_h2"] = .1
        with self.assertRaisesRegex(ValueError, "columns differ"):
            self.validate()

    def test_boolean_mask_and_utility_label_rejected(self):
        self.records[0]["history"]["mask"][0] = False
        with self.assertRaisesRegex(ContractError, "mask must be numeric"):
            self.validate()
        self.setUp()
        self.records[0]["labels"]["lhv2"] = False
        with self.assertRaisesRegex(ContractError, "finite numeric label"):
            self.validate()

    def test_sibling_parent_precision_cannot_disagree_below_float32_resolution(self):
        from src.third_eye.feature_contract import accuracy_deltas, weighted_utility
        row = self.records[2]
        parent = row["provenance"]["ranking_accuracy"]
        parent["target"] += 1e-10
        for horizon in ("h1", "h2"):
            deltas = accuracy_deltas(parent, row["provenance"][horizon]["accuracy"])
            row["labels"].update({"delta_%s_%s" % (k, horizon): float(v) for k, v in zip(parent, deltas)})
        row["labels"]["lhv2"] = float(weighted_utility([row["labels"]["delta_%s_h2" % k] for k in parent]))
        with self.assertRaisesRegex(ContractError, "Siblings disagree"):
            self.validate()

    def test_missing_applied_observation_not_silently_padded(self):
        self.records[3]["provenance"]["ledger"] = []
        with self.assertRaisesRegex(ContractError, "Missing expected applied"):
            self.validate()

    def test_future_and_simulated_history_rejected(self):
        self.records[3]["provenance"]["ledger"][0]["observed_at"] = 111
        with self.assertRaisesRegex(ContractError, "unavailable"):
            self.validate()
        self.setUp()
        self.records[3]["provenance"]["ledger"][0]["record_type"] = "simulated_h2"
        with self.assertRaisesRegex(ContractError, "simulated branch"):
            self.validate()

    def test_h1_selected_identity_join_is_strict(self):
        self.records[3]["provenance"]["ledger"][0]["candidate_id"] = 1
        with self.assertRaisesRegex(ContractError, "selected candidate"):
            self.validate()

    def test_h2_endpoint_delta_not_increment_from_h1(self):
        row = self.records[1]
        h1, h2 = row["provenance"]["h1"], row["provenance"]["h2"]
        row["labels"]["delta_target_h2"] = h2["accuracy"]["target"] - h1["accuracy"]["target"]
        with self.assertRaisesRegex(ContractError, "endpoint-minus-ranking-parent"):
            self.validate()

    def test_shared_draw_allows_child_specific_batches(self):
        siblings = self.records[:3]
        self.assertEqual(siblings[0]["provenance"]["continuation"], siblings[1]["provenance"]["continuation"])
        self.assertNotEqual(siblings[0]["provenance"]["child_batch"], siblings[1]["provenance"]["child_batch"])
        self.validate()
        siblings[1]["provenance"]["continuation"]["prompt_ids"].reverse()
        with self.assertRaisesRegex(ContractError, "shared draw"):
            self.validate()

    def test_outcome_values_do_not_change_continuation_plan(self):
        row = self.records[2]  # not the applied child, so no downstream ledger changes
        before = copy.deepcopy(row["provenance"]["continuation"])
        row["provenance"]["h1"]["accuracy"]["target"] += 1/32
        row["labels"]["delta_target_h1"] += 1/32
        self.validate()
        self.assertEqual(row["provenance"]["continuation"], before)

    def test_late_plan_and_foreign_child_prompt_rejected(self):
        self.records[0]["provenance"]["plan_sealed_at"] = 11
        with self.assertRaisesRegex(ContractError, "sealed before"):
            self.validate()
        self.setUp()
        self.records[0]["provenance"]["child_batch"]["examples"] = [["foreign-prompt", "answer"]]
        with self.assertRaisesRegex(ContractError, "outside the shared"):
            self.validate()

    def test_manifest_names_coverage_and_preassignment(self):
        ds = from_records(self.records, self.layout)
        for manifest in ({}, {k: "validation" for k in self.manifest},
                         {**self.manifest, "foreign-group": "test"}):
            with self.assertRaises(ValueError):
                fixed_split(ds, manifest)
        key = next(iter(self.manifest))
        self.manifest[key] = "test" if self.manifest[key] != "test" else "train"
        with self.assertRaisesRegex(ContractError, "preassigned"):
            self.validate()

    def test_cross_group_prefix_leakage_rejected(self):
        from src.third_eye.forecasting_handoff import make_handoff_spec
        binding = self.spec["bindings"]
        a, b = binding["lineage"]["recipes"][:2]
        a["prefix_ids"] = [b["adapted_root_id"]]
        self.spec = make_handoff_spec(self.contract, binding, synthetic_only=True)
        for row in self.records:
            row["semantic_hash"] = self.spec["semantic_hash"]
        with self.assertRaisesRegex(ContractError, "cross lineage groups"):
            self.validate()

    def test_wrong_hash_order_and_identity_rejected(self):
        self.records[0]["semantic_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "semantic_hash"):
            self.validate()
        self.setUp()
        self.layout = replace(self.layout, groups=tuple(reversed(self.layout.groups)))
        with self.assertRaisesRegex(ContractError, "layout order"):
            self.validate()
        self.setUp()
        self.records[0]["provenance"]["identity"]["root_sha256"] = "b" * 64
        with self.assertRaisesRegex(ContractError, "pinned corpus"):
            self.validate()

    def test_zero_gradient_diagnostics_required(self):
        self.records[0]["provenance"]["zero_gradient_diagnostics"] = {}
        with self.assertRaisesRegex(ContractError, "zero-gradient diagnostics required"):
            self.validate()

    def test_duplicate_descendant_ids_rejected_but_equal_weight_hashes_are_not_ids(self):
        h1 = self.records[1]["provenance"]["h1"]
        h1["endpoint_id"] = self.records[0]["provenance"]["h1"]["endpoint_id"]
        with self.assertRaisesRegex(ContractError, "Duplicate descendant"):
            self.validate()
        self.setUp()
        row = self.records[1]
        row["provenance"]["h1"]["endpoint_sha256"] = self.records[0]["provenance"]["h1"]["endpoint_sha256"]
        row["provenance"]["h2"]["parent_sha256"] = row["provenance"]["h1"]["endpoint_sha256"]
        self.validate()

    def test_shared_actual_root_cannot_hide_behind_different_recipe_names(self):
        from src.third_eye.forecasting_handoff import make_handoff_spec
        binding = self.spec["bindings"]
        a, b = list(binding["lineage"]["identities"])[:2]
        binding["lineage"]["identities"][b]["root_sha256"] = binding["lineage"]["identities"][a]["root_sha256"]
        self.spec = make_handoff_spec(self.contract, binding, synthetic_only=True)
        for row in self.records:
            row["semantic_hash"] = self.spec["semantic_hash"]
        with self.assertRaisesRegex(ContractError, "cross lineage groups"):
            self.validate()

    def test_cli_validate_only_really_reads_manifest_and_does_not_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            out = write_fixture(Path(temp) / "synthetic")
            load_handoff_spec(out / "handoff_spec.json")
            argv = ["run_forecasting_eval", "--records", str(out / "records.jsonl"), "--validate-only",
                    "--split-manifest", str(out / "split_manifest.json"),
                    "--handoff-spec", str(out / "handoff_spec.json")]
            with patch("sys.argv", argv), patch.object(cli, "fit") as fit, contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli.main()
            fit.assert_not_called()
            self.assertIn("synthetic=True", stdout.getvalue())
            self.assertIn("split manifest:", stdout.getvalue())
            (out / "split_manifest.json").write_text("{}")
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "manifest"):
                cli.main()
            argv = argv[:-2]  # also check the upstream/default path, not only our adapter
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                cli.main()

    def test_fixture_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = write_fixture(Path(temp) / "synthetic")
            with self.assertRaises(FileExistsError):
                write_fixture(out)


class SemanticHashCliTests(unittest.TestCase):
    """Both contributors' CLI options; synthetic records only, no fitting."""

    def invoke(self, argv):
        with patch("sys.argv", ["run_forecasting_eval", *argv]), \
                patch.object(cli, "fit") as fit, \
                patch.object(cli, "cross_validate") as cross_validate, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            cli.main()
        fit.assert_not_called()
        cross_validate.assert_not_called()
        return stdout.getvalue()

    def test_each_hash_flag_enforces_plain_record_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SYNTHETIC-records.jsonl"
            rows = cli.make_synthetic_records(cli._layout("contract", 6, "all"),
                n_trajectories=3, states_per_trajectory=2, n_groups=3,
                semantic_hash="synthetic-definitions-only")
            path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
            base = ["--records", str(path), "--validate-only"]
            for flag in ("--semantic-hash", "--expected-semantic-hash"):
                with self.subTest(flag=flag):
                    self.assertIn("validation:", self.invoke([*base, flag, "synthetic-definitions-only"]))
                    with self.assertRaisesRegex(ValueError, "semantic_hash"):
                        self.invoke([*base, flag, "different-synthetic-definitions"])
            for row in rows:
                row.pop("semantic_hash")
            path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
            for flag in ("--semantic-hash", "--expected-semantic-hash"):
                with self.subTest(missing_hash=flag), self.assertRaisesRegex(ValueError, "semantic_hash"):
                    self.invoke([*base, flag, "synthetic-definitions-only"])

    def test_each_hash_flag_preserves_strict_handoff_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            out = write_fixture(Path(temp) / "SYNTHETIC-handoff")
            spec = json.loads((out / "handoff_spec.json").read_text())
            base = ["--records", str(out / "records.jsonl"), "--validate-only",
                    "--split-manifest", str(out / "split_manifest.json"),
                    "--handoff-spec", str(out / "handoff_spec.json")]
            for flag in ("--semantic-hash", "--expected-semantic-hash"):
                with self.subTest(flag=flag):
                    self.assertIn("synthetic=True", self.invoke([*base, flag, spec["semantic_hash"]]))
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        self.invoke([*base, flag, "wrong-synthetic-hash"])
                    self.assertEqual(raised.exception.code, 2)
            self.assertIn("synthetic=True", self.invoke([*base,
                "--semantic-hash", spec["semantic_hash"],
                "--expected-semantic-hash", spec["semantic_hash"]]))
            (out / "split_manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest"):
                self.invoke([*base, "--semantic-hash", spec["semantic_hash"]])

    def test_conflicting_or_empty_hashes_fail_before_record_read(self):
        for flags in (["--semantic-hash", "a", "--expected-semantic-hash", "b"],
                      ["--semantic-hash", ""], ["--expected-semantic-hash", ""]):
            with self.subTest(flags=flags), patch.object(cli, "read_jsonl") as read, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                self.invoke(["--records", "SYNTHETIC-do-not-read.jsonl", "--validate-only", *flags])
            self.assertEqual(raised.exception.code, 2)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
