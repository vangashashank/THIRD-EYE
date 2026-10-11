"""Synthetic-only D13 regression/selection/checkpoint fixtures."""
import dataclasses
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from scripts.build_synthetic_handoff import make_fixture
from scripts.run_forecasting_eval import _run_d13
from src.third_eye.feature_contract import ContractError, load_feature_contract
from src.third_eye.forecasting.data import from_records, fixed_split
from src.third_eye.forecasting.layout import FeatureLayout
from src.third_eye.forecasting.train import TrainConfig, fit, predict, save_fit, load_fit
from src.third_eye.sprint2_evaluation import (Sprint2Standardizer, baseline_tables, checkpoint_score,
    constant_heads, policy_report, weighted_stats)


class D13Tests(unittest.TestCase):
    def setUp(self):
        self.spec, self.rows, self.manifest = make_fixture(states=5)
        self.ds = from_records(self.rows, FeatureLayout.from_dict(self.spec["layout"]))
        idx = fixed_split(self.ds, self.manifest)
        self.train, self.val, self.test = [self.ds.subset(idx[k]) for k in ("train", "val", "test")]

    def test_native_targets_padding_training_only(self):
        norm = Sprint2Standardizer().fit(self.train)
        self.assertTrue(np.array_equal(norm.transform(self.train)["y_h2"], self.train.y_h2))
        self.assertTrue(np.all(norm.transform(self.val)["hist"][self.val.hist_mask == 0] == 0))
        before = norm.to_dict()
        self.val.state[:] = 999
        self.val.y_h2[:] = 999
        norm.transform(self.val)
        self.assertEqual(before, norm.to_dict())
        with self.assertRaises(ContractError):
            norm.add_model_stats(self.val)
        mean, scale = weighted_stats([[0], [10]], [.75, .25])
        self.assertEqual(mean[0], 2.5)

    def test_exact_ties_expected_regret_and_group_ci(self):
        scores = np.zeros(len(self.test))
        report = policy_report(self.test, scores, bootstrap=10)
        self.assertAlmostEqual(report["tie_size"], 3)
        self.assertAlmostEqual(report["top1"], 1/3)
        self.assertAlmostEqual(report["selected_retention_h2_delta"], self.test.y_h2[:, 2].mean(), places=7)
        self.assertIsNone(report["ci95"])
        near = scores.copy()
        near[self.test.candidate_id == 0] = 1e-15
        self.assertAlmostEqual(policy_report(self.test, near, bootstrap=0)["tie_size"], 1)
        table = baseline_tables(self.train, self.test, {}, bootstrap=0)
        self.assertEqual(table["policy_table"]["constant"]["regret"], table["policy_table"]["uniform_random"]["regret"])
        self.assertAlmostEqual(table["policy_table"]["highest_pre_update_nll"]["tie_size"], 1)
        self.assertIn("constant_minus_highest_pre_update_nll", table["paired_against_each_baseline"])

    def test_selection_and_hashed_checkpoint_roundtrip(self):
        cfg = TrainConfig(kind="direct", hidden=8, dropout=0, max_epochs=4, patience=4, protocol="sprint2_d13")
        result = fit(self.train, self.val, cfg)
        best = min(result.history, key=lambda r: (r["val_lhv_regret"], r["val_native_mse"]))
        self.assertEqual(result.best_epoch, best["epoch"])
        self.assertEqual(checkpoint_score(self.val, predict(result, self.val)),
                         (best["val_lhv_regret"], best["val_native_mse"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"synthetic.pt"
            save_fit(result, path)
            loaded = load_fit(path, expected_semantic_hash=self.spec["semantic_hash"])
            self.assertEqual(loaded.config.protocol, "sprint2_d13")
            self.assertIsInstance(loaded.standardizer, Sprint2Standardizer)
            np.testing.assert_allclose(predict(loaded, self.test).h2, predict(result, self.test).h2, atol=1e-7)
            with self.assertRaises(ValueError):
                predict(loaded, dataclasses.replace(self.test, semantic_hash=""))
        with self.assertRaises(ValueError):
            fit(self.train, self.train, cfg)
        with self.assertRaises(ValueError):
            fit(self.train, None, cfg)
        with self.assertRaises(ValueError):
            fit(self.train, self.val, dataclasses.replace(cfg, utility_weights=(1., 0., 0.)))

    def test_constant_does_not_use_test_labels(self):
        mean, _ = constant_heads(self.train)
        self.test.y_h2[:] = 99
        other, _ = constant_heads(self.train)
        np.testing.assert_array_equal(mean, other)

    def test_opt_in_cli_writes_synthetic_tables_and_refuses_overwrite(self):
        import json
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = path/"SYNTHETIC-manifest.json"
            manifest.write_text(json.dumps(self.manifest))
            args = types.SimpleNamespace(split_manifest=str(manifest), relative=False, ablate=False, rank_weight=0,
                models="one_step,direct", out=str(path/"SYNTHETIC-CPU-fit"), synthetic=False, boot=5)
            report = types.SimpleNamespace(summary=lambda: "SYNTHETIC fixtures only")
            def tiny_fit(train, val, cfg):
                return fit(train, val, dataclasses.replace(cfg, max_epochs=2, patience=2, hidden=8, dropout=0))
            with mock.patch("scripts.run_forecasting_eval.fit", side_effect=tiny_fit):
                _run_d13(args, self.ds, report, (self.spec, None, None))
            result = json.loads((Path(args.out)/"metrics.json").read_text())
            self.assertTrue(result["synthetic"])
            self.assertFalse(result["dynamics_active"])
            with (Path(args.out)/"policy_table.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            self.assertIn("selected_retention_h2_delta", rows[0])
            with self.assertRaises(ValueError):
                _run_d13(args, self.ds, report, (self.spec, None, None))


if __name__ == "__main__":
    unittest.main()
