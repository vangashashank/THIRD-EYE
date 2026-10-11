"""Torch tests: architecture invariants, checkpoints, and training controls.

The training tests use synthetic planted-signal data.  They show the pipeline
recovers a known signal (positive control) and does not invent one (negative
control).  They are not evidence about the real LLM meta-dataset.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.third_eye.forecasting import metrics as M
from src.third_eye.forecasting.data import from_records
from src.third_eye.forecasting.layout import contract_layout
from src.third_eye.forecasting.models import MaskedGRU, build_model, num_parameters
from src.third_eye.forecasting.synthetic import make_synthetic_records
from src.third_eye.forecasting.train import (
    TrainConfig, _pair_loss, constant_baseline, cross_validate, fit, load_fit, predict, save_fit)

LAYOUT = contract_layout(6, h1_heads=("target", "ood", "retention"))


def tiny_ds(**kw):
    kw.setdefault("n_trajectories", 8)
    kw.setdefault("states_per_trajectory", 4)
    return from_records(make_synthetic_records(LAYOUT, **kw), LAYOUT)


def batch_of(ds, rows=None):
    rows = np.arange(len(ds)) if rows is None else rows
    return [torch.from_numpy(a[rows]).float() for a in (ds.state, ds.cand, ds.hist, ds.hist_mask)]


class ArchitectureTests(unittest.TestCase):
    def test_all_models_run_and_stay_small(self):
        ds = tiny_ds()
        for kind in ("one_step", "direct", "dynamics"):
            m = build_model(kind, LAYOUT)
            out = m(*batch_of(ds))
            self.assertLess(num_parameters(m), 1_000_000)
            if kind == "one_step":
                self.assertEqual(out["h1"].shape, (len(ds), 3))
            else:
                self.assertEqual(out["h2"].shape, (len(ds), 3))
        z = build_model("dynamics", LAYOUT)(*batch_of(ds))["z"]
        self.assertEqual(z.shape[0], 3)
        self.assertFalse(torch.allclose(z[1], z[2]))

    def test_masked_gru_ignores_padding(self):
        torch.manual_seed(0)
        gru = MaskedGRU(4, 8)
        x = torch.randn(5, 3, 4)
        mask = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=torch.float32)
        base = gru(x, mask)
        x2 = x.clone()
        x2[mask == 0] = torch.randn(int((mask == 0).sum()), 4) * 100      # scribble on padding
        self.assertTrue(torch.equal(base, gru(x2, mask)))
        self.assertTrue(torch.equal(base[0], torch.zeros(8)))              # empty history -> zero state
        for i in range(5):                                                  # batch == individual
            self.assertTrue(torch.allclose(base[i], gru(x[i:i + 1], mask[i:i + 1])[0], atol=1e-6))

    def test_model_output_invariant_to_padded_history_values(self):
        ds = tiny_ds()
        for kind in ("direct", "dynamics", "one_step"):
            m = build_model(kind, LAYOUT).eval()
            s, c, h, k = batch_of(ds)
            h2 = h.clone()
            h2[k == 0] = 7.0
            a, b = m(s, c, h, k), m(s, c, h2, k)
            key = "h1" if kind == "one_step" else "h2"
            self.assertTrue(torch.allclose(a[key], b[key], atol=1e-6), kind)

    def test_ablated_groups_still_build(self):
        ds = tiny_ds().drop_groups("current_state_performance", "generation_index", "recursive_history")
        self.assertEqual(ds.state.shape[1], 0)
        for kind in ("direct", "dynamics", "one_step"):
            m = build_model(kind, ds.layout).eval()
            out = m(*batch_of(ds))
            self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))

    def test_gradients_reach_parameters(self):
        ds = tiny_ds()
        for kind in ("direct", "dynamics"):
            m = build_model(kind, LAYOUT)
            out = m(*batch_of(ds))
            (out["h2"].pow(2).mean() + out["h1"].pow(2).mean()).backward()
            dead = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            self.assertEqual(dead, [], kind)

    def test_pair_loss_prefers_correct_order(self):
        sid = torch.tensor([0, 0, 0])
        true = torch.tensor([1.0, 2.0, 3.0])
        self.assertLess(float(_pair_loss(true, true, sid)), float(_pair_loss(-true, true, sid)))


class FitAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.ds = tiny_ds(n_trajectories=10)
        self.cfg = TrainConfig(kind="direct", max_epochs=15, patience=5, rank_weight=0.5)

    def test_predict_does_not_read_labels(self):
        res = fit(self.ds, None, self.cfg)
        a = predict(res, self.ds).h2
        b = predict(res, self.ds.without_labels()).h2
        np.testing.assert_array_equal(a, b)

    def test_checkpoint_roundtrip_and_guards(self):
        res = fit(self.ds, None, self.cfg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f.pt"
            save_fit(res, path, metadata={"note": "synthetic"})
            with self.assertRaises(FileExistsError):
                save_fit(res, path)
            back = load_fit(path, expected_layout=LAYOUT)
            np.testing.assert_allclose(predict(back, self.ds).h2, predict(res, self.ds).h2, atol=1e-6)
            with self.assertRaises(ValueError):
                load_fit(path, expected_layout=LAYOUT.drop("short_probe_dynamics"))
        with self.assertRaises(ValueError):                       # wrong-layout data at predict time
            predict(res, self.ds.drop_groups("short_probe_dynamics"))

    def test_all_kinds_fit_with_validation(self):
        from src.third_eye.forecasting.data import split_by_group
        a, b = split_by_group(self.ds, 0.3)
        for kind in ("one_step", "direct", "dynamics"):
            cfg = TrainConfig(kind=kind, max_epochs=10, patience=3)
            res = fit(self.ds.subset(a), self.ds.subset(b), cfg)
            self.assertTrue(all(np.isfinite(r["train_loss"]) for r in res.history))
            p = predict(res, self.ds.subset(b))
            self.assertTrue(np.isfinite(p.score()).all())


class SemanticAndBaselineTests(unittest.TestCase):
    def test_semantic_hash_is_enforced_at_predict_and_load(self):
        recs = make_synthetic_records(LAYOUT, n_trajectories=8, states_per_trajectory=3, semantic_hash="aaaa")
        ds = from_records(recs, LAYOUT)
        res = fit(ds, None, TrainConfig(kind="direct", max_epochs=3))
        self.assertEqual(res.config.semantic_hash, "aaaa")
        other = ds.subset(np.arange(len(ds)))
        other.semantic_hash = "bbbb"
        with self.assertRaises(ValueError):
            predict(res, other)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f.pt"
            save_fit(res, path)
            load_fit(path, expected_semantic_hash="aaaa")
            with self.assertRaises(ValueError):
                load_fit(path, expected_semantic_hash="bbbb")

    def test_constant_baseline_is_group_weighted_and_ties_candidates(self):
        ds = tiny_ds(n_trajectories=8, n_groups=4)
        pred = constant_baseline(ds, ds)
        self.assertTrue(np.allclose(pred.h2, pred.h2[0]))
        self.assertAlmostEqual(M.top1_accuracy(ds.lhv2(), pred.score(), ds.state_index)[0],
                               M.top1_accuracy(ds.lhv2(), np.zeros(len(ds)), ds.state_index)[0])


class TrainingControlTests(unittest.TestCase):
    """Positive/negative controls on planted-signal data."""

    CFG = dict(max_epochs=80, patience=12, lr=5e-3)

    def cv(self, ds, **over):
        cfg = TrainConfig(**{**self.CFG, **over})
        return cross_validate(ds, cfg, n_splits=4, seeds=(0,))

    def test_positive_control_future_signal_is_recovered_and_beats_one_step(self):
        ds = tiny_ds(n_trajectories=80, states_per_trajectory=5, immediate_weight=0.0,
                     signal=1.5, noise=0.3, seed=11)
        direct = self.cv(ds, kind="direct").pred.score()
        one_step = self.cv(ds, kind="one_step").pred.score()
        y = ds.lhv2()
        rep = M.ranking_report(y, direct, ds.state_index)
        ci = M.bootstrap_ci(M.m_within_state_spearman, y, direct, ds.state_index, n_boot=150)
        self.assertGreater(rep["within_state_spearman"], 0.30)
        self.assertGreater(rep["top1"], 0.45)
        self.assertGreater(ci["lo"], 0.0)
        diff = M.paired_bootstrap_diff(M.m_within_state_spearman, y, direct, one_step,
                                       ds.state_index, n_boot=150)
        self.assertGreater(diff["lo"], 0.0)          # H=2 training beats H=1 when they disagree

    def test_negative_control_pure_noise_gives_no_skill(self):
        ds = tiny_ds(n_trajectories=80, states_per_trajectory=5, null=True, seed=12)
        y = ds.lhv2()
        for kind in ("direct", "dynamics"):
            pred = self.cv(ds, kind=kind).pred.score()
            ci = M.bootstrap_ci(M.m_within_state_spearman, y, pred, ds.state_index, n_boot=150)
            self.assertLess(ci["lo"], 0.0 + 1e-9, kind)
            self.assertGreater(ci["hi"], 0.0 - 1e-9, kind)
            top1, chance = M.top1_accuracy(y, pred, ds.state_index)
            self.assertLess(top1, chance + 0.12, kind)


if __name__ == "__main__":
    unittest.main()
