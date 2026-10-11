"""Numpy-only tests for the Third Eye forecasting stack.

Fixtures with hand-computed answers.  Synthetic data here exercises code paths;
it says nothing about real forecasting accuracy.
"""
import copy
import unittest
from pathlib import Path

import numpy as np

from src.third_eye.feature_contract import load_feature_contract
from src.third_eye.forecasting import features as F
from src.third_eye.forecasting.data import (
    DEFAULT_WEIGHTS, RecordError, Standardizer, expected_history_count, fixed_split,
    from_records, grouped_kfold, leave_one_group_out, split_by_group, utility, validate_records)
from src.third_eye.forecasting.semantics import SemanticConfig
from src.third_eye.forecasting.layout import (
    FeatureGroup, FeatureLayout, contract_layout, depth_bin_names, legacy_layout)
from src.third_eye.forecasting import metrics as M
from src.third_eye.forecasting import selection as S
from src.third_eye.forecasting.synthetic import make_synthetic_records

SCHEMA = Path(__file__).resolve().parents[1] / "configs" / "feature_schema.yaml"
LAYOUT = contract_layout(6, h1_heads=("target", "ood", "retention"))


class FeatureDefinitionTests(unittest.TestCase):
    def test_depth_bins_identity_constant_and_cross_depth(self):
        v = np.arange(6.0)
        np.testing.assert_allclose(F.depth_bin_pool(v, 6), v)
        np.testing.assert_allclose(F.depth_bin_pool(np.full(28, 3.0), 6), np.full(6, 3.0))
        np.testing.assert_allclose(F.depth_bin_pool(np.arange(36.0), 6),
                                   [2.5, 8.5, 14.5, 20.5, 26.5, 32.5])
        for L in (28, 32, 34, 36):          # different backbone depths, same width
            self.assertEqual(F.depth_bin_pool(np.ones(L), 6).shape, (6,))
        pooled = F.depth_bin_pool([1.0, 3.0], 6)   # fewer blocks than bins
        self.assertEqual(pooled.shape, (6,))
        np.testing.assert_allclose(pooled, [1, 1, 1, 3, 3, 3])
        total = np.random.RandomState(0).rand(28)
        self.assertAlmostEqual(F.depth_bin_pool(total, 6, "sum").sum(), total.sum())

    def test_energy_fractions_are_scale_and_depth_free(self):
        a = F.layerwise_energy_fractions(np.ones(28), 6)
        b = F.layerwise_energy_fractions(10 * np.ones(36), 6)
        np.testing.assert_allclose(a, np.full(6, 1 / 6))
        np.testing.assert_allclose(b, a)
        self.assertAlmostEqual(F.layerwise_energy_fractions(np.random.rand(30), 6).sum(), 1.0)
        np.testing.assert_allclose(F.layerwise_energy_fractions(np.zeros(5), 4), np.full(4, 0.25))
        with self.assertRaises(ValueError):
            F.layerwise_energy_fractions([1.0, -1.0], 2)

    def test_data_statistics(self):
        self.assertEqual(F.token_length_stats([2, 4]), [3.0, 1.0])
        self.assertEqual(F.token_length_stats([7]), [7.0, 0.0])
        # token-weighted (13/6), not mean of per-example means (2.5)
        self.assertAlmostEqual(F.token_weighted_nll([10.0, 3.0], [5, 1]), 13 / 6)
        mean, std = F.sequence_confidence([[-1.0, -3.0], [-2.0]])
        self.assertAlmostEqual(mean, -2.0)
        self.assertAlmostEqual(std, 0.0)
        self.assertEqual(F.verifier_pass_rate([True, True, False, False]), 0.5)
        with self.assertRaises(ValueError):
            F.verifier_pass_rate([])
        self.assertAlmostEqual(
            F.difficulty_proxy([[True, False, False, False], [False] * 4]), (0.75 + 1.0) / 2)

    def test_embedding_diversity(self):
        self.assertAlmostEqual(F.embedding_diversity(np.eye(2)), 1.0)
        self.assertAlmostEqual(F.embedding_diversity(np.array([[1., 0], [2., 0]])), 0.0)
        self.assertEqual(F.embedding_diversity(np.array([[1., 2.]])), 0.0)
        with self.assertRaises(ValueError):
            F.embedding_diversity(np.array([[0., 0.], [1., 0.]]))

    def test_duplicate_rate_text_rule_d07(self):
        A = "solve for x when 2x plus 3 equals 11 today"
        B = "a completely different question about triangles and areas"
        self.assertAlmostEqual(F.duplicate_rate([A, A, B]), 1 / 3)       # extra copies / N
        self.assertAlmostEqual(F.duplicate_rate([A, A, A]), 2 / 3)
        self.assertAlmostEqual(F.duplicate_rate([B, A, A]), 1 / 3)       # order free
        self.assertEqual(F.duplicate_rate([A]), 0.0)
        self.assertAlmostEqual(F.duplicate_rate(["  " + A.replace(" ", "   ") + " ", A]), 0.5)  # whitespace
        self.assertEqual(F.duplicate_rate([A, A.upper()]), 0.0)          # case is kept
        self.assertAlmostEqual(F.duplicate_rate(["yes", "yes", "no"]), 1 / 3)   # <3 tokens: exact match
        # 9/10 trigram overlap links at 0.90, not at 0.95
        base = " ".join("w%d" % i for i in range(12))                    # 10 trigrams
        near = base + " extra"                                           # 11 trigrams, Jaccard 10/11 = .909
        self.assertAlmostEqual(F.duplicate_rate([base, near], 0.90), 0.5)
        self.assertAlmostEqual(F.duplicate_rate([base, near], 0.95), 0.0)
        with self.assertRaises(ValueError):
            F.duplicate_rate([])

    def test_gradient_features(self):
        self.assertAlmostEqual(F.grad_l2_norm([3.0, 4.0]), 5.0)
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(F.grad_cosine(a, a), 1.0)
        self.assertAlmostEqual(F.grad_cosine(a, -a), -1.0)
        self.assertAlmostEqual(F.grad_cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(F.grad_cosine(np.zeros(3), a), 0.0)
        with self.assertRaises(ValueError):
            F.grad_cosine([1, 2], [1, 2, 3])
        # D09: all elements, sign(0)=0.  Signs (1,-1,0,1) vs (1,1,1,-1) agree only at idx 0
        self.assertAlmostEqual(F.sign_agreement([1, -1, 0, 2], [1, 1, 5, -2]), 0.25)
        self.assertAlmostEqual(F.sign_agreement([3, 4], [3, -4]), 0.5)      # Shashank's check
        self.assertEqual(F.sign_agreement([0, 0], [0, 0]), 1.0)             # zeros agree...
        self.assertEqual(F.zero_sign_stats([0, 0, 1, 1], [0, 1, 0, 1])["zero_in_both"], 0.25)  # ...so log them
        self.assertEqual(F.sign_agreement([0, 0], [1, 1]), 0.0)
        with self.assertRaises(ValueError):
            F.grad_l2_norm([1.0, float("nan")])

    def test_probe_features(self):
        self.assertAlmostEqual(F.linear_slope([1, 3, 5, 7]), 2.0)
        self.assertAlmostEqual(F.linear_slope([5.0, 3.0, 1.0], x=[0, 5, 10]), -0.4)
        with self.assertRaises(ValueError):
            F.linear_slope([1.0])
        z = np.random.RandomState(0).normal(size=(4, 7))
        self.assertAlmostEqual(F.mean_token_kl(z, z), 0.0, places=12)
        p, q = np.log([[0.5, 0.5]]), np.log([[0.9, 0.1]])
        expected = 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)
        self.assertAlmostEqual(F.mean_token_kl(p, q), expected, places=9)
        self.assertGreater(abs(F.mean_token_kl(p, q) - F.mean_token_kl(q, p)), 0.05)  # asymmetric
        both = np.vstack([p, q])
        self.assertAlmostEqual(F.mean_token_kl(both, np.vstack([p, p]), mask=[1, 0]), 0.0)
        self.assertAlmostEqual(F.param_delta_norm([0, 0], [3, 4]), 5.0)


class LayoutTests(unittest.TestCase):
    def test_widths_match_shashanks_contract_loader(self):
        B = 6
        contract = load_feature_contract(
            str(SCHEMA), dimensions={"num_lora_layers": B, "history_len": 3},
            layer_order=list(depth_bin_names(B)))
        summary = contract.dimension_summary()
        layout = contract_layout(B)
        self.assertEqual(layout.dim("state") - 1, summary["static_state_feature_dim"])  # minus gen index
        self.assertEqual(layout.dim("candidate"), summary["static_candidate_feature_dim"])
        self.assertEqual(len(layout.history_channels) * layout.history_len + layout.history_len,
                         summary["history_feature_dim"])   # 4 channels + mask, x H

    def test_column_order_follows_yaml(self):
        contract = load_feature_contract(
            str(SCHEMA), dimensions={"num_lora_layers": 6, "history_len": 3},
            layer_order=list(depth_bin_names(6)))
        expected = []
        for g in contract.input_groups:
            if g.name in ("candidate_data_stats", "gradient_interference_stats", "short_probe_dynamics"):
                for f in g.fields:
                    expected.extend(depth_bin_names(6) if f.name == "layerwise_grad_norms" else [f.name])
        self.assertEqual(contract_layout(6).columns("candidate"), expected)

    def test_drop_fingerprint_roundtrip(self):
        L = contract_layout(6)
        smaller = L.drop("short_probe_dynamics")
        self.assertEqual(smaller.dim("candidate"), L.dim("candidate") - 5)
        self.assertNotEqual(smaller.fingerprint(), L.fingerprint())
        self.assertEqual(FeatureLayout.from_dict(L.to_dict()).fingerprint(), L.fingerprint())
        with self.assertRaises(KeyError):
            L.drop("nope")
        swapped = FeatureLayout("x", tuple(reversed(L.groups)), 3)
        self.assertNotEqual(swapped.fingerprint(), FeatureLayout("x", L.groups, 3).fingerprint())
        self.assertEqual(L.group_slice("current_state_performance"), slice(0, 5))
        self.assertFalse(L.drop("recursive_history").has_history)

    def test_legacy_layout_and_duplicate_columns(self):
        self.assertEqual((legacy_layout().dim("state"), legacy_layout().dim("candidate")), (3, 4))
        with self.assertRaises(ValueError):
            FeatureLayout("dup", (FeatureGroup("a", "state", ("x",)),
                                  FeatureGroup("b", "candidate", ("x",))))


class RecordValidationTests(unittest.TestCase):
    def setUp(self):
        self.recs = make_synthetic_records(LAYOUT, n_trajectories=4, states_per_trajectory=4, seed=3)

    def bad(self, mutate, msg=None):
        recs = copy.deepcopy(self.recs)
        mutate(recs)
        with self.assertRaises(RecordError, msg=msg):
            validate_records(recs, LAYOUT)

    def test_valid_records_pass_with_only_the_few_groups_warning(self):
        rep = validate_records(self.recs, LAYOUT)
        self.assertEqual((rep.n_trajectories, rep.n_states), (4, 16))
        self.assertEqual(len(rep.warnings), 1)
        self.assertIn("lineage groups", rep.warnings[0])

    def test_rejects_contract_violations(self):
        self.bad(lambda r: r[0]["state"].update({"target_val_accuracy": float("nan")}))
        self.bad(lambda r: r[0]["candidate"].pop("grad_norm"))
        self.bad(lambda r: r[0]["candidate"].update({"extra": 1.0}))
        self.bad(lambda r: r[0]["labels"].pop("delta_target_h2"))
        self.bad(lambda r: r[0]["state"].update({"current_gen_index": 99.0}))
        # history policy: right aligned, zero padded, count == min(t, H)
        idx = next(i for i, r in enumerate(self.recs) if r["state_id"] == 2)
        self.bad(lambda r: r[idx]["history"].update({"mask": [1.0, 0.0, 1.0]}))
        self.bad(lambda r: r[idx]["history"]["values"]["prev_target_change"].__setitem__(0, 0.5))
        self.bad(lambda r: r[idx]["history"].update({"mask": [0.0, 0.0, 1.0]}))   # t=2 needs 2 entries
        # siblings must share state; a state needs >= 2 candidates; one model per trajectory
        self.bad(lambda r: r[1]["state"].update({"target_val_accuracy": 123.0}))
        self.bad(lambda r: [r.pop(1), r.pop(1)])                   # leaves a 1-candidate state
        self.bad(lambda r: r[0].update({"model": "other"}))

    def test_label_named_input_columns_rejected(self):
        bad_layout = FeatureLayout("leaky", (
            FeatureGroup("s", "state", ("a",)), FeatureGroup("c", "candidate", ("delta_target_h2",))))
        with self.assertRaises(RecordError):
            validate_records(self.recs, bad_layout)

    def test_history_alignment_warning(self):
        recs = copy.deepcopy(self.recs)
        for r in recs:
            if r["state_id"] == 2:
                r["history"]["values"]["prev_target_change"][-1] += 5.0
        rep = validate_records(recs, LAYOUT)
        self.assertTrue(any("history alignment" in w for w in rep.warnings))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.recs = make_synthetic_records(LAYOUT, n_trajectories=10, states_per_trajectory=4, seed=5)
        self.ds = from_records(self.recs, LAYOUT)

    def test_inputs_do_not_depend_on_labels(self):
        mutated = copy.deepcopy(self.recs)
        for r in mutated:
            for k in r["labels"]:
                r["labels"][k] = 1e3
        other = from_records(mutated, LAYOUT)
        for name in ("state", "cand", "hist", "hist_mask"):
            np.testing.assert_array_equal(getattr(self.ds, name), getattr(other, name))
        self.assertTrue(np.isnan(self.ds.without_labels().y_h2).all())

    def test_shapes(self):
        n = len(self.recs)
        self.assertEqual(self.ds.state.shape, (n, 6))
        self.assertEqual(self.ds.cand.shape, (n, 22))
        self.assertEqual(self.ds.hist.shape, (n, 3, 4))
        self.assertEqual(self.ds.y_h1.shape, (n, 3))
        self.assertEqual(self.ds.n_states, 40)

    def test_folds_never_split_trajectories_or_siblings(self):
        folds = grouped_kfold(self.ds, 5, seed=1)
        covered = np.concatenate([te for _, te in folds])
        self.assertEqual(sorted(covered), list(range(len(self.ds))))
        for tr, te in folds:
            self.assertFalse(set(self.ds.trajectory[tr]) & set(self.ds.trajectory[te]))
            self.assertFalse(set(self.ds.state_index[tr]) & set(self.ds.state_index[te]))
        again = grouped_kfold(self.ds, 5, seed=1)
        for (a, b), (c, d) in zip(folds, again):
            np.testing.assert_array_equal(b, d)
        with self.assertRaises(ValueError):
            grouped_kfold(self.ds, 11)
        tr, va = split_by_group(self.ds, 0.2, seed=0)
        self.assertFalse(set(self.ds.trajectory[tr]) & set(self.ds.trajectory[va]))

    def test_leave_one_model_out(self):
        a = make_synthetic_records(LAYOUT, n_trajectories=3, model="m-a", seed=1)
        b = make_synthetic_records(LAYOUT, n_trajectories=3, model="m-b", seed=2)
        ds = from_records(a + b, LAYOUT)
        held = {g: (tr, te) for g, tr, te in leave_one_group_out(ds, "model")}
        self.assertEqual(set(held), {"m-a", "m-b"})
        tr, te = held["m-a"]
        self.assertEqual(set(ds.model[te]), {"m-a"})
        self.assertEqual(set(ds.model[tr]), {"m-b"})

    def test_standardizer_uses_train_stats_and_keeps_padding_zero(self):
        tr, te = grouped_kfold(self.ds, 2, seed=0)[0]
        train, test = self.ds.subset(tr), self.ds.subset(te)
        norm = Standardizer().fit(train)
        z = norm.transform(train)
        np.testing.assert_allclose(z["state"].mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(z["cand"].std(0), 1, atol=1e-4)
        before = copy.deepcopy(norm.stats)
        zt = norm.transform(test)
        np.testing.assert_allclose(norm.stats["global"]["cand"][0], before["global"]["cand"][0])
        self.assertTrue(np.all(zt["hist"][test.hist_mask == 0] == 0))
        round_trip = Standardizer.from_dict(norm.to_dict())
        np.testing.assert_allclose(round_trip.transform(test)["cand"], zt["cand"], atol=1e-6)
        np.testing.assert_allclose(norm.inverse_h2(z["y_h2"]), train.y_h2, atol=1e-5)

    def test_per_model_norm_needs_stats_for_unseen_model(self):
        a = from_records(make_synthetic_records(LAYOUT, n_trajectories=3, model="m-a", seed=1), LAYOUT)
        b = from_records(make_synthetic_records(LAYOUT, n_trajectories=3, model="m-b", seed=2), LAYOUT)
        norm = Standardizer(per_model=True).fit(a)
        with self.assertRaises(KeyError):
            norm.transform(b)
        norm.add_model_stats(b)                      # unlabeled features only
        self.assertEqual(norm.transform(b)["cand"].shape, b.cand.shape)

    def test_drop_groups_and_relative_candidates(self):
        no_probe = self.ds.drop_groups("short_probe_dynamics")
        self.assertEqual(no_probe.cand.shape[1], 17)
        np.testing.assert_array_equal(no_probe.cand, np.delete(self.ds.cand, range(17, 22), axis=1))
        no_hist = self.ds.drop_groups("recursive_history")
        self.assertEqual(no_hist.hist.shape, (len(self.ds), 0, 0))
        rel = self.ds.with_relative_candidates()
        self.assertEqual(rel.cand.shape[1], 44)
        for s in np.unique(rel.state_index):
            rows = rel.state_index == s
            np.testing.assert_allclose(rel.cand[rows][:, 22:].mean(0), 0, atol=1e-5)


class MetricTests(unittest.TestCase):
    sid = np.repeat(np.arange(4), 3)

    def truth(self):
        return np.array([1., 2, 3, 0, 5, 4, 7, 6, 8, 2, 1, 3])

    def test_perfect_and_reversed(self):
        t = self.truth()
        r = M.ranking_report(t, t, self.sid)
        self.assertAlmostEqual(r["within_state_spearman"], 1.0)
        self.assertAlmostEqual(r["top1"], 1.0)
        self.assertAlmostEqual(r["ndcg"], 1.0)
        self.assertAlmostEqual(r["regret"], 0.0)
        rev = M.ranking_report(t, -t, self.sid)
        self.assertAlmostEqual(rev["within_state_spearman"], -1.0)
        self.assertAlmostEqual(rev["top1"], 0.0)
        self.assertGreater(rev["regret"], 0)

    def test_constant_prediction_equals_chance(self):
        t = self.truth()
        top1, chance = M.top1_accuracy(t, np.zeros(12), self.sid)
        self.assertAlmostEqual(top1, 1 / 3)
        self.assertAlmostEqual(chance, 1 / 3)
        expected_regret = np.mean([t[r].max() - t[r].mean() for r in M.state_groups(self.sid)])
        self.assertAlmostEqual(M.selection_regret(t, np.zeros(12), self.sid), expected_regret)

    def test_hand_computed_kendall_ndcg_ties(self):
        one = np.zeros(3, dtype=int)
        self.assertAlmostEqual(M.mean_state_kendall([1, 2, 3], [1, 3, 2], one), 1 / 3)
        # truth tie for best, prediction picks one of them -> full credit; chance 2/3
        top1, chance = M.top1_accuracy([1, 1, 0], [1, 0, 0], one)
        self.assertEqual((top1, chance), (1.0, 2 / 3))
        # reversed order of relevance [2,1,0]: DCG 1/log2(3)+2/log2(4) over ideal 2+1/log2(3)
        want = (1 / np.log2(3) + 2 / 2) / (2 + 1 / np.log2(3))
        self.assertAlmostEqual(M.ndcg([3, 2, 1], [1, 2, 3], one), want)
        self.assertEqual(M.ndcg([1, 1, 1], [3, 2, 1], one), 1.0)

    def test_pooled_spearman_is_fooled_by_state_offsets_within_is_not(self):
        rng = np.random.RandomState(0)
        S = 200
        sid = np.repeat(np.arange(S), 3)
        offset = np.repeat(rng.normal(0, 3, S), 3)
        truth = offset + rng.normal(0, 1, 3 * S)
        pred = offset + rng.normal(0, 1, 3 * S)             # knows the state, not the candidate
        self.assertGreater(M.pooled_spearman(truth, pred), 0.7)
        self.assertLess(abs(M.within_state_spearman(truth, pred, sid)), 0.2)

    def test_bootstrap_ci_and_paired_difference(self):
        rng = np.random.RandomState(1)
        S = 150
        sid = np.repeat(np.arange(S), 3)
        t = rng.normal(size=3 * S)
        good = t + rng.normal(0, 0.7, 3 * S)
        bad = rng.normal(size=3 * S)
        ci = M.bootstrap_ci(M.m_within_state_spearman, t, good, sid, n_boot=200, seed=0)
        self.assertLess(ci["lo"], ci["point"])
        self.assertLess(ci["point"], ci["hi"])
        self.assertGreater(ci["lo"], 0.3)
        same = M.paired_bootstrap_diff(M.m_top1, t, good, good, sid, n_boot=100)
        self.assertEqual(same["diff"], 0.0)
        diff = M.paired_bootstrap_diff(M.m_within_state_spearman, t, good, bad, sid, n_boot=200)
        self.assertGreater(diff["lo"], 0.0)
        self.assertLess(diff["p_two_sided"], 0.05)

    def test_mismatch_rate(self):
        sid = np.repeat(np.arange(4), 2)
        h1 = np.array([1, 0, 0, 1, 1, 0, 0, 1.0])
        h2 = np.array([1, 0, 1, 0, 0, 1, 0, 1.0])       # states 1 and 2 flip
        r = M.immediate_future_mismatch(h1, h2, sid)
        self.assertEqual(r["mismatch_rate"], 0.5)
        self.assertEqual(M.immediate_future_mismatch(h1, h2, sid, margin=2.0)["meaningful_mismatch_rate"], 0.0)

    def test_regression_report(self):
        y = np.array([[1., -1], [2, -2], [3, -3], [4, -4]])
        rep = M.regression_report(y, y, ["a", "b"])
        for head in ("a", "b"):
            self.assertAlmostEqual(rep[head]["mae"], 0)
            self.assertAlmostEqual(rep[head]["skill_vs_constant"], 1)
            self.assertAlmostEqual(rep[head]["calibration_slope"], 1)
            self.assertEqual(rep[head]["sign_accuracy"], 1.0)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ds = from_records(make_synthetic_records(
            LAYOUT, n_trajectories=30, states_per_trajectory=4, immediate_weight=0.2, seed=2), LAYOUT)

    def test_replay_bounds(self):
        ds = self.ds
        oracle = S.replay_policy(ds, S.future_oracle_scores(ds))
        rand = S.replay_policy(ds, S.random_scores(ds))
        self.assertAlmostEqual(float(oracle.regret.mean()), 0.0)
        sib_mean = np.mean([ds.lhv2()[r].mean() for r in M.state_groups(ds.state_index)])
        self.assertAlmostEqual(float(rand.lhv2.mean()), sib_mean)
        self.assertGreater(oracle.lhv2.mean(), rand.lhv2.mean())
        greedy = S.replay_policy(ds, S.greedy_immediate_oracle_scores(ds))
        self.assertLess(greedy.lhv2.mean(), oracle.lhv2.mean())     # immediate != future here
        table = S.compare_policies(ds, {"random": S.random_scores(ds), "oracle": S.future_oracle_scores(ds)},
                                   n_boot=50)
        self.assertEqual([r["policy"] for r in table], ["random", "oracle"])
        self.assertAlmostEqual(table[0]["top1"], table[0]["top1_chance"])
        self.assertAlmostEqual(table[1]["top1"], 1.0)

    def test_heuristic_scores_need_the_column(self):
        sc = S.heuristic_scores(self.ds, "verifier_pass_rate")
        self.assertEqual(sc.shape, (len(self.ds),))
        with self.assertRaises(KeyError):
            S.heuristic_scores(self.ds.drop_groups("candidate_data_stats"), "verifier_pass_rate")

    def test_argmax_candidate_tie_breaking(self):
        self.assertEqual(S.argmax_candidate([0.1, 0.5, 0.5]), 1)
        rng = np.random.RandomState(0)
        picks = {S.argmax_candidate([0.5, 0.5, 0.1], rng) for _ in range(50)}
        self.assertEqual(picks, {0, 1})

    def test_gates(self):
        g1 = S.gate1(self.ds)
        self.assertIn("mismatch", g1)
        self.assertTrue(g1["passed"])
        rep = {"within_state_spearman": 0.35, "top1": 0.50, "top1_chance": 1 / 3}
        self.assertTrue(S.gate2(rep)["passed_point"])
        self.assertFalse(S.gate2({**rep, "top1": 0.40})["passed_point"])
        ci = {"rho": {"lo": 0.1}, "top1": {"lo": 0.30}}          # lower bound below chance
        self.assertFalse(S.gate2(rep, ci)["passed_ci"])
        self.assertTrue(S.gate2(rep, {"rho": {"lo": 0.1}, "top1": {"lo": 0.40}})["passed_ci"])
        self.assertTrue(S.pd_activation_gate(True, 0.7, -0.1, 0.2)["activate"])
        self.assertFalse(S.pd_activation_gate(False, 0.7, 0.3, 0.2)["activate"])
        self.assertFalse(S.pd_activation_gate(True, 0.5, -0.1, 0.2)["activate"])
        self.assertFalse(S.pd_activation_gate(True, 0.7, 0.3, 0.05)["activate"])
        direct = {"within_state_spearman": .30, "top1": .45, "ndcg": .80, "regret": .020}
        better = {"within_state_spearman": .40, "top1": .46, "ndcg": .80, "regret": .020}
        worse = {"within_state_spearman": .40, "top1": .30, "ndcg": .80, "regret": .020}
        self.assertTrue(S.dynamics_gate(direct, better)["promote"])
        self.assertFalse(S.dynamics_gate(direct, worse)["promote"])

    def test_tradeoff_fraction(self):
        ds = self.ds.subset(np.arange(len(self.ds)))
        ds.y_h2[:] = [0.01, 0.01, 0.01]
        self.assertEqual(S.tradeoff_fraction(ds), 0.0)
        ds.y_h2[::3] = [0.05, 0.0, -0.05]               # first candidate of every state
        self.assertEqual(S.tradeoff_fraction(ds), 1.0)


class PrimalDualTests(unittest.TestCase):
    def two_candidates(self):
        # A: utility 1.0 but retention -0.5 (g=+0.5); B: utility 0.8, retention +0.1 (g=-0.1)
        h2 = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.1]])
        return np.array([1.0, 0.8]), S.constraint_values(h2)

    def test_signed_constraint(self):
        _, g = self.two_candidates()
        np.testing.assert_allclose(g[:, 0], [0.5, -0.1])
        h2 = np.array([[0.1, -0.2, 0.0]])
        np.testing.assert_allclose(S.constraint_values(h2, True)[0], [0.0, 0.2])

    def test_inactive_constraint_equals_unconstrained(self):
        sel = S.PrimalDualSelector(S.PDConfig(eps_forget=10.0))
        u, g = self.two_candidates()
        for _ in range(20):
            self.assertEqual(sel.select(u, g), 0)
            sel.update(g[0])
        self.assertEqual(float(sel.lam[0]), 0.0)

    def test_dual_ascent_drives_average_forgetting_to_eps(self):
        cfg = S.PDConfig(eps_forget=0.0, eta=0.2)
        sel = S.PrimalDualSelector(cfg)
        u, g = self.two_candidates()
        chosen_g, lams = [], []
        for _ in range(600):
            k = sel.select(u, g)
            sel.update(g[k])
            chosen_g.append(g[k, 0])
            lams.append(sel.lam[0])
        self.assertLess(abs(np.mean(chosen_g[200:])), 0.03)        # long-run average g ~ eps
        self.assertGreater(max(lams), 0.1)                          # the dual actually moved
        self.assertLess(max(lams), cfg.lambda_max)
        self.assertGreater(np.mean([x == 0.5 for x in chosen_g[200:]]), 0.05)  # still uses A sometimes

    def test_lambda_is_clipped(self):
        sel = S.PrimalDualSelector(S.PDConfig(eta=100.0, lambda_max=3.0))
        sel.update(np.array([5.0]))
        self.assertEqual(float(sel.lam[0]), 3.0)
        sel.update(np.array([-50.0]))
        self.assertEqual(float(sel.lam[0]), 0.0)

    def test_replay_reduces_true_forgetting(self):
        ds = from_records(make_synthetic_records(
            LAYOUT, n_trajectories=30, states_per_trajectory=5, immediate_weight=0.0, seed=4), LAYOUT)
        h2 = ds.y_h2.astype(np.float64)                  # oracle forecasts isolate the selector logic
        scores = S.utility(h2)
        free = S.fixed_penalty_replay(ds, h2, scores, penalty=0.0)
        pd = S.pd_replay(ds, h2, scores, S.PDConfig(eps_forget=0.0, eta=0.5))
        self.assertLessEqual(pd["mean_true_forgetting"], free["mean_true_forgetting"] + 1e-9)
        self.assertLessEqual(pd["mean_chosen_lhv2"], free["mean_chosen_lhv2"] + 1e-9)


class DecisionUpdateTests(unittest.TestCase):
    """Behaviour agreed in Shashank's sprint-2 decisions (D04, D05, D02, D10, D13, D15)."""

    def test_d04_weights_and_worked_example(self):
        self.assertEqual(DEFAULT_WEIGHTS, (0.50, 0.25, 0.25))
        parent, h2 = np.array([0.60, 0.50, 0.80]), np.array([0.64, 0.48, 0.79])
        self.assertAlmostEqual(float(utility(h2 - parent)), 0.0125)

    def test_d05_module_to_block_energy_with_numeric_order(self):
        e = F.module_energy(np.array([3.0]), np.array([4.0]))
        self.assertAlmostEqual(e, 25.0)
        mods = {(i, m): 0.0 for i in range(11) for m in ("q_proj", "v_proj")}
        mods[(2, "q_proj")], mods[(2, "v_proj")], mods[(10, "q_proj")] = 1.0, 2.0, 5.0
        blocks = F.block_energies(mods)
        self.assertEqual(blocks.shape, (11,))
        self.assertEqual((blocks[2], blocks[10]), (3.0, 5.0))             # q+v summed, 2 before 10
        frac = F.energy_fractions_from_blocks(blocks, 6)
        self.assertAlmostEqual(frac.sum(), 1.0)
        with self.assertRaises(ValueError):
            F.block_energies({(0, "q"): 1.0, (2, "q"): 1.0})              # missing block 1

    def test_d10_history_rules(self):
        self.assertEqual([expected_history_count(t, 3, "applied_h1") for t in range(5)], [0, 1, 2, 3, 3])
        self.assertEqual([expected_history_count(t, 3, "applied_h2") for t in range(5)], [0, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            expected_history_count(1, 3, "nope")
        h1 = make_synthetic_records(LAYOUT, n_trajectories=3, seed=1, history_rule="applied_h1")
        h2 = make_synthetic_records(LAYOUT, n_trajectories=3, seed=1, history_rule="applied_h2")
        validate_records(h1, LAYOUT, history_rule="applied_h1")
        validate_records(h2, LAYOUT, history_rule="applied_h2")
        with self.assertRaises(RecordError):                               # wrong rule is caught
            validate_records(h2, LAYOUT, history_rule="applied_h1")

    def test_d02_group_ids_fixed_split_and_validation(self):
        recs = make_synthetic_records(LAYOUT, n_trajectories=12, states_per_trajectory=3, n_groups=6, seed=2)
        ds = from_records(recs, LAYOUT)
        self.assertEqual(len(np.unique(ds.group)), 6)
        names = sorted(np.unique(ds.group).tolist())
        manifest = {g: ("test" if i < 2 else "val" if i < 3 else "train") for i, g in enumerate(names)}
        idx = fixed_split(ds, manifest)
        for a in idx:
            for b in idx:
                if a != b:
                    self.assertFalse(set(ds.group[idx[a]]) & set(ds.group[idx[b]]))
        self.assertEqual(sum(len(v) for v in idx.values()), len(ds))
        with self.assertRaises(ValueError):
            fixed_split(ds, {**manifest, "extra-group": "train"})
        folds = grouped_kfold(ds, 3, seed=0, by="group")
        for tr, te in folds:
            self.assertFalse(set(ds.group[tr]) & set(ds.group[te]))
        bad = copy.deepcopy(recs)
        for r in bad:
            if r["trajectory_id"].endswith("chain-000") and r["state_id"] == 1:
                r["group_id"] = "somewhere-else"
        with self.assertRaises(RecordError):
            validate_records(bad, LAYOUT)

    def test_d13_constant_baseline_and_group_bootstrap(self):
        y = np.array([[1.0, 0, 0]] * 6 + [[3.0, 0, 0]] * 2)
        g = np.array(["a"] * 6 + ["b"] * 2)
        self.assertAlmostEqual(float(M.group_weighted_mean(y, g)[0]), 2.0)   # not the row mean 1.5
        rng = np.random.RandomState(0)
        sid = np.repeat(np.arange(40), 3)
        grp = np.repeat(np.arange(8), 15)
        t = rng.normal(size=120)
        p = t + rng.normal(0, 0.5, 120)
        few = M.bootstrap_ci(M.m_within_state_spearman, t, p, sid, n_boot=100, cluster_index=np.repeat([0, 1], 60))
        self.assertIn("warning", few)                                        # 2 units: unreliable
        many = M.bootstrap_ci(M.m_within_state_spearman, t, p, sid, n_boot=100,
                              cluster_index=np.repeat(np.arange(20), 6))
        self.assertNotIn("warning", many)
        self.assertLess(many["lo"], many["point"])
        rep = M.per_group_report(t, p, sid, grp)
        self.assertEqual(len(rep), 8)
        names = np.array(["g%02d" % i for i in range(20)])[np.repeat(np.arange(20), 6)]   # string ids
        as_str = M.bootstrap_ci(M.m_within_state_spearman, t, p, sid, n_boot=50, cluster_index=names)
        self.assertEqual(as_str["n_units"], 20)
        self.assertEqual(len(M.per_group_report(t, p, sid, np.array(["a", "b"])[np.repeat([0, 1], 60)])), 2)

    def test_d15_semantic_hash_changes_with_definitions(self):
        base = SemanticConfig()
        self.assertEqual(base.fingerprint(), SemanticConfig().fingerprint())
        self.assertNotEqual(base.fingerprint(), SemanticConfig(history_rule="applied_h2").fingerprint())
        self.assertNotEqual(base.fingerprint(), SemanticConfig(lhv_weights=(1 / 3, 1 / 3, 1 / 3)).fingerprint())
        h = base.fingerprint()
        recs = make_synthetic_records(LAYOUT, n_trajectories=2, seed=1, semantic_hash=h)
        self.assertEqual(from_records(recs, LAYOUT, expected_semantic_hash=h).semantic_hash, h)
        with self.assertRaises(RecordError):
            validate_records(recs, LAYOUT, expected_semantic_hash="different")


if __name__ == "__main__":
    unittest.main()
