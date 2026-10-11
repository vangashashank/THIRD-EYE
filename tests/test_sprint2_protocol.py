"""Synthetic fixtures only: no extracted features, outcomes, models or training."""
from dataclasses import FrozenInstanceError, replace
import random
import unittest

import numpy as np

from src.third_eye.feature_contract import ContractError
from src.third_eye.sprint2_protocol import (
    CandidateKey, LineageRecipe, LoraModule, assign_lineage_splits,
    bind_child_failure_batch, build_continuation_plan, canonical_lora_manifest,
    expected_selection_regret, fit_training_statistics, group_macro_weights,
    lineage_components, prediction_errors, prompt_processing_seed, verify_lineage_splits,
    build_pilot_state_plan,
)


def synthetic_keys(group="synthetic-train", trajectory="synthetic-path", states=1):
    return tuple(CandidateKey(group, trajectory, state, candidate)
                 for state in range(states) for candidate in range(3))


class LineageTests(unittest.TestCase):
    def test_transitive_roots_prefixes_and_source_families(self):
        rows = [LineageRecipe("A", "root-A", ("shared",), ("family-A",)),
                LineageRecipe("B", "root-B", ("shared",), ("family-BC",)),
                LineageRecipe("C", "root-C", (), ("family-BC",)),
                LineageRecipe("D", "root-C", (), ())]
        self.assertEqual(lineage_components(rows), (("A", "B", "C", "D"),))

    def test_outcome_blind_8_2_2_assignment_is_order_invariant(self):
        rows = [LineageRecipe(f"synthetic-{i}", f"root-{i}", (), (f"family-{i}",)) for i in range(12)]
        membership, splits = assign_lineage_splits(rows)
        self.assertEqual([list(splits.values()).count(name) for name in ("train", "validation", "test")], [8, 2, 2])
        self.assertEqual(assign_lineage_splits(rows[::-1]), (membership, splits))
        verify_lineage_splits(rows, membership, splits)
        with self.assertRaises(TypeError):
            splits["fake"] = "test"
        with self.assertRaisesRegex(ContractError, "12 independent"):
            assign_lineage_splits(rows[:11])

    def test_cross_split_prefix_and_missing_metadata_fail(self):
        rows = [LineageRecipe("A", "root", (), ()), LineageRecipe("B", "root", (), ())]
        with self.assertRaises(ContractError):
            verify_lineage_splits(rows, {"A": "a", "B": "b"}, {"a": "train", "b": "test"})
        with self.assertRaises(ContractError):
            verify_lineage_splits(rows, {"A": "a"}, {"a": "train"})
        with self.assertRaises(ContractError):
            lineage_components([rows[0], rows[0]])

    def test_explicit_module_count_and_numeric_order(self):
        modules = [LoraModule(block, projection, f"synthetic.block.{block}.{projection}")
                   for block in (10, 2) for projection in ("v_proj", "q_proj")]
        count, order = canonical_lora_manifest(modules, placement_order=("q_proj", "v_proj"))
        self.assertEqual(count, 4)
        self.assertEqual(order, ("synthetic.block.2.q_proj", "synthetic.block.2.v_proj",
                                 "synthetic.block.10.q_proj", "synthetic.block.10.v_proj"))
        for bad in ([modules[0], modules[0]], [replace(modules[0], projection="unknown")], ["guessed.module"]):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                canonical_lora_manifest(bad, placement_order=("q_proj", "v_proj"))


class ContinuationTests(unittest.TestCase):
    def test_pilot_candidate_keyed_draws_shared_h2_exclusion_and_no_redraw(self):
        args = dict(group_id="SYNTHETIC", state=0, prompt_pool=["fake-%d" % i for i in range(128)],
                    candidate_master_seed=4, continuation_master_seed=4, recipe={"SYNTHETIC": True})
        plan = build_pilot_state_plan(**args)
        self.assertEqual([len(set(c.prompt_ids)) for c in plan.candidates], [32, 32, 32])
        self.assertEqual(len({c.generation_seed for c in plan.candidates}), 3)
        self.assertEqual(len(plan.continuation.prompt_ids), 32)
        self.assertFalse(set(plan.continuation.prompt_ids) & set(plan.excluded_h1_prompt_ids))
        self.assertEqual(build_pilot_state_plan(**{**args, "prompt_pool": args["prompt_pool"][::-1]}), plan)
        for _ in range(20):
            random.random()
        self.assertEqual(build_pilot_state_plan(**args), plan)
        self.assertNotEqual(build_pilot_state_plan(**{**args, "state": 1}), plan)
        with self.assertRaisesRegex(ContractError, "never redraw"):
            build_pilot_state_plan(**{**args, "prompt_pool": args["prompt_pool"][:32]})
    def plan(self, **changes):
        args = dict(group_id="synthetic-group", state=2, recipe={"policy": "synthetic-fixed-v1"},
                    prompt_pool=[f"synthetic-prompt-{i}" for i in range(20)], prompt_count=5, master_seed=17)
        return build_continuation_plan(**{**args, **changes})

    def test_one_shared_immutable_plan_with_child_specific_batches(self):
        plan = self.plan()
        batches = [bind_child_failure_batch(plan, candidate_id=0, corrected_failures={plan.prompt_ids[0]: "A"}),
                   bind_child_failure_batch(plan, candidate_id=1, corrected_failures={plan.prompt_ids[-1]: "B"}),
                   bind_child_failure_batch(plan, candidate_id=2, corrected_failures={})]
        self.assertEqual({row.plan_sha256 for row in batches}, {plan.sha256})
        self.assertEqual({row.training_seed for row in batches}, {plan.training_seed})
        self.assertNotEqual(batches[0].examples, batches[1].examples)
        self.assertEqual(batches[2].examples, ())
        with self.assertRaises(FrozenInstanceError):
            plan.state = 3

    def test_variable_child_processing_and_outcomes_do_not_change_draw(self):
        before = self.plan()
        rng_state = random.getstate()
        try:
            for synthetic_outcome in (0.0, -0.2, 1.0):
                for _ in range(int((synthetic_outcome + 1) * 100)):
                    random.random()
                self.assertEqual(self.plan(), before)
        finally:
            random.setstate(rng_state)
        forward = {prompt: prompt_processing_seed(before, prompt, stream="generation") for prompt in before.prompt_ids}
        reverse = {prompt: prompt_processing_seed(before, prompt, stream="generation") for prompt in reversed(before.prompt_ids)}
        self.assertEqual(forward, reverse)
        self.assertNotEqual(before.generation_seed, before.prompt_seed)
        self.assertNotEqual(before.training_seed, before.correction_seed)
        self.assertNotEqual(self.plan(state=3).sha256, before.sha256)
        self.assertNotEqual(self.plan(group_id="other-group").sha256, before.sha256)
        self.assertEqual(self.plan(prompt_pool=list(reversed([f"synthetic-prompt-{i}" for i in range(20)]))), before)

    def test_invalid_plans_unknown_batches_and_unapproved_draw_count_fail(self):
        for changes in ({"state": 5}, {"recipe": {"bad": float("nan")}}, {"prompt_count": 21},
                        {"prompt_pool": ["duplicate", "duplicate"]}, {"prompt_pool": "not-a-pool"},
                        {"master_seed": True}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.plan(**changes)
        with self.assertRaises(TypeError):
            self.plan(draws=2)
        with self.assertRaises(TypeError):
            self.plan(outcomes=[0, 1, 2])
        for candidate, failures in ((3, {}), (0, {"outside-draw": "A"})):
            with self.assertRaises(ContractError):
                bind_child_failure_batch(self.plan(), candidate_id=candidate, corrected_failures=failures)


class EvaluationTests(unittest.TestCase):
    def test_group_macro_weights_do_not_count_rows_as_independent(self):
        keys = synthetic_keys("A", "path-A", 5) + synthetic_keys("B", "path-B", 1)
        weights = group_macro_weights(keys)
        self.assertAlmostEqual(weights[:15].sum(), .5)
        self.assertAlmostEqual(weights[15:].sum(), .5)
        for bad in (keys[:-1], keys + (keys[0],), ("row",)):
            with self.assertRaises(ContractError):
                group_macro_weights(bad)

    def test_train_only_mean_constant_and_masked_normalization(self):
        keys = synthetic_keys()
        values = np.array([[0, 2, 0], [0, 4, 0], [0, 6, 0]], float)
        observed = np.array([[True, True, False]] * 3)
        fitted = fit_training_statistics(values, keys, {"synthetic-train": "train"}, observed=observed)
        np.testing.assert_equal(fitted.mean, [0, 4, 0])
        np.testing.assert_equal(fitted.scale[[0, 2]], [1, 1])
        np.testing.assert_equal(fitted.observed_weight, [1, 1, 0])
        normalized = fitted.transform(values, observed=observed)
        np.testing.assert_equal(normalized[:, [0, 2]], 0)
        self.assertFalse(fitted.mean.flags.writeable)
        # The fitted native-unit mean is also the constant-output baseline.
        for heldout in ("validation", "test", "unknown"):
            with self.assertRaisesRegex(ContractError, "training groups only"):
                fit_training_statistics(values + 1, keys, {"synthetic-train": heldout})
        np.testing.assert_equal(fitted.mean, [0, 4, 0])

    def test_invalid_values_and_masks_fail(self):
        keys = synthetic_keys()
        for values, observed in (([[0], [float("nan")], [0]], None),
                                 ([[1], [0], [0]], np.zeros((3, 1), bool)),
                                 ([[0], [0], [0]], np.zeros((3, 1), float)),
                                 ([[0], [0]], None)):
            with self.subTest(values=values), self.assertRaises(ContractError):
                fit_training_statistics(values, keys, {"synthetic-train": "train"}, observed=observed)

    def test_native_prediction_error_and_explicit_expected_ties(self):
        errors = prediction_errors(np.zeros((3, 2)), [[-1, 0], [0, 0], [1, 0]], synthetic_keys())
        np.testing.assert_allclose(errors["mae"], [2/3, 0])
        np.testing.assert_allclose(errors["mse"], [2/3, 0])
        np.testing.assert_allclose(errors["rmse"], [np.sqrt(2/3), 0])
        random_regret = expected_selection_regret([.03, .01, 0])
        constant_regret = expected_selection_regret([.03, .01, 0], [.1, .1, .1])
        self.assertAlmostEqual(random_regret["regret"], .03-.04/3)
        self.assertEqual(random_regret["tie_size"], 3)
        np.testing.assert_allclose(constant_regret["selection_probabilities"], [1/3]*3)
        self.assertEqual(expected_selection_regret([.03, .01, 0], [3, 2, 1])["regret"], 0)
        self.assertAlmostEqual(expected_selection_regret([.03, .01, 0], [2, 2, 1])["regret"], .01)
        with self.assertRaises(ContractError):
            expected_selection_regret([0, 1])
        with self.assertRaisesRegex(ContractError, "Nonfinite prediction error"):
            prediction_errors([[1e308]] * 3, [[-1e308]] * 3, synthetic_keys())


if __name__ == "__main__":
    unittest.main()
