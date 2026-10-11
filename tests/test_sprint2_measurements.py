"""SYNTHETIC tensor/scorer fixtures, not extracted research features."""
import copy
import random
import unittest

import numpy as np
import torch

from src.third_eye.feature_contract import ContractError
from src.third_eye.sprint2_measurements import (answer_logprobs, answer_nll, candidate_data,
    duplicate_components, gradient_measurements, probe_measurements, score_completions)
from src.third_eye.sprint2_runtime import TrainingSnapshot, restored_training_state


class TinyLora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([3., 4.]))
        self.b = torch.nn.Parameter(torch.tensor([0.]))
        self.register_buffer("counter", torch.tensor(0.), persistent=False)
        self.child = torch.nn.Dropout(.2)


MANIFEST = [{"a_name": "a", "a_shape": [2], "b_name": "b", "b_shape": [1]}]


class Stateful:
    def __init__(self):
        self.position = 4
    def state_dict(self):
        return {"position": self.position}
    def load_state_dict(self, value):
        self.position = value["position"]


class RuntimeTests(unittest.TestCase):
    def test_restore_all_owned_state_success_and_exception(self):
        for failure in (False, True):
            model = TinyLora()
            model.child.eval()
            opt = torch.optim.AdamW(model.parameters(), lr=.1)
            (model.a.sum()+model.b.sum()).backward()
            opt.step()
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 1)
            sampler = Stateful()
            scaler = torch.amp.GradScaler("cpu")
            generator = torch.Generator().manual_seed(2)
            owned = (opt, scheduler, sampler, scaler)
            control = TrainingSnapshot(model, stateful=owned, generators=(generator,))
            with self.assertRaises(RuntimeError) if failure else self.subTest(failure=failure):
                with restored_training_state(model, stateful=owned, generators=(generator,)):
                    model.eval()
                    model.a.grad = None
                    model.b.requires_grad_(False)
                    with torch.no_grad():
                        model.a.add_(3)
                        model.counter.add_(1)
                    model.counter = torch.tensor(123.)
                    sampler.position += 8
                    opt.step()
                    scheduler.step()
                    random.random()
                    np.random.rand()
                    torch.rand(2)
                    torch.rand(2, generator=generator)
                    if failure:
                        raise RuntimeError("synthetic injected failure")
            self.assertEqual(control.fingerprint(), control.before)

    def test_restored_next_optimizer_step_matches_control(self):
        a = TinyLora()
        b = copy.deepcopy(a)
        opt_a = torch.optim.AdamW(a.parameters(), lr=.01)
        opt_b = torch.optim.AdamW(b.parameters(), lr=.01)
        with restored_training_state(a, stateful=(opt_a,)):
            (a.a.square().sum()+a.b.sum()).backward()
            opt_a.step()
        for model, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad()
            (model.a.sum()+model.b.sum()).backward()
            opt.step()
        self.assertTrue(torch.equal(a.a, b.a))
        self.assertTrue(torch.equal(a.b, b.b))


class MeasurementTests(unittest.TestCase):
    def test_causal_masks_and_unequal_answer_reductions(self):
        logits = torch.tensor([[[0., 1.], [1., 0.], [1., 0.]],
                               [[0., 1.], [1., 0.], [1., 0.]]], requires_grad=True)
        ids = torch.tensor([[0, 1, 0], [0, 0, 1]])
        mask = torch.tensor([[False, True, False], [False, True, True]])
        values = answer_logprobs(logits, ids, mask, torch.ones_like(ids))
        self.assertEqual([len(p) for p in values], [1, 2])
        self.assertAlmostEqual(float(values[0][0].detach()), -.31326166, places=6)
        self.assertNotEqual(float(answer_nll(values, reduction="equal_answer_v1").detach()),
                            float(answer_nll(values, reduction="token_weighted_v1").detach()))
        bad = mask.clone()
        bad[0, 0] = True
        with self.assertRaises(ContractError):
            answer_logprobs(logits, ids, bad, torch.ones_like(ids))
        with self.assertRaises(ContractError):
            answer_logprobs(logits, ids, torch.zeros_like(mask), torch.ones_like(ids))

    def test_scorer_denominator_parse_and_truncation(self):
        rows = [{"prompt_id": str(i), "text": text, "reference": "2", "at_eos": False,
                 "generated_tokens": 5, "cap": 5, "truncated": truncated}
                for i, (text, truncated) in enumerate([
                    ("Final answer: 2\n", False), ("Final answer: 3\n", False),
                    ("unparsed", False), ("Final answer: ", True)])]
        report = score_completions(rows, "final_numeric_line_v1")
        self.assertEqual((report["total"], report["correct"], report["parse_failures"], report["truncated"]), (4, 1, 2, 1))
        self.assertEqual(report["accuracy"], .25)
        with self.assertRaises(ContractError):
            score_completions([], "final_numeric_line_v1")

    def test_gradient_alignment_raw_norms_and_zero_diagnostics(self):
        model = TinyLora()
        before = TrainingSnapshot(model)
        stats, diag = gradient_measurements(model, MANIFEST,
            lambda: (model.a*torch.tensor([3., 4.])).sum()+model.b.sum()*0,
            lambda: (model.a*torch.tensor([3., -4.])).sum()+model.b.sum()*0, zero_epsilon=0)
        self.assertEqual(stats["grad_norm"], 5)
        self.assertEqual(stats["layerwise_grad_norms"], [5])
        self.assertAlmostEqual(stats["grad_cosine_retention"], -.28)
        self.assertAlmostEqual(stats["sign_agreement"], 2/3)
        self.assertEqual(diag["zero_in_both"], 1/3)
        self.assertEqual(before.fingerprint(), before.before)
        with self.assertRaises(ContractError):
            gradient_measurements(model, MANIFEST, lambda: model.a.sum(), lambda: model.a.sum(), zero_epsilon=0)

    def test_duplicates_exact_threshold_and_order(self):
        result = duplicate_components(["A A", "A  A", "B B"], threshold=.9, tolerance=0)
        self.assertEqual(result["rate"], 1/3)
        self.assertEqual(result["components"], [[0, 1], [2]])
        self.assertEqual(duplicate_components(["B B", "A A", "A  A"], threshold=.9, tolerance=0)["rate"], 1/3)

    def test_candidate_measurements_empty_nonfinite_and_attempt_denominator(self):
        args = dict(logprobs=[torch.tensor([-1., -3.]), torch.tensor([-4.])], lengths=[2, 4],
            correction_passes=[True, False, True, False], original_passes=[False, True],
            embeddings=[[1., 0.], [0., 1.]], texts=["A", "B"],
            definitions={"nll_reduction": "equal_answer_v1", "duplicate_threshold": .9, "duplicate_tolerance": 0})
        values, _ = candidate_data(**args)
        self.assertEqual(values["pre_update_nll"], 3)
        self.assertEqual(values["mean_confidence"], -3)
        self.assertEqual(values["verifier_pass_rate"], .5)
        self.assertEqual(values["embedding_diversity"], 1)
        for change in ({"embeddings": [[0., 0.], [1., 0.]]}, {"original_passes": []},
                       {"logprobs": [torch.tensor([float("nan")]), torch.tensor([1.])]}):
            with self.assertRaises(ContractError):
                candidate_data(**{**args, **change})

    def test_actual_tiny_probe_measurements_and_exception_restore(self):
        model = TinyLora()
        control = TrainingSnapshot(model)
        def loss():
            return model.a.square().mean()+model.b.square().mean()
        args = dict(model=model, manifest=MANIFEST, train_loss=lambda step: loss(),
            anchor_loss=loss, val_loss=loss, anchor_logits=lambda: torch.stack([model.a, model.a]),
            steps=3, cadence=[0, 1, 3], optimizer_config={"lr": .01, "weight_decay": 0}, seed=5)
        values, proof = probe_measurements(**args)
        self.assertGreater(values["lora_delta_norm"], 0)
        self.assertTrue(proof["restoration_verified"])
        self.assertEqual(control.fingerprint(), control.before)
        def failing(step):
            if step == 2:
                raise RuntimeError("synthetic probe failure")
            return loss()
        with self.assertRaises(RuntimeError):
            probe_measurements(**{**args, "train_loss": failing})
        self.assertEqual(control.fingerprint(), control.before)


if __name__ == "__main__":
    unittest.main()
