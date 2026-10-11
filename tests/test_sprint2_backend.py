"""SYNTHETIC tiny CPU language-model plumbing; no Qwen/data download or GPU."""
import contextlib
import copy
import tempfile
import types
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.third_eye.feature_contract import ContractError
from src.third_eye.sprint2_backend import HFCollectionBackend
from src.third_eye.sprint2_protocol import build_continuation_plan
from src.third_eye.sprint2_runtime import TrainingSnapshot
from src.third_eye.sprint2_collection import validate_source_batch
from src.third_eye.sprint2_protocol import build_pilot_state_plan


class TinySyntheticLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 4)
        self.embedding.weight.requires_grad_(False)
        self.projection = torch.nn.Linear(4, 9, bias=False)
        self.projection.weight.requires_grad_(False)
        self.a = torch.nn.Parameter(torch.full((4, 2), .01))
        self.b = torch.nn.Parameter(torch.zeros(2, 9))
        self.adapter_active = True
        self.config = types.SimpleNamespace(max_position_embeddings=512)
        self.generation_config = types.SimpleNamespace(eos_token_id=8)
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        hidden = self.embedding(input_ids)
        logits = self.projection(hidden)
        if self.adapter_active:
            logits = logits + hidden @ self.a @ self.b
        return types.SimpleNamespace(logits=logits, hidden_states=(hidden,))
    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_active = False
        try:
            yield
        finally:
            self.adapter_active = True
    def generate(self, input_ids, **kwargs):
        # Fixed fabricated emitted text; ONLY a tokenizer/collector unit fixture.
        answer = 5 if input_ids.shape[1] > 25 else 6
        suffix = torch.tensor([[2, 3, answer, 8]])
        return torch.cat([input_ids, suffix], dim=1)


def synthetic_backend():
    raw = Tokenizer(models.WordLevel({"<unk>": 0, "<pad>": 1, "Final": 2, "answer:": 3,
                                     "x": 4, "2": 5, "3": 6, "VALUE": 7, "<eos>": 8}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}\n"
    b = HFCollectionBackend.__new__(HFCollectionBackend)
    b.tokenizer, b.model, b.device = tokenizer, TinySyntheticLM(), torch.device("cpu")
    b.deadline, b.output_path, b.state_output_path = None, None, None
    b.spec = {"semantic_hash": "0"*64, "synthetic_only": True}
    b.config = {"task_rules": {"SYNTHETIC": "final_numeric_line_v1"}, "generation_caps": {"SYNTHETIC": 16},
        "update": {"steps": 3, "learning_rate": .01, "weight_decay": 0, "max_length": 128,
                   "betas": [.9, .999], "eps": 1e-8},
        "module_manifest": [{"a_name": "a", "a_shape": [4, 2], "b_name": "b", "b_shape": [2, 9]}],
        "definitions": {"nll_reduction": "equal_answer_v1", "duplicate_threshold": .9,
                        "duplicate_tolerance": 0, "cosine_zero_epsilon": 0},
        "probe": {"steps": 3, "cadence": [0, 1, 3], "optimizer": {"lr": .01, "weight_decay": 0}},
        "candidate_recipes": [{"seed": 1, "prompt_count": 1}, {"seed": 2, "prompt_count": 1}, {"seed": 3, "prompt_count": 1}]}
    b.config["continuation"] = {"recipe": {"SYNTHETIC": True}, "candidate_master_seed": 1, "master_seed": 2}
    b.populations = {role: {"data": {"task_id": "SYNTHETIC", "prompts": [{"prompt_id": "fake-"+role,
        "prompt": "x", "reference": "2", "source_family": "fake-family-"+role}]}}
        for role in ("target", "ood", "retention", "confidence", "retention_gradient", "anchor_probe", "val_probe", "adaptation:g")}
    for role, population in b.populations.items():
        population["manifest_sha256"] = "0"*64
    row = b.populations["adaptation:g"]["data"]["prompts"][0]
    b.populations["adaptation:g"]["data"]["prompts"] = [{**row, "prompt_id": "fake-source-%d" % i} for i in range(128)]
    return b


class BackendTests(unittest.TestCase):
    def test_parent_scoring_actual_token_alignment_and_features(self):
        b = synthetic_backend()
        accuracy, features, proof = b.measure_parent()
        self.assertEqual(accuracy, {"target": 0, "ood": 0, "retention": 0})
        self.assertEqual(proof["scoring"]["target"]["total"], 1)
        self.assertLess(features["gen_confidence_mean"][0], 0)
        control = TrainingSnapshot(b.model)
        features, evidence, batch, zero = b.prepare_candidate("g", 0, 0)
        self.assertGreater(features["candidate_data_stats"]["pre_update_nll"][0], 0)
        self.assertEqual(features["gradient_interference_stats"]["layerwise_grad_norms"].shape, (1,))
        self.assertTrue(evidence["probe"]["restoration_verified"])
        plan = build_pilot_state_plan(group_id="g", state=0, prompt_pool=[r["prompt_id"] for r in b._rows("adaptation:g")],
            recipe=b.config["continuation"]["recipe"], candidate_master_seed=1, continuation_master_seed=2).candidates[0]
        bindings = {"task_rules": b.config["task_rules"], "generation_calibration": {"caps": b.config["generation_caps"]}}
        validate_source_batch(evidence["attempts"], evidence["corrections"], plan, b.parent_sha(), bindings)
        for mutation in ("wrong_prompt", "wrong_seed", "wrong_parent", "unverified_reference"):
            bad = copy.deepcopy(evidence)
            if mutation == "wrong_prompt":
                bad["attempts"]["original"][0]["prompt_id"] = "not-the-sealed-source"
            elif mutation == "wrong_seed":
                bad["attempts"]["original"][0]["seed"] += 1
            elif mutation == "wrong_parent":
                bad["attempts"]["corrections"][0]["parent_sha256"] = "f"*64
            else:
                bad["corrections"][plan.prompt_ids[0]] = "an invented training answer"
            with self.assertRaises(ContractError):
                validate_source_batch(bad["attempts"], bad["corrections"], plan, b.parent_sha(), bindings)
        self.assertEqual(control.fingerprint(), control.before)

    def test_h1_h2_measurement_shared_failure_batch_and_restore(self):
        b = synthetic_backend()
        _, _, batch, _ = b.prepare_candidate("g", 0, 0)
        prompt = b._rows("adaptation:g")[0]["prompt_id"]
        plan = build_continuation_plan(group_id="g", state=0, recipe={"SYNTHETIC": True},
            prompt_pool=[prompt], prompt_count=1, master_seed=2)
        control = TrainingSnapshot(b.model)
        with tempfile.TemporaryDirectory() as tmp:
            h1, h2, corrected, snapshot, runtime = b.run_sibling(batch, plan, 0, Path(tmp)/"fake")
            self.assertEqual(h1["parent_sha256"], b.parent_sha())
            self.assertEqual(h2["parent_sha256"], h1["endpoint_sha256"])
            self.assertNotEqual(h1["parent_sha256"], h1["endpoint_sha256"])
            self.assertEqual(corrected, {prompt: "2"})
            self.assertTrue(runtime["restoration_verified"])
            self.assertTrue(snapshot.exists())
        self.assertEqual(control.fingerprint(), control.before)

    def test_empty_failure_batch_and_training_span_truncation_fail(self):
        b = synthetic_backend()
        row = b._rows("adaptation:g")[0]
        b.config["update"]["max_length"] = 1
        with self.assertRaises(ContractError):
            b._prepare(row)
        b.config["update"]["max_length"] = 128
        b.populations["adaptation:g"]["data"]["prompts"][0]["reference"] = "3"
        plan = build_continuation_plan(group_id="g", state=0, recipe={"SYNTHETIC": True},
            prompt_pool=[row["prompt_id"]], prompt_count=1, master_seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            b.state_output_path = Path(tmp)
            with self.assertRaises(ContractError):
                b._failure_batch("g", plan)
            self.assertEqual(len((Path(tmp)/"generation_attempts.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
