"""Offline unit tests. All training fixtures here are synthetic, not research data."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from scripts.fit_direct_forecaster import fit_forecaster
from src.models import model_loader, lora_model
from src.third_eye.checkpoint_manager import CheckpointManager
from src.third_eye.direct_forecaster import ThirdEyeDirect
from src.third_eye.forecaster_checkpoint import save_forecaster, load_forecaster
from src.third_eye.history_store import HistoryStore
from src.third_eye.run_workspace import create_trajectory_workspace
from src.training.checkpoint import adapter_state_digest, save_adapter
from src.training.trainer import train_lora


class SprintOneTests(unittest.TestCase):
    def test_training_outcomes_and_finite_guard(self):
        class SyntheticModel(torch.nn.Module):
            def __init__(self, finite=True):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.finite = finite

            def forward(self, **batch):
                loss = self.weight.square() if self.finite else self.weight * float("nan")
                return SimpleNamespace(loss=loss)

        data = [dict(input_ids=torch.ones(2, dtype=torch.long),
                     attention_mask=torch.ones(2, dtype=torch.long),
                     labels=torch.ones(2, dtype=torch.long))] * 2
        model = SyntheticModel()
        train_lora(model, data, torch.device("cpu"))
        self.assertEqual(model.last_training_metrics["optimizer_steps"], 2)
        self.assertEqual(len(model.last_training_metrics["step_losses"]), 2)
        self.assertIsNone(model.last_training_metrics["peak_gpu_allocated_bytes"])
        with self.assertRaises(ValueError):
            train_lora(SyntheticModel(False), data, torch.device("cpu"))
        with self.assertRaises(ValueError):
            train_lora(model, [], torch.device("cpu"))

    def test_quantized_placement_and_default_lora(self):
        quantized = SimpleNamespace(is_loaded_in_4bit=True, to=MagicMock())
        self.assertIs(model_loader.place_model(quantized, torch.device("cuda")), quantized)
        quantized.to.assert_not_called()
        ordinary = SimpleNamespace(to=MagicMock(return_value="placed"))
        self.assertEqual(model_loader.place_model(ordinary, torch.device("cpu")), "placed")
        with patch.object(lora_model, "get_peft_model", return_value="adapter") as attach:
            self.assertEqual(lora_model.attach_lora(ordinary), "adapter")
            config = attach.call_args.args[1]
            self.assertEqual(config.r, 8)
            self.assertEqual(config.lora_alpha, 16)
            self.assertEqual(config.target_modules, {"q_proj", "v_proj"})

    def test_nf4_loader_options(self):
        model = SimpleNamespace(is_loaded_in_4bit=True, to=MagicMock())
        tokenizer = SimpleNamespace(pad_token_id=1)
        with patch.object(model_loader, "get_device", return_value=torch.device("cuda")), \
                patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
                patch.object(torch.cuda, "current_device", return_value=0), \
                patch.object(model_loader, "BitsAndBytesConfig") as quant_config, \
                patch.object(model_loader.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
                patch.object(model_loader.AutoModelForCausalLM, "from_pretrained", return_value=model) as load:
            model_loader.load_model_and_tokenizer("fixture", quantization="nf4", local_files_only=True)
            self.assertEqual(load.call_args.kwargs["device_map"], {"": 0})
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            self.assertEqual(quant_config.call_args.kwargs["bnb_4bit_quant_type"], "nf4")
            self.assertEqual(quant_config.call_args.kwargs["bnb_4bit_compute_dtype"], torch.bfloat16)
            model.to.assert_not_called()
        with patch.object(model_loader, "get_device", return_value=torch.device("cpu")):
            with self.assertRaises(RuntimeError):
                model_loader.load_model_and_tokenizer("fixture", quantization="nf4")
        with self.assertRaises(ValueError):
            model_loader.load_model_and_tokenizer("fixture", quantization="unknown")

    def test_kbit_preparation(self):
        base = SimpleNamespace(is_loaded_in_4bit=True, config=SimpleNamespace(use_cache=True))
        with patch.object(lora_model, "prepare_model_for_kbit_training", return_value=base) as prepare:
            lora_model.prepare_adapter_base(base, is_trainable=False)
            self.assertFalse(prepare.call_args.kwargs["use_gradient_checkpointing"])
            self.assertFalse(base.config.use_cache)

    def test_checkpoint_copies_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "adapter"
            source.mkdir()
            (source / "fixture.txt").write_text("parent")
            manager = CheckpointManager(root / "checkpoints")
            parent = manager.save_parent(source, 0)
            manager.save_candidate(source, 0, 1)
            promoted = manager.promote_candidate(0, 1, 1)
            self.assertEqual((promoted / "fixture.txt").read_text(), "parent")
            self.assertEqual(manager.rollback_to_parent(0), parent)
            with self.assertRaises(FileExistsError):
                manager.save_parent(source, 0)
            with self.assertRaises(FileExistsError):
                manager.promote_candidate(0, 1, 1)
            with self.assertRaises(FileExistsError):
                save_adapter(MagicMock(), MagicMock(), source)
            self.assertEqual((parent / "fixture.txt").read_text(), "parent")

    def test_adapter_digest_includes_values(self):
        a = {"x": torch.tensor([1., 2.], dtype=torch.bfloat16)}
        self.assertEqual(adapter_state_digest(a), adapter_state_digest(a))
        self.assertNotEqual(adapter_state_digest(a), adapter_state_digest({"x": a["x"] + 1}))

    def test_forecaster_roundtrip_and_schema(self):
        with tempfile.TemporaryDirectory() as root:
            model = ThirdEyeDirect(3, 4, 3).eval()
            state, candidate, history = torch.randn(3, 3), torch.randn(3, 4), torch.randn(3, 2, 3)
            path = Path(root) / "forecaster.pth"
            save_forecaster(model, path, metadata={"synthetic_fixture": True})
            restored, checkpoint = load_forecaster(path)
            for h in (None, history):
                self.assertTrue(torch.equal(model(state, candidate, h), restored(state, candidate, h)))
            self.assertTrue(checkpoint["metadata"]["synthetic_fixture"])
            with self.assertRaises(FileExistsError):
                save_forecaster(model, path)
            with self.assertRaises(ValueError):
                load_forecaster(path, expected_schema={})

    def test_history_and_workspace_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            first = create_trajectory_workspace("fixture", "trajectory_000", root)
            second = create_trajectory_workspace("fixture", "trajectory_001", root)
            with self.assertRaises(FileExistsError):
                create_trajectory_workspace("fixture", "trajectory_000", root)
            store = HistoryStore(first["metadata"] / "history.jsonl")
            losses = dict(target=2., ood=3., retention=4.)
            after = dict(target=1., ood=2., retention=3.)
            self.assertEqual(store.features_before(0), [])
            store.record_applied_transition(from_state=0, selected_candidate_id=1,
                                            before_losses=losses, after_losses=after)
            self.assertEqual(store.features_before(0), [])
            self.assertEqual(store.features_before(1), [[1., 1., 1.]])
            self.assertEqual(HistoryStore(second["metadata"] / "history.jsonl").features_before(0), [])
            with self.assertRaises(ValueError):
                store.features_before(2)

    def test_forecaster_fit_preupdate_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            losses = dict(target=2., ood=3., retention=4.)
            labels, proposals = [], []
            for i in range(3):
                labels.append(dict(candidate_id=i, state_id=0, horizon=2,
                                   label_order=["target", "ood", "retention"],
                                   parent_losses=losses, labels=[.1 + i * .01, .2, .3]))
                proposals.append(dict(candidate_id=i, state_id=0, parent_checkpoint="fixture",
                                      record_type="pre_update_proposal", learning_rate=1e-4,
                                      epochs=1, batch_size=1, max_length=128))
            for name, rows in (("labels", labels), ("proposals", proposals)):
                (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            (root / "parent").write_text(json.dumps(dict(state_id=0,
                record_type="pre_update_parent_evaluation", losses=losses)))
            metrics = fit_forecaster(labels_path=root / "labels", proposals_path=root / "proposals",
                                     parent_metrics_path=root / "parent", output_path=root / "model.pth")
            self.assertEqual(metrics["reload_prediction_max_abs_difference"], 0.)
            self.assertFalse(metrics["held_out_evaluation"])


if __name__ == "__main__":
    unittest.main()
