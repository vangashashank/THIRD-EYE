"""SYNTHETIC CPU readiness fixtures. Not cached Qwen or real research evidence."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import inspect_sprint2_server
from scripts.measure_sprint2_throughput import main
from scripts.inspect_sprint2_model_bindings import module_manifest, verify_loaded_root
from src.third_eye.feature_contract import ContractError
from src.third_eye.sprint2_collection import binding_blockers, load_config
from src.third_eye.sprint2_runtime import digest_state, TrainingSnapshot
from src.third_eye.sprint2_throughput import measure_backend, operation_counts


ROOT = Path(__file__).resolve().parents[1]


class SyntheticBackend:
    """Tiny actual CPU parameter updates with fabricated source identities/tokens."""
    def __init__(self):
        self.model = torch.nn.Linear(2, 1)
        self.model.eval()  # The benchmark starts from the explicitly loaded eval-mode root.
        self.config = {"continuation": {"prompt_pools": {"SYNTHETIC-g": ["fake-%d" % i for i in range(128)]},
            "recipe": {"SYNTHETIC": True}, "candidate_master_seed": 1, "master_seed": 2},
            "update": {"steps": 32}, "diagnostic_counts": {"target": 32, "ood": 32, "retention": 256}}
        self.spec = {"synthetic_only": True, "semantic_hash": "f"*64,
            "bindings": {"lineage": {"identities": {"SYNTHETIC-t": {"root_sha256": self.parent_sha()}}}}}
        self.fail = False
    def check_limits(self):
        pass
    def load_parent(self, trajectory, snapshot):
        self.model.eval()
    def parent_sha(self):
        return digest_state(dict(self.model.named_parameters()))
    def _rows(self, role):
        return [{"prompt_id": "SYNTHETIC-"+role+str(i)} for i in range(4)]
    def _generate(self, row):
        return {**row, "generated_tokens": 2, "truncated": False, "text": "SYNTHETIC"}
    def _failure_batch(self, group, plan):
        a = [self._generate({"prompt_id": p}) for p in plan.prompt_ids]
        return [torch.ones(1, 2)], {p: "SYNTHETIC" for p in plan.prompt_ids}, {"original": a, "corrections": copy.deepcopy(a)}
    def _update(self, examples, seed):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01)
        losses = []
        for i in range(32):
            optimizer.zero_grad()
            loss = self.model(examples[0]).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if self.fail:
                raise RuntimeError("SYNTHETIC interrupted update")
        return losses


class ReadinessTests(unittest.TestCase):
    def test_metadata_root_actual_tensor_roundtrip_not_missing_initializations(self):
        loaded = {"SYNTHETIC-A": torch.ones(2, 3)}
        verify_loaded_root(loaded, {"SYNTHETIC-A": torch.ones(2, 3, dtype=torch.float16)})
        for bad in ({}, {"SYNTHETIC-A": torch.zeros(2, 3)},
                    {"SYNTHETIC-A": torch.full((2, 3), float("nan"))}):
            with self.assertRaises(ContractError):
                verify_loaded_root(loaded, bad)

    def test_metadata_manifest_actual_names_shapes_and_numeric_order(self):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict()
        for block in (10, 2):
            layer = torch.nn.Module()
            for projection in ("v_proj", "q_proj"):
                module = torch.nn.Module()
                module.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
                module.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})
                setattr(layer, projection, module)
            model.layers[str(block)] = layer
        count, manifest = module_manifest(model, ["q_proj", "v_proj"])
        self.assertEqual(count, 4)
        self.assertEqual([r["qualified_name"] for r in manifest],
            ["layers.2.q_proj", "layers.2.v_proj", "layers.10.q_proj", "layers.10.v_proj"])
        self.assertEqual(manifest[0]["a_name"], "layers.2.q_proj.lora_A.default.weight")
        self.assertEqual(manifest[0]["a_shape"], [2, 3])
        self.assertEqual(manifest[0]["a_dtype"], "torch.float32")
        model.extra = torch.nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ContractError, "Unexpected trainable"):
            module_manifest(model, ["q_proj", "v_proj"])

    def test_metadata_manifest_missing_pair_cannot_guess_dimensions(self):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict({"2": torch.nn.Module()})
        model.layers["2"].q_proj = torch.nn.Module()
        model.layers["2"].q_proj.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
        with self.assertRaisesRegex(ContractError, "Unsupported/missing"):
            module_manifest(model, ["q_proj", "v_proj"])

    def test_counts_preserve_t5_and_batch_scope(self):
        self.assertEqual(operation_counts()["optimizer_updates"], 1185)
        self.assertEqual(operation_counts()["endpoint_scoring_generations"], 11200)
        self.assertEqual(operation_counts(6)["states"], 30)
        self.assertEqual(operation_counts(6)["records"], 90)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ContractError):
                operation_counts(value)

    def test_benchmark_actual_cpu_updates_restore_no_labels_or_history(self):
        b = SyntheticBackend()
        config = copy.deepcopy(b.config)
        control = TrainingSnapshot(b.model)
        with tempfile.TemporaryDirectory() as tmp:
            report = measure_backend(b, "SYNTHETIC-t", "SYNTHETIC-g", Path(tmp))
            self.assertTrue(report["synthetic_only"])
            self.assertTrue(report["restoration_verified"])
            self.assertFalse(report["feasibility_established"])
            self.assertEqual(report["optimizer_updates_measured"], 32)
            self.assertEqual(report["diagnostic_generations_measured"], 12)
            times = report["timings"]
            expected = 960*times["update_seconds"]/32 + 35*(8*times["target_generation_seconds"]
                + 8*times["ood_generation_seconds"] + 64*times["retention_generation_seconds"])
            self.assertAlmostEqual(report["partial_workload_extrapolation_seconds"], expected)
            self.assertNotEqual(report["parent_sha256"], report["temporary_endpoint_sha256"])
            self.assertTrue((Path(tmp)/"adapter_endpoint.pt").exists())
            self.assertFalse((Path(tmp)/"meta.jsonl").exists())
            self.assertFalse((Path(tmp)/"manifest.json").exists())
            self.assertEqual(control.fingerprint(), control.before)
            with self.assertRaises(FileExistsError):
                measure_backend(b, "SYNTHETIC-t", "SYNTHETIC-g", Path(tmp))
        self.assertEqual(b.config, config)

    def test_benchmark_failure_restores_and_retains_sealed_plan(self):
        b = SyntheticBackend()
        b.fail = True
        control = TrainingSnapshot(b.model)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                measure_backend(b, "SYNTHETIC-t", "SYNTHETIC-g", Path(tmp))
            self.assertTrue((Path(tmp)/"plan.json").exists())
            self.assertTrue((Path(tmp)/"generations.json").exists())
            self.assertFalse((Path(tmp)/"report.json").exists())
            self.assertEqual(control.fingerprint(), control.before)

    def test_root_identity_mismatch_before_generation(self):
        b = SyntheticBackend()
        b.spec["bindings"]["lineage"]["identities"]["SYNTHETIC-t"]["root_sha256"] = "0"*64
        with tempfile.TemporaryDirectory() as tmp, patch.object(b, "_generate") as generate:
            with self.assertRaisesRegex(ContractError, "root differs"):
                measure_backend(b, "SYNTHETIC-t", "SYNTHETIC-g", Path(tmp))
            generate.assert_not_called()

    def test_config_blockers_precise_and_offline_cli_no_model_load(self):
        blockers = binding_blockers(load_config(ROOT/"configs/sprint2_collection.yaml"))
        for field in ("datasets.target", "datasets.retention", "continuation.prompt_pools", "continuation.recipe",
                      "definitions.nll_reduction", "definitions.duplicate_tolerance", "caps_review_reference"):
            self.assertIn("unbound "+field, blockers)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--config", str(ROOT/"configs/sprint2_collection.yaml"),
                                   "--trajectory", "SYNTHETIC-NOT-A-BOUND-ID"]), 2)

    def test_server_inventory_only_allowlisted_read_only_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(inspect_sprint2_server, "command", side_effect=lambda argv, cwd=None: {"command": argv}) as command, \
                 patch("pathlib.Path.home", return_value=Path(tmp)), \
                 patch.dict("os.environ", {"HF_HOME": str(Path(tmp)/"fake-cache"), "HF_DATASETS_CACHE": str(Path(tmp)/"fake-datasets")}):
                report = inspect_sprint2_server.inspect(tmp)
            programs = {c.args[0][0] for c in command.call_args_list}
            self.assertEqual(programs, {"hostname", "date", "nvidia-smi", "sinfo", "squeue", "sacct", "quota", "df", "du", "git"})
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertEqual(report["data_manifests"], [])
            self.assertTrue(all(not r["pinned_snapshot_exists"] for r in report["cached_models"]))

    def test_source_comparison_preserves_server_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"src").mkdir()
            (root/"src/local.py").write_text("# SYNTHETIC fixture\n")
            path = root/"server.json"
            path.write_text(json.dumps({"source_hashes": {"scripts/slurm/server-only.sbatch": "f"*64}}))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                inspect_sprint2_server.main(["--project", tmp, "--compare", str(path)])
            report = json.loads(stream.getvalue())
            self.assertEqual(report["server_only_preserve"], ["scripts/slurm/server-only.sbatch"])
            self.assertEqual(report["local_only"], ["src/local.py"])


if __name__ == "__main__":
    unittest.main()
