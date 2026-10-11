"""Audit cached model/root metadata independently of feature formulas; no updates/labels.

Default: inspect supplied adapter config/checksums only. --execute loads the cached
model inside a separately authorized allocation and reports ACTUAL runtime names,
shapes, dtypes, template and trainable-state hash. Does not invent root independence.
"""
import argparse
import hashlib
import json
import os
import re
import socket
from pathlib import Path

from src.third_eye.feature_contract import ContractError
from src.third_eye.sprint2_collection import file_sha, load_config, write_json
from src.third_eye.sprint2_protocol import canonical_lora_manifest, LoraModule


def module_manifest(model, placement_order):
    tensors = {n: p for n, p in model.named_parameters() if p.requires_grad}
    pairs = {}
    for name, p in tensors.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])\.default\.weight", name)
        if not match:
            raise ContractError("Unexpected trainable parameter; cannot guess a LoRA manifest: "+name)
        module, side = match.groups()
        row = pairs.setdefault(module, {"qualified_name": module})
        row[side.lower()+"_name"] = name
        row[side.lower()+"_shape"] = list(p.shape)
        row[side.lower()+"_dtype"] = str(p.dtype)
    for name, row in pairs.items():
        block = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        projection = name.rsplit(".", 1)[-1]
        if not block or projection not in placement_order or "a_name" not in row or "b_name" not in row:
            raise ContractError("Unsupported/missing actual module A/B/block/placement: "+name)
        row.update(block_index=int(block.group(1)), projection=projection)
    count, order = canonical_lora_manifest([LoraModule(r["block_index"], r["projection"], n)
        for n, r in pairs.items()], placement_order=placement_order)
    return count, [pairs[n] for n in order]


def verify_loaded_root(loaded, saved):
    import torch
    if set(loaded) != set(saved):
        raise ContractError("Loaded root keys differ from saved adapter; no initialized missing parameters accepted")
    for name, value in loaded.items():
        original = saved[name]
        if (value.shape != original.shape or not bool(torch.isfinite(original).all())
                or not torch.equal(value.detach().cpu(), original.to(dtype=value.dtype, device="cpu"))):
            raise ContractError("Actual loaded root tensor differs from cached adapter: "+name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/sprint2_collection.yaml")
    ap.add_argument("--root-adapter", required=True, help="Existing real adapter; no independence inferred")
    ap.add_argument("--out")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    try:
        config = load_config(args.config)
        root = Path(args.root_adapter).resolve()
        adapter = json.loads((root/"adapter_config.json").read_text())
        expected = {"base_model_name_or_path": config["model"]["name"], "bias": "none",
            "r": config["lora"]["r"], "lora_alpha": config["lora"]["alpha"], "lora_dropout": config["lora"]["dropout"]}
        if any(adapter.get(k) != v for k, v in expected.items()) or set(adapter.get("target_modules", [])) != set(config["placement_order"]):
            raise ContractError("Actual adapter/model/placement recipe differs from pinned pilot")
        report = {"purpose": "metadata_audit_not_independence_or_measurement_validation", "model": config["model"],
            "root_path": str(root), "root_files": {n: file_sha(root/n) for n in ("adapter_config.json", "adapter_model.safetensors")},
            "independent_lineage_provenance": "UNBOUND: audit adaptation families, prefixes and assigned group separately"}
        if not args.execute:
            print(json.dumps({**report, "status": "adapter_metadata_only_no_model_load"}, indent=2))
            return 0
        if not args.out or not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("dgx-login"):
            raise ContractError("Runtime audit needs new --out and separately authorized existing compute allocation")
        if Path(args.out).exists() or config["model"]["local_files_only"] is not True:
            raise ContractError("New output only; cached model only")
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict
        from safetensors.torch import load_file
        from src.models.model_loader import load_model_and_tokenizer
        from src.third_eye.sprint2_runtime import digest_state
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractError("Expected one allocated CUDA/MIG device")
        binding = config["model"]
        model, tokenizer, _ = load_model_and_tokenizer(binding["name"], revision=binding["revision"],
            quantization=binding["quantization"], local_files_only=True)
        if binding["quantization"]:
            model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, str(root), is_trainable=True, local_files_only=True)
        verify_loaded_root(get_peft_model_state_dict(model), load_file(str(root/"adapter_model.safetensors")))
        count, manifest = module_manifest(model, config["placement_order"])
        props = torch.cuda.get_device_properties(0)
        report.update(num_lora_layers=count, module_manifest=manifest,
            root_sha256=digest_state({n: p for n, p in model.named_parameters() if p.requires_grad}),
            chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            tokenizer_class=type(tokenizer).__name__, fast_tokenizer=tokenizer.is_fast,
            device_name=props.name, device_total_memory_bytes=props.total_memory,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            job_id=os.environ["SLURM_JOB_ID"], updates=0, generated_answers=0)
        if not tokenizer.is_fast:
            raise ContractError("Pinned runtime requires exact fast-tokenizer offsets")
        write_json(args.out, report)
        print(json.dumps(report, indent=2))
        return 0
    except (ContractError, OSError, ValueError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
