from peft import PeftModel
from pathlib import Path
import hashlib

import torch
from peft import get_peft_model_state_dict

from src.models.lora_model import prepare_adapter_base
from src.models.model_loader import place_model


def adapter_state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_state_sha256(model):
    return adapter_state_digest(get_peft_model_state_dict(model))


def save_adapter(model, tokenizer, save_path):
    path = Path(save_path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite adapter: {path}")
    print(f"\nSaving adapter to: {save_path}")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Adapter saved successfully.")


def load_adapter(base_model, adapter_path, device, *, is_trainable=False):
    print(f"\nLoading adapter from: {adapter_path}")

    base_model = prepare_adapter_base(base_model, is_trainable=is_trainable)
    model = PeftModel.from_pretrained(
        base_model, str(adapter_path), is_trainable=is_trainable,
    )

    model = place_model(model, device)

    print("Adapter loaded successfully.")

    return model
