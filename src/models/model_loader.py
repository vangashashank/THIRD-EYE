import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def place_model(model, device):
    """Quantized models are placed by their loader, not by a second .to()."""
    if not (getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)):
        model = model.to(device)
    return model


def load_model_and_tokenizer(
    model_name: str, *, quantization=None, local_files_only=False, revision=None,
):
    if quantization not in (None, "nf4"):
        raise ValueError("quantization must be None or nf4")
    device = get_device()
    if quantization and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("NF4/bf16 QLoRA requires a BF16-capable CUDA allocation")

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    source_options = {"local_files_only": local_files_only, "revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **source_options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    options = {"dtype": dtype, **source_options}
    if quantization:
        options.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **options)

    model = place_model(model, device)

    print("Model loaded successfully.")

    return model, tokenizer, device
