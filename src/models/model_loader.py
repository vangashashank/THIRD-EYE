import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str):
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device.type == "cuda":
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype
    )

    model = model.to(device)

    print("Model loaded successfully.")

    return model, tokenizer, device
