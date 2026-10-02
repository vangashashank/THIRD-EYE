import yaml
import torch

from src.models.model_loader import load_model_and_tokenizer
from src.training.checkpoint import load_adapter


CONFIG_PATH = "configs/qwen3_0.6b.yaml"


# --------------------------------------------------
# 1. Load config
# --------------------------------------------------

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)


model_name = config["model"]["name"]
adapter_path = config["output"]["adapter_path"]


# --------------------------------------------------
# 2. Load fresh base model
# --------------------------------------------------

model, tokenizer, device = load_model_and_tokenizer(
    model_name
)


# --------------------------------------------------
# 3. Load trained adapter
# --------------------------------------------------

model = load_adapter(
    base_model=model,
    adapter_path=adapter_path,
    device=device
)

model.eval()


# --------------------------------------------------
# 4. Inference
# --------------------------------------------------

prompt = "Question: What is 2 + 3?\nAnswer:"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )


generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)


print("\nPrompt:")
print(prompt)

print("\nModel response:")
print(response)
