import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "outputs/adapters/task2_smoke"


# --------------------------------------------------
# 1. Select device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)


# --------------------------------------------------
# 3. Load fresh base model
# --------------------------------------------------

print("Loading fresh base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

base_model = base_model.to(device)


# --------------------------------------------------
# 4. Load trained LoRA adapter
# --------------------------------------------------

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model = model.to(device)
model.eval()

print("LoRA adapter loaded successfully!")


# --------------------------------------------------
# 5. Test inference
# --------------------------------------------------

prompt = "Question: What is 2 + 3?\nAnswer:"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)


print("\nGenerating response...")


with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )


response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


print("\nResponse:")
print(response)
