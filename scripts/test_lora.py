import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen3-0.6B"


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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# 3. Load base model
# --------------------------------------------------

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model = model.to(device)

print("Base model loaded.")


# --------------------------------------------------
# 4. Configure LoRA
# --------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none",
    task_type="CAUSAL_LM"
)


# --------------------------------------------------
# 5. Attach LoRA
# --------------------------------------------------

print("Attaching LoRA...")

model = get_peft_model(
    model,
    lora_config
)


# --------------------------------------------------
# 6. Show trainable parameters
# --------------------------------------------------

print("\nLoRA attached successfully!\n")

model.print_trainable_parameters()
