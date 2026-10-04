import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen3-0.6B"


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 2. Load tokenizer + model
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model = model.to(device)


# --------------------------------------------------
# 3. Attach LoRA
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

model = get_peft_model(model, lora_config)

print("\nTrainable parameters:")
model.print_trainable_parameters()


# --------------------------------------------------
# 4. Tiny training example
# --------------------------------------------------

text = """
Question: What is 2 + 3?
Answer: 5
"""

inputs = tokenizer(
    text,
    return_tensors="pt"
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)


# --------------------------------------------------
# 5. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# --------------------------------------------------
# 6. Tiny training loop
# --------------------------------------------------

model.train()

print("\nStarting training...\n")

for step in range(5):

    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids
    )

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    print(
        f"Step {step + 1} | Loss: {loss.item():.4f}"
    )


print("\nTraining completed!")


# --------------------------------------------------
# 7. Save LoRA adapter
# --------------------------------------------------

SAVE_PATH = "outputs/adapters/task2_smoke"

print("\nSaving LoRA adapter...")

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"Adapter saved to: {SAVE_PATH}")
