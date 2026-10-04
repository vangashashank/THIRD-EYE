from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"

# 1. Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# 2. Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")
print("Parameters:", f"{model.num_parameters():,}")

# 4. Prepare input
prompt = "What is machine learning?"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

# 5. Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

# 6. Decode
response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nResponse:")
print(response)
