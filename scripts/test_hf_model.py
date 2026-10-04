from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"

print("Downloading/loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading/loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

print("Model loaded successfully!")

print("Model:", MODEL_NAME)
print("Parameters:", f"{model.num_parameters():,}")
