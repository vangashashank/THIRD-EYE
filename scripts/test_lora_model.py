from src.models.model_loader import load_model_and_tokenizer
from src.models.lora_model import attach_lora


MODEL_NAME = "Qwen/Qwen3-0.6B"


# Load base model
model, tokenizer, device = load_model_and_tokenizer(
    MODEL_NAME
)


# Attach LoRA
model = attach_lora(model)


print("\nLoRA verification:")
model.print_trainable_parameters()
