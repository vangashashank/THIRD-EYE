from src.models.model_loader import load_model_and_tokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"


model, tokenizer, device = load_model_and_tokenizer(
    MODEL_NAME
)


print("\nVerification")
print("Device:", device)
print("Parameters:", f"{model.num_parameters():,}")
print("Tokenizer:", tokenizer.__class__.__name__)
