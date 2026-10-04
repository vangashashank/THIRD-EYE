from src.models.model_loader import load_model_and_tokenizer
from src.models.lora_model import attach_lora
from src.data.dataset import TextDataset
from src.training.trainer import train_lora
from src.training.checkpoint import save_adapter, load_adapter


MODEL_NAME = "Qwen/Qwen3-0.6B"
SAVE_PATH = "outputs/adapters/task2_modular"


# --------------------------------------------------
# 1. Load model
# --------------------------------------------------

model, tokenizer, device = load_model_and_tokenizer(
    MODEL_NAME
)


# --------------------------------------------------
# 2. Attach LoRA
# --------------------------------------------------

model = attach_lora(model)

print("\nTrainable parameters:")
model.print_trainable_parameters()


# --------------------------------------------------
# 3. Dataset
# --------------------------------------------------

texts = [
    "Question: What is 2 + 3?\nAnswer: 5",
    "Question: What is 10 - 4?\nAnswer: 6",
]

dataset = TextDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=32
)


# --------------------------------------------------
# 4. Train
# --------------------------------------------------

model = train_lora(
    model=model,
    dataset=dataset,
    device=device,
    epochs=1,
    batch_size=1,
    learning_rate=1e-4
)


# --------------------------------------------------
# 5. Save adapter
# --------------------------------------------------

save_adapter(
    model=model,
    tokenizer=tokenizer,
    save_path=SAVE_PATH
)


print("\nCheckpoint test completed!")
