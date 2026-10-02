import yaml

from src.models.model_loader import load_model_and_tokenizer
from src.models.lora_model import attach_lora
from src.data.dataset import TextDataset
from src.training.trainer import train_lora
from src.training.checkpoint import save_adapter


CONFIG_PATH = "configs/qwen3_0.6b.yaml"


# --------------------------------------------------
# 1. Load config
# --------------------------------------------------

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)


# --------------------------------------------------
# 2. Load model
# --------------------------------------------------

model_name = config["model"]["name"]

model, tokenizer, device = load_model_and_tokenizer(
    model_name
)


# --------------------------------------------------
# 3. Attach LoRA
# --------------------------------------------------

model = attach_lora(
    model=model,
    r=config["lora"]["r"],
    alpha=config["lora"]["alpha"],
    dropout=config["lora"]["dropout"],
    target_modules=config["lora"]["target_modules"]
)

print("\nTrainable parameters:")
model.print_trainable_parameters()


# --------------------------------------------------
# 4. Temporary tiny dataset
# --------------------------------------------------

texts = [
    "Question: What is 2 + 3?\nAnswer: 5",
    "Question: What is 10 - 4?\nAnswer: 6",
]


dataset = TextDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=config["training"]["max_length"]
)


# --------------------------------------------------
# 5. Train
# --------------------------------------------------

model = train_lora(
    model=model,
    dataset=dataset,
    device=device,
    epochs=config["training"]["epochs"],
    batch_size=config["training"]["batch_size"],
    learning_rate=config["training"]["learning_rate"]
)


# --------------------------------------------------
# 6. Save adapter
# --------------------------------------------------

save_adapter(
    model=model,
    tokenizer=tokenizer,
    save_path=config["output"]["adapter_path"]
)


print("\nTask 2 training pipeline completed successfully!")
