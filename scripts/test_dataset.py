from transformers import AutoTokenizer
from src.data.dataset import TextDataset


MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

texts = [
    "Question: What is 2 + 3?\nAnswer: 5",
    "Question: What is 10 - 4?\nAnswer: 6",
]

dataset = TextDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=32
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("\nFirst sample:")
print("input_ids shape:", sample["input_ids"].shape)
print("attention_mask shape:", sample["attention_mask"].shape)
print("labels shape:", sample["labels"].shape)

print("\nDecoded text:")
print(tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
