from peft import PeftModel


def save_adapter(model, tokenizer, save_path):
    print(f"\nSaving adapter to: {save_path}")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Adapter saved successfully.")


def load_adapter(base_model, adapter_path, device):
    print(f"\nLoading adapter from: {adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )

    model = model.to(device)

    print("Adapter loaded successfully.")

    return model
