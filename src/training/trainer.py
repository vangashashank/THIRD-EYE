import math
import time

import torch
from torch.utils.data import DataLoader


def train_lora(
    model,
    dataset,
    device,
    epochs=1,
    batch_size=1,
    learning_rate=1e-4
):
    if min(epochs, batch_size) <= 0 or len(dataset) == 0:
        raise ValueError("Training requires positive epochs/batch size and nonempty data")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    model.train()
    started = time.perf_counter()
    step_losses = []

    print("\nStarting training...\n")

    for epoch in range(epochs):

        total_loss = 0.0

        for step, batch in enumerate(dataloader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite training loss")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_losses.append(loss.item())

            print(
                f"Epoch {epoch + 1} | "
                f"Step {step + 1} | "
                f"Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(dataloader)

        print(
            f"\nEpoch {epoch + 1} completed | "
            f"Average Loss: {avg_loss:.4f}\n"
        )

    print("Training completed!")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.last_training_metrics = {
        "optimizer_steps": len(step_losses),
        "step_losses": step_losses,
        "mean_training_loss": sum(step_losses) / len(step_losses),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    return model
