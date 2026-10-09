import math

import torch
from torch.utils.data import DataLoader


def evaluate_loss(model, dataset, device, batch_size=1):
    """Compute mean loss per valid predicted token."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    was_training = model.training
    total_loss = 0.0
    total_tokens = 0

    model.eval()

    try:
        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                # Causal LMs predict the next token, so the
                # first label is excluded from the loss.
                valid_tokens = (
                    batch["labels"][:, 1:] != -100
                ).sum().item()

                if valid_tokens == 0:
                    continue

                loss = model(**batch).loss.item()

                if not math.isfinite(loss):
                    raise ValueError("Evaluation loss is not finite.")

                total_loss += loss * valid_tokens
                total_tokens += valid_tokens
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("No valid prediction tokens to evaluate.")

    return total_loss / total_tokens
