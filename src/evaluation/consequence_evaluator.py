import torch
from torch.utils.data import DataLoader


def evaluate_loss(
    model,
    dataset,
    device,
    batch_size=1,
):
    """
    Compute average validation loss.

    Lower loss = better performance
    on this validation set.
    """

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for batch in dataloader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1

    model.train()

    if total_batches == 0:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    return total_loss / total_batches
