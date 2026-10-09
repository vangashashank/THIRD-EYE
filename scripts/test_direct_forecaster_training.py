import json
from pathlib import Path

import torch

from src.third_eye.direct_forecaster import ThirdEyeDirect


def main():
    torch.manual_seed(42)

    path = Path(
        "outputs/forecasting/multiconsequence_smoke_labels.jsonl"
    )

    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    records.sort(key=lambda row: row["candidate_id"])

    assert len(records) == 3
    assert [row["candidate_id"] for row in records] == [0, 1, 2]

    metrics = ["target", "ood", "retention"]

    for row in records:
        assert row["state_id"] == 0
        assert row["horizon"] == 2
        assert row["label_order"] == metrics
        assert row["parent_losses"] == records[0]["parent_losses"]

    # Parent measurements are available before candidate updates.
    state_features = torch.tensor(
        [[records[0]["parent_losses"][key] for key in metrics]],
        dtype=torch.float32,
    ).repeat(3, 1)

    # Settings from test_candidate_training.py:
    # [learning_rate / 1e-4, epochs, batch_size, max_length / 128]
    # All three candidates have identical feature descriptions.
    candidate_features = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float32,
    ).repeat(3, 1)

    targets = torch.tensor(
        [row["labels"] for row in records],
        dtype=torch.float32,
    )
    assert torch.isfinite(targets).all()

    model = ThirdEyeDirect(
        state_dim=3,
        candidate_dim=4,
        history_dim=3,
        hidden_dim=32,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def predict():
        return model(
            state_features,
            candidate_features,
            history=None,
        )

    before = [
        parameter.detach().clone()
        for parameter in model.predictor.parameters()
    ]

    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(
            predict(), targets
        ).item()

    for _ in range(200):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(predict(), targets)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = predict()
        final_loss = torch.nn.functional.mse_loss(
            predictions, targets
        ).item()

        # Best possible MSE when all rows get the same prediction.
        mean_targets = targets.mean(dim=0, keepdim=True)
        constant_prediction_floor = (
            (targets - mean_targets).square().mean().item()
        )

    assert final_loss < initial_loss
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.predictor.parameters())
    )
    assert torch.allclose(predictions, predictions[0:1].expand_as(predictions))

    print("Direct forecaster training smoke test passed.")
    print(f"Initial training MSE: {initial_loss:.8f}")
    print(f"Final training MSE:   {final_loss:.8f}")
    print(f"Constant-output MSE floor: {constant_prediction_floor:.8f}")
    print("Predictions:")
    print(predictions)
    print("Predictor weights updated; history encoder was unused.")
    print("One-state training check only; no held-out evaluation.")


if __name__ == "__main__":
    main()
