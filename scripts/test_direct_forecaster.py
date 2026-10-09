import torch

from src.third_eye.direct_forecaster import ThirdEyeDirect


def main():
    torch.manual_seed(42)

    # Temporary dimensions for architecture testing.
    # We will set the real dimensions during feature extraction.
    model = ThirdEyeDirect(
        state_dim=3,
        candidate_dim=4,
        history_dim=3,
        hidden_dim=32,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count < 1_000_000

    # Synthetic inputs: one shared state, three proposed candidates.
    state_features = torch.randn(1, 3).repeat(3, 1)
    candidate_features = torch.randn(3, 4)
    history = torch.randn(1, 2, 3).repeat(3, 1, 1)

    predictions = model(
        state_features,
        candidate_features,
        history,
    )

    assert predictions.shape == (3, 3)
    assert torch.isfinite(predictions).all()

    # Verify that training gradients reach every model parameter.
    synthetic_targets = torch.randn(3, 3)

    loss = torch.nn.functional.mse_loss(
        predictions,
        synthetic_targets,
    )

    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    # The initial state has no recursive history.
    with torch.no_grad():
        initial_predictions = model(
            state_features,
            candidate_features,
            history=None,
        )

    assert initial_predictions.shape == (3, 3)
    assert torch.isfinite(initial_predictions).all()

    print("Third Eye-Direct smoke test passed.")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Output shape: {tuple(predictions.shape)}")
    print("Forward pass, backward pass, and no-history case passed.")
    print("Synthetic test only; forecasting accuracy is not evaluated.")


if __name__ == "__main__":
    main()
