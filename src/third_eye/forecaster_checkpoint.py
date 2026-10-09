from pathlib import Path

import torch

from src.third_eye.direct_forecaster import ThirdEyeDirect


FEATURE_SCHEMA = {
    "state": ["parent_target_loss", "parent_ood_loss", "parent_retention_loss"],
    "candidate": ["learning_rate_div_1e-4", "epochs", "batch_size", "max_length_div_128"],
    "history": ["applied_target_loss_reduction", "applied_ood_loss_reduction",
                "applied_retention_loss_reduction"],
    "labels": ["target", "ood", "retention"],
    "horizon": 2,
}


def save_forecaster(model, path, *, metadata=None):
    config = {
        "state_dim": model.state_dim,
        "candidate_dim": model.candidate_dim,
        "history_dim": model.history_dim,
        "hidden_dim": model.history_encoder.hidden_size,
    }
    if (config["state_dim"], config["candidate_dim"], config["history_dim"]) != (3, 4, 3):
        raise ValueError("Model dimensions do not match the feature schema")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        torch.save({
            "format_version": 1,
            "model_config": config,
            "feature_schema": FEATURE_SCHEMA,
            "state_dict": {name: value.detach().cpu()
                           for name, value in model.state_dict().items()},
            "metadata": metadata or {},
        }, file)
    return path


def load_forecaster(path, *, expected_schema=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    schema = expected_schema if expected_schema is not None else FEATURE_SCHEMA
    if checkpoint["format_version"] != 1 or checkpoint["feature_schema"] != schema:
        raise ValueError("Unsupported checkpoint version or feature schema")
    model = ThirdEyeDirect(**checkpoint["model_config"])
    if (model.state_dim, model.candidate_dim, model.history_dim) != (3, 4, 3):
        raise ValueError("Checkpoint dimensions do not match the feature schema")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
