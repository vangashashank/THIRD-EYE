import argparse
import json
import socket
from pathlib import Path

import torch

from src.third_eye.direct_forecaster import ThirdEyeDirect
from src.third_eye.forecaster_checkpoint import (
    FEATURE_SCHEMA, load_forecaster, save_forecaster,
)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fit_forecaster(*, labels_path, proposals_path, parent_metrics_path, output_path):
    if socket.gethostname() == "dgx-login01":
        raise RuntimeError("Run forecaster fitting on a Slurm compute node")
    labels = sorted(read_rows(labels_path), key=lambda row: row["candidate_id"])
    proposals = sorted(read_rows(proposals_path), key=lambda row: row["candidate_id"])
    parent = json.loads(Path(parent_metrics_path).read_text())
    assert parent["record_type"] == "pre_update_parent_evaluation"
    assert parent["state_id"] == 0
    for rows in (labels, proposals):
        assert len(rows) == 3 and [row["candidate_id"] for row in rows] == [0, 1, 2]
        assert all(row["state_id"] == 0 for row in rows)
    assert len({row["parent_checkpoint"] for row in proposals}) == 1
    assert all(row["record_type"] == "pre_update_proposal" for row in proposals)
    metrics = FEATURE_SCHEMA["labels"]
    for row in labels:
        assert row["horizon"] == 2 and row["label_order"] == metrics
        assert row["parent_losses"] == parent["losses"]
    # Inputs come from the pre-update parent measurements and recorded proposals.
    state = torch.tensor([[parent["losses"][key] for key in metrics]], dtype=torch.float32).repeat(3, 1)
    candidate = torch.tensor([
        [row["learning_rate"] / 1e-4, row["epochs"], row["batch_size"], row["max_length"] / 128]
        for row in proposals
    ], dtype=torch.float32)
    targets = torch.tensor([row["labels"] for row in labels], dtype=torch.float32)
    assert all(torch.isfinite(value).all() for value in (state, candidate, targets))
    torch.manual_seed(42)
    model = ThirdEyeDirect(3, 4, 3, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = torch.nn.functional.mse_loss(model(state, candidate), targets).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(state, candidate), targets)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(state, candidate)
        final = torch.nn.functional.mse_loss(predictions, targets).item()
        baseline = (targets - targets.mean(0, keepdim=True)).square().mean().item()
    assert final < initial
    record = {
        "new_fit": True, "seed": 42, "optimizer": "Adam", "learning_rate": 0.01,
        "optimizer_steps": 200, "training_rows": 3, "history_used": False,
        "initial_training_mse": initial, "final_training_mse": final,
        "constant_output_mse": baseline, "held_out_evaluation": False,
        "feature_schema": FEATURE_SCHEMA,
        "predictions": predictions.tolist(), "targets": targets.tolist(),
        "identical_candidate_features": bool(torch.equal(candidate, candidate[0:1].repeat(3, 1))),
    }
    save_forecaster(model, output_path, metadata=record)
    restored, _ = load_forecaster(output_path)
    with torch.no_grad():
        restored_predictions = restored(state, candidate)
    assert torch.equal(predictions, restored_predictions)
    record["reload_prediction_max_abs_difference"] = (predictions - restored_predictions).abs().max().item()
    record["checkpoint_saved_and_reloaded"] = True
    print(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--parent-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_forecaster(labels_path=args.labels, proposals_path=args.proposals,
                   parent_metrics_path=args.parent_metrics, output_path=args.output)


if __name__ == "__main__":
    main()
