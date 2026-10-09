import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from src.data.dataset import TextDataset
from src.evaluation.consequence_evaluator import evaluate_loss
from src.models.model_loader import load_model_and_tokenizer
from src.third_eye.checkpoint_manager import CheckpointManager


STATE_ID = 0
MAX_LENGTH = 128
METRICS = ("target", "ood", "retention")

SETS_PATH = Path("configs/forecasting_smoke_sets.json")
OUTPUT_PATH = Path("outputs/forecasting/multiconsequence_smoke_labels.jsonl")


def evaluate_adapter(adapter_path, evaluation_sets):
    with (adapter_path / "adapter_config.json").open() as file:
        adapter_config = json.load(file)

    base_model = model = tokenizer = None

    try:
        base_model, tokenizer, device = load_model_and_tokenizer(
            adapter_config["base_model_name_or_path"]
        )

        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=False,
        )
        model.to(device)

        losses = {}

        for metric in METRICS:
            dataset = TextDataset(
                texts=evaluation_sets[metric],
                tokenizer=tokenizer,
                max_length=MAX_LENGTH,
            )

            losses[metric] = evaluate_loss(
                model=model,
                dataset=dataset,
                device=device,
                batch_size=1,
            )

            print(f"  {metric}: {losses[metric]:.6f}")

        return losses

    finally:
        del model, base_model, tokenizer
        gc.collect()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    with SETS_PATH.open() as file:
        evaluation_sets = json.load(file)

    for metric in METRICS:
        texts = evaluation_sets[metric]

        if (
            not isinstance(texts, list)
            or not texts
            or any(
                not isinstance(text, str) or not text.strip()
                for text in texts
            )
        ):
            raise ValueError(f"Invalid evaluation set: {metric}")

    manager = CheckpointManager()
    parent_path = manager.get_parent(STATE_ID)

    descendant_paths = [
        Path("outputs/trajectory_cache")
        / f"state_{STATE_ID:03d}"
        / f"candidate_{candidate_id}"
        / "t2"
        for candidate_id in range(3)
    ]

    # Check all paths before loading any models.
    for path in [parent_path] + descendant_paths:
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter config: {path}")

    print("Evaluating parent...")
    parent_losses = evaluate_adapter(parent_path, evaluation_sets)

    records = []

    for candidate_id, path in enumerate(descendant_paths):
        print(f"\nEvaluating candidate {candidate_id} at t+2...")
        t2_losses = evaluate_adapter(path, evaluation_sets)

        improvements = [
            parent_losses[metric] - t2_losses[metric]
            for metric in METRICS
        ]

        records.append({
            "state_id": STATE_ID,
            "candidate_id": candidate_id,
            "horizon": 2,
            "smoke_test_only": True,
            "metric_type": "full_text_token_mean_loss_reduction",
            "label_order": list(METRICS),
            "labels": improvements,
            "parent_losses": parent_losses,
            "t2_losses": t2_losses,
            "parent_checkpoint": str(parent_path),
            "t2_checkpoint": str(path),
            "evaluation_sets": {
                metric: evaluation_sets[metric]
                for metric in METRICS
            },
            "max_length": MAX_LENGTH,
        })

        print("  Improvements:", improvements)

    # Save only after every evaluation succeeds.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    serialized = "".join(
        json.dumps(record, allow_nan=False) + "\n"
        for record in records
    )
    OUTPUT_PATH.write_text(serialized)

    print(f"\nSaved {len(records)} candidate label rows.")
    print("Label order:", list(METRICS))
    print("Output:", OUTPUT_PATH)
    print("Smoke-test proxies only; forecasting accuracy is not evaluated.")


if __name__ == "__main__":
    main()
