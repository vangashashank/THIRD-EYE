"""
Standalone local test of extract_candidate_features(), run against the
REAL state_000 parent checkpoint already committed in the THIRD-EYE repo.

This does NOT retrain or modify anything in checkpoints/ -- it only loads
the existing parent adapter, measures gradient/probe features on it, and
restores it. Safe to run repeatedly.

Run from the repo root:
    python test_feature_extraction.py
"""

from pathlib import Path

from peft import PeftModel

from src.models.model_loader import load_model_and_tokenizer
from src.third_eye.checkpoint_manager import CheckpointManager

from feature_extractor import extract_candidate_features


# A small, fixed set of general (non-math) sentences standing in for the
# retention anchor set. Replace with whatever Shashank's real retention
# set ends up being -- this is just enough to smoke-test the code.
RETENTION_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "She picked up her umbrella before leaving the house.",
    "The museum opens at nine in the morning.",
    "He enjoys reading mystery novels on weekends.",
    "The train was delayed by twenty minutes.",
]

# A tiny stand-in training batch for this candidate -- replace with the
# real candidate training texts once wiring this into candidate_generator.py.
TRAIN_TEXTS = [
    "If John has 5 apples and gives away 2, he has 3 apples left.",
    "A rectangle with length 4 and width 3 has an area of 12.",
    "Sarah saved $10 each week for 6 weeks, totaling $60.",
]


def main():
    checkpoint_manager = CheckpointManager(checkpoint_root="checkpoints")

    state_id = 2
    candidate_id = 0

    parent_path = checkpoint_manager.get_parent(state_id) if hasattr(
        checkpoint_manager, "get_parent"
    ) else Path("checkpoints") / f"state_{state_id:03d}" / "parent"

    print(f"Loading parent checkpoint from: {parent_path}")

    base_model_name = "Qwen/Qwen3-0.6B"

    base_model, tokenizer, device = load_model_and_tokenizer(base_model_name)

    model = PeftModel.from_pretrained(
        base_model,
        str(parent_path),
        is_trainable=True,
    )
    model.to(device)

    print("\nRunning feature extraction (gradient check + short probe)...")
    print("This does a few real training steps then restores the weights.\n")

    features = extract_candidate_features(
        model=model,
        tokenizer=tokenizer,
        device=device,
        train_texts=TRAIN_TEXTS,
        retention_texts=RETENTION_TEXTS,
        state_id=state_id,
        candidate_id=candidate_id,
        probe_steps=5,  # kept small for a quick local smoke test
        output_file="outputs/test_candidate_features.jsonl",
    )

    print("\n--- Extracted features ---")
    for key, value in features.items():
        print(f"{key}: {value}")

    print("\nWrote one line to outputs/test_candidate_features.jsonl")
    print("If these numbers look sane (no NaNs, no crashes), the code is ready to hand off.")


if __name__ == "__main__":
    main()
