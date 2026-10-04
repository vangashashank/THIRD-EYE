"""
Standalone local test of evaluate_current_state(), run against the real
state_000 parent checkpoint. This calls model.generate() 15 times total
(5 problems x 3 sets), so expect this to take longer than the feature
extraction test, especially on CPU.

Run from the repo root:
    python test_evaluation.py
"""

from pathlib import Path

from peft import PeftModel

from src.models.model_loader import load_model_and_tokenizer
from src.third_eye.checkpoint_manager import CheckpointManager

from evaluation import evaluate_current_state


def main():
    checkpoint_manager = CheckpointManager(checkpoint_root="checkpoints")

    state_id = 0
    parent_path = checkpoint_manager.get_parent(state_id) if hasattr(
        checkpoint_manager, "get_parent"
    ) else Path("checkpoints") / f"state_{state_id:03d}" / "parent"

    print(f"Loading parent checkpoint from: {parent_path}")

    base_model_name = "Qwen/Qwen3-0.6B"

    base_model, tokenizer, device = load_model_and_tokenizer(base_model_name)

    model = PeftModel.from_pretrained(
        base_model,
        str(parent_path),
        is_trainable=False,  # evaluation only, no training needed here
    )
    model.to(device)

    print("\nRunning target/OOD/retention accuracy evaluation...")
    print("This calls model.generate() 15 times -- may take a few minutes on CPU.\n")

    results = evaluate_current_state(model, tokenizer, device)

    print("\n--- Current-state accuracy ---")
    for key, value in results.items():
        print(f"{key}: {value:.2%}")

    print("\nNote: these numbers use the small PLACEHOLDER problem sets in evaluation.py.")
    print("Not meaningful research numbers yet -- this just confirms the verifier logic works.")


if __name__ == "__main__":
    main()
