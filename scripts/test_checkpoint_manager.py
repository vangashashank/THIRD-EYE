from pathlib import Path
from src.third_eye.checkpoint_manager import CheckpointManager


def main():
    manager = CheckpointManager()

    # Change this path to your existing LoRA adapter folder
    source_checkpoint = "outputs/adapters/task2_final"

    print("1. Saving parent checkpoint...")

    parent = manager.save_parent(
        source_checkpoint,
        state_id=0
    )

    print("Parent saved at:", parent)

    print("\n2. Saving candidate checkpoint...")

    candidate = manager.save_candidate(
        source_checkpoint,
        state_id=0,
        candidate_id=0
    )

    print("Candidate saved at:", candidate)

    print("\n3. Testing rollback...")

    rollback = manager.rollback_to_parent(
        state_id=0
    )

    print("Rollback checkpoint:", rollback)

    print("\n4. Testing promotion...")

    promoted = manager.promote_candidate(
        current_state_id=0,
        candidate_id=0,
        next_state_id=1
    )

    print("Candidate promoted to:", promoted)

    print("\nCheckpoint manager test completed successfully.")


if __name__ == "__main__":
    main()
