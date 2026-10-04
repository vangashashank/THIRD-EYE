from src.third_eye.checkpoint_manager import CheckpointManager
from src.third_eye.state_manager import StateManager


def main():

    checkpoint_manager = CheckpointManager(
        checkpoint_root="checkpoints"
    )

    state_manager = StateManager(
        checkpoint_manager=checkpoint_manager
    )

    print("Recording candidate metadata...\n")

    candidates = [
        {
            "candidate_id": 0,
            "seed": 11,
            "training_loss": 1.7554,
        },
        {
            "candidate_id": 1,
            "seed": 22,
            "training_loss": 1.7545,
        },
        {
            "candidate_id": 2,
            "seed": 33,
            "training_loss": 1.7458,
        },
    ]

    for candidate in candidates:

        checkpoint_path = (
            f"checkpoints/state_000/"
            f"candidates/"
            f"candidate_{candidate['candidate_id']}"
        )

        record = state_manager.record_candidate(
            state_id=0,
            candidate_id=candidate["candidate_id"],
            seed=candidate["seed"],
            training_loss=candidate["training_loss"],
            learning_rate=1e-4,
            batch_size=1,
            checkpoint_path=checkpoint_path,
        )

        print(record)

    print("\nMetadata recorded.")

    print("\nSelecting candidate 2...")

    promoted_path = (
        state_manager.promote_candidate(
            state_id=0,
            candidate_id=2,
        )
    )

    print(
        "\nCandidate promoted to:"
    )

    print(promoted_path)

    print(
        "\nTask 3C state-manager test completed."
    )


if __name__ == "__main__":
    main()
