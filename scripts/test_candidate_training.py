from src.third_eye.checkpoint_manager import CheckpointManager
from src.third_eye.candidate_generator import CandidateGenerator


def main():

    manager = CheckpointManager(
        checkpoint_root="checkpoints"
    )

    generator = CandidateGenerator(
        checkpoint_manager=manager,
        k_candidates=3,
    )

    texts = [
        "Question: What is 2 + 3?\nAnswer: 5",
        "Question: What is 10 - 4?\nAnswer: 6",
    ]

    print(
        "Starting Task 3B REAL candidate training..."
    )

    candidates = generator.generate_candidates(
        state_id=0,
        texts=texts,
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        max_length=128,
    )

    print("\n==========================")
    print("Generated trained candidates")
    print("==========================")

    for candidate in candidates:
        print(candidate)

    print(
        "\nTask 3B candidate training completed."
    )


if __name__ == "__main__":
    main()
