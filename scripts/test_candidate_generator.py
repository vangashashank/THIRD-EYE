from src.third_eye.checkpoint_manager import CheckpointManager
from src.third_eye.candidate_generator import CandidateGenerator


def main():

    manager = CheckpointManager(
        checkpoint_root="checkpoints"
    )

    generator = CandidateGenerator(
        checkpoint_manager=manager,
        k_candidates=3
    )

    print("Starting Task 3B candidate-generation test...")

    candidates = generator.generate_candidates(
        state_id=0
    )

    print("\nGenerated candidates:")

    for candidate in candidates:
        print(candidate)

    print("\nTask 3B candidate-generation test completed.")


if __name__ == "__main__":
    main()
