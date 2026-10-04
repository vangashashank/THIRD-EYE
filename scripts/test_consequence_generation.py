from pathlib import Path
from src.third_eye.consequence_generator import (
    ConsequenceGenerator
)


def main():

    output_file = Path(
        "outputs/consequence_labels.jsonl"
 )

    if output_file.exists():
        output_file.unlink()


    generator = ConsequenceGenerator()

    # Tiny training data used only
    # for the local trajectory test.
    train_texts = [
        "Question: What is 4 + 5?\nAnswer: 9",
        "Question: What is 8 - 3?\nAnswer: 5",
    ]

    # Keep validation examples separate.
    validation_texts = [
        "Question: What is 6 + 2?\nAnswer: 8",
        "Question: What is 9 - 4?\nAnswer: 5",
    ]

    for candidate_id in range(3):

        candidate_path = (
            "checkpoints/"
            "state_000/"
            "candidates/"
            f"candidate_{candidate_id}"
        )

        result = (
            generator.generate_for_candidate(
                state_id=0,
                candidate_id=candidate_id,
                candidate_path=candidate_path,

                train_texts=train_texts,

                validation_texts=
                    validation_texts,

                epochs=1,
                batch_size=1,
                learning_rate=1e-4,
                max_length=128,
            )
        )

        print("\nConsequence label:")
        print(result)

    print(
        "\nTask 3D H=2 consequence "
        "generation completed."
    )


if __name__ == "__main__":
    main()
