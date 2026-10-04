import hashlib
from pathlib import Path


def file_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def main():

    root = Path(
        "checkpoints/state_000/candidates"
    )

    hashes = {}

    for candidate_id in range(3):

        model_path = (
            root
            / f"candidate_{candidate_id}"
            / "adapter_model.safetensors"
        )

        hash_value = file_hash(model_path)

        hashes[candidate_id] = hash_value

        print(
            f"Candidate {candidate_id}: "
            f"{hash_value}"
        )

    print("\nComparison:")

    if len(set(hashes.values())) == 3:
        print(
            "SUCCESS: All 3 candidates "
            "have different adapter weights."
        )
    else:
        print(
            "WARNING: Some candidate weights "
            "are identical."
        )


if __name__ == "__main__":
    main()
