import json
from pathlib import Path

from src.third_eye.checkpoint_manager import CheckpointManager


class StateManager:
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        metadata_file="outputs/candidate_metadata.jsonl",
    ):
        self.checkpoint_manager = checkpoint_manager

        self.metadata_file = Path(metadata_file)
        self.metadata_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def record_candidate(
        self,
        state_id: int,
        candidate_id: int,
        seed: int,
        training_loss: float,
        learning_rate: float,
        batch_size: int,
        checkpoint_path: str,
        selected: bool = False,
    ):

        record = {
            "state_id": state_id,
            "candidate_id": candidate_id,
            "seed": seed,
            "training_loss": training_loss,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "checkpoint_path": checkpoint_path,
            "selected": selected,
        }

        with open(self.metadata_file, "a") as f:
            f.write(
                json.dumps(record) + "\n"
            )

        return record

    def mark_selected(
        self,
        state_id: int,
        candidate_id: int,
    ):

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                "Candidate metadata file does not exist."
            )

        records = []

        with open(self.metadata_file, "r") as f:
            for line in f:
                record = json.loads(line)

                if record["state_id"] == state_id:
                    record["selected"] = (
                        record["candidate_id"]
                        == candidate_id
                    )

                records.append(record)

        with open(self.metadata_file, "w") as f:
            for record in records:
                f.write(
                    json.dumps(record) + "\n"
                )

    def promote_candidate(
        self,
        state_id: int,
        candidate_id: int,
    ):

        next_state_id = state_id + 1

        promoted_path = (
            self.checkpoint_manager.promote_candidate(
                current_state_id=state_id,
                candidate_id=candidate_id,
                next_state_id=next_state_id,
            )
        )

        self.mark_selected(
            state_id=state_id,
            candidate_id=candidate_id,
        )

        return promoted_path
