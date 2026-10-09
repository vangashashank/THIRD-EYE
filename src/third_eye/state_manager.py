import hashlib
import math
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
    def record_proposal(
        self,
        *,
        run_id,
        trajectory_id,
        state_id,
        candidate_id,
        parent_checkpoint,
        seed,
        learning_rate,
        batch_size,
        epochs,
        max_length,
        texts,
        parent_weights_sha256=None,
    ):
        """Record the proposed update before candidate training."""
        if not run_id or not trajectory_id:
            raise ValueError("Run and trajectory IDs are required.")

        for name, value in (
            ("state_id", state_id),
            ("candidate_id", candidate_id),
            ("seed", seed),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid {name}.")

        for name, value in (
            ("batch_size", batch_size),
            ("epochs", epochs),
            ("max_length", max_length),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid {name}.")

        if (
            not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("Invalid learning rate.")

        if (
            not isinstance(texts, (list, tuple))
            or not texts
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            raise ValueError("Expected a nonempty sequence of texts.")

        serialized_texts = json.dumps(
            list(texts), ensure_ascii=False
        ).encode("utf-8")

        record = {
            "run_id": str(run_id),
            "trajectory_id": str(trajectory_id),
            "state_id": state_id,
            "candidate_id": candidate_id,
            "parent_checkpoint": str(parent_checkpoint),
            "seed": seed,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_length": max_length,
            "num_examples": len(texts),
            "planned_optimizer_steps": (
                (len(texts) + batch_size - 1) // batch_size
            ) * epochs,
            "data_sha256": hashlib.sha256(serialized_texts).hexdigest(),
            "record_type": "pre_update_proposal",
        }
        if parent_weights_sha256 is not None:
            record["parent_weights_sha256"] = parent_weights_sha256

        path = self.metadata_file.with_name("candidate_proposals.jsonl")
        identity_keys = (
            "run_id", "trajectory_id", "state_id", "candidate_id"
        )

        if path.exists():
            with path.open() as file:
                for line in file:
                    if not line.strip():
                        continue
                    previous = json.loads(line)
                    if all(
                        previous[key] == record[key]
                        for key in identity_keys
                    ):
                        raise ValueError("Proposal already recorded.")

        with path.open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

        return record

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
        actual_training=None,
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
        if actual_training is not None:
            record["actual_training"] = actual_training

        with open(self.metadata_file, "a") as f:
            f.write(
                json.dumps(record, allow_nan=False) + "\n"
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
