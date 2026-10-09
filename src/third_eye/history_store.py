import json
import math
from pathlib import Path


METRICS = ("target", "ood", "retention")


class HistoryStore:
    """One file per trajectory, containing applied transitions only."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self):
        if not self.path.exists():
            return []

        with self.path.open() as file:
            return [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    @staticmethod
    def _validate_losses(losses):
        if set(losses) != set(METRICS):
            raise ValueError("Expected target, ood, and retention losses.")

        for value in losses.values():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Losses must be finite and nonnegative.")

    def record_applied_transition(
        self,
        *,
        from_state,
        selected_candidate_id,
        before_losses,
        after_losses,
    ):
        self._validate_losses(before_losses)
        self._validate_losses(after_losses)

        if type(from_state) is not int or from_state < 0:
            raise ValueError("Invalid state ID.")

        if (
            type(selected_candidate_id) is not int
            or selected_candidate_id not in (0, 1, 2)
        ):
            raise ValueError("Expected candidate ID 0, 1, or 2.")

        records = self._read()

        # Require a complete, ordered trajectory starting at state 0.
        if from_state != len(records):
            raise ValueError("Transition is duplicated or out of order.")

        if records and records[-1]["after_losses"] != before_losses:
            raise ValueError(
                "Before-losses must match the previous state's saved losses."
            )

        record = {
            "from_state": from_state,
            "to_state": from_state + 1,
            "selected_candidate_id": selected_candidate_id,
            "before_losses": before_losses,
            "after_losses": after_losses,
            "loss_reductions": [
                before_losses[key] - after_losses[key]
                for key in METRICS
            ],
        }

        with self.path.open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

        return record

    def features_before(self, state_id):
        if type(state_id) is not int or state_id < 0:
            raise ValueError("Invalid state ID.")

        records = self._read()

        if state_id > len(records):
            raise ValueError("History is incomplete for this state.")

        # Exclude all transitions occurring after the requested state.
        return [
            record["loss_reductions"]
            for record in records[:state_id]
        ]
