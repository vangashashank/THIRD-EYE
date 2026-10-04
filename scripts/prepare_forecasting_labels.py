import json
import math
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path("outputs/consequence_labels.jsonl")
OUTPUT_PATH = Path("outputs/forecasting/loss_labels.jsonl")


def main():
    states = defaultdict(dict)

    with INPUT_PATH.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            state_id = record["state_id"]
            candidate_id = record["candidate_id"]

            if type(state_id) is not int or state_id < 0:
                raise ValueError(
                    f"Line {line_number}: invalid state_id."
                )

            if type(candidate_id) is not int or candidate_id not in (0, 1, 2):
                raise ValueError(
                    f"Line {line_number}: expected candidate_id 0, 1, or 2."
                )

            if record["horizon"] != 2:
                raise ValueError(
                    f"Line {line_number}: expected horizon=2."
                )

            for key in ("t1_validation_loss", "t2_validation_loss"):
                value = record[key]

                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        f"Line {line_number}: invalid {key}."
                    )

            if candidate_id in states[state_id]:
                raise ValueError(
                    f"Duplicate state={state_id}, candidate={candidate_id}. "
                    "Check whether multiple runs were appended."
                )

            states[state_id][candidate_id] = record

    if not states:
        raise ValueError("No consequence records found.")

    prepared = []

    for state_id, candidates in sorted(states.items()):
        if set(candidates) != {0, 1, 2}:
            raise ValueError(
                f"State {state_id}: expected all three candidates."
            )

        losses = [
            record["t2_validation_loss"]
            for record in candidates.values()
        ]

        best_loss = min(losses)

        for candidate_id, record in sorted(candidates.items()):
            t1_loss = record["t1_validation_loss"]
            t2_loss = record["t2_validation_loss"]

            # Equal losses receive equal ranks.
            rank = 1 + sum(loss < t2_loss for loss in losses)

            prepared.append({
                "state_id": state_id,
                "candidate_id": candidate_id,
                "horizon": 2,
                "candidate_checkpoint": record["candidate_checkpoint"],
                "labels": {
                    "t1_validation_loss": t1_loss,
                    "t2_validation_loss": t2_loss,
                    "continuation_loss_reduction": t1_loss - t2_loss,
                    "t2_rank": rank,
                    "t2_loss_regret": t2_loss - best_loss,
                },
            })

        winners = [
            candidate_id
            for candidate_id, record in sorted(candidates.items())
            if record["t2_validation_loss"] == best_loss
        ]

        print(f"State {state_id}: best t+2 candidate(s) = {winners}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as file:
        for record in prepared:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    print(f"Saved {len(prepared)} labels from {len(states)} state(s).")
    print(f"Output: {OUTPUT_PATH}")

    if len(states) < 2:
        print(
            "Smoke-test labels only: more independent states are "
            "needed for separate training and evaluation."
        )


if __name__ == "__main__":
    main()
