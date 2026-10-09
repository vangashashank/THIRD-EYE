import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.third_eye.run_workspace import create_trajectory_workspace
from src.third_eye.checkpoint_manager import CheckpointManager
from src.third_eye.state_manager import StateManager
from src.third_eye.candidate_generator import CandidateGenerator
from src.third_eye.consequence_generator import ConsequenceGenerator


def main():
    with TemporaryDirectory() as temporary_root:
        proposal_files = []

        for trajectory_id in ("trajectory_000", "trajectory_001"):
            paths = create_trajectory_workspace(
                run_id="isolation_test",
                trajectory_id=trajectory_id,
                root=temporary_root,
            )

            checkpoints = CheckpointManager(
                checkpoint_root=paths["checkpoints"],
            )

            metadata_file = paths["metadata"] / "candidate_metadata.jsonl"

            candidates = CandidateGenerator(
                checkpoint_manager=checkpoints,
                metadata_file=metadata_file,
            )

            consequences = ConsequenceGenerator(
                output_file=paths["metadata"] / "consequence_labels.jsonl",
                trajectory_root=paths["trajectory_cache"],
            )

            states = StateManager(
                checkpoint_manager=checkpoints,
                metadata_file=candidates.metadata_file,
            )

            # Test metadata only: no real adapter is required here.
            proposal = {
                "run_id": "isolation_test",
                "trajectory_id": trajectory_id,
                "state_id": 0,
                "candidate_id": 0,
                "parent_checkpoint": paths["checkpoints"] / "state_000/parent",
                "seed": 11,
                "learning_rate": 1e-4,
                "batch_size": 1,
                "epochs": 1,
                "max_length": 128,
                "texts": ["Metadata isolation test only."],
            }

            states.record_proposal(**proposal)

            # An accidental duplicate must not append another row.
            try:
                states.record_proposal(**proposal)
            except ValueError as error:
                assert str(error) == "Proposal already recorded."
            else:
                raise AssertionError("Duplicate proposal was accepted.")

            proposal_files.append(
                Path(metadata_file).with_name("candidate_proposals.jsonl")
            )

            assert consequences.trajectory_root == paths["trajectory_cache"]

        assert proposal_files[0] != proposal_files[1]

        for index, path in enumerate(proposal_files):
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]

            assert len(rows) == 1
            assert rows[0]["trajectory_id"] == f"trajectory_{index:03d}"

        # Reusing an existing workspace must fail.
        try:
            create_trajectory_workspace(
                run_id="isolation_test",
                trajectory_id="trajectory_000",
                root=temporary_root,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing workspace was accepted.")

    print("Trajectory isolation test passed.")
    print("Separate proposal files and duplicate protection verified.")
    print("Temporary test files removed; no model loaded or trained.")


if __name__ == "__main__":
    main()
