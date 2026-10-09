from pathlib import Path
import shutil


class CheckpointManager:
    def __init__(self, checkpoint_root="checkpoints"):
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

    def create_state_dir(self, state_id: int):
        state_dir = self.checkpoint_root / f"state_{state_id:03d}"
        state_dir.mkdir(parents=True, exist_ok=True)

        (state_dir / "candidates").mkdir(exist_ok=True)

        return state_dir

    def save_parent(self, source_checkpoint, state_id: int):
        source = Path(source_checkpoint)
        state_dir = self.create_state_dir(state_id)

        destination = state_dir / "parent"

        shutil.copytree(source, destination)

        return destination

    def save_candidate(self, source_checkpoint, state_id: int, candidate_id: int):
        source = Path(source_checkpoint)
        state_dir = self.create_state_dir(state_id)

        destination = (
            state_dir
            / "candidates"
            / f"candidate_{candidate_id}"
        )

        shutil.copytree(source, destination)

        return destination

    def get_parent(self, state_id: int):
        parent = (
            self.checkpoint_root
            / f"state_{state_id:03d}"
            / "parent"
        )

        if not parent.exists():
            raise FileNotFoundError(
                f"Parent checkpoint not found: {parent}"
            )

        return parent

    def get_candidate(self, state_id: int, candidate_id: int):
        candidate = (
            self.checkpoint_root
            / f"state_{state_id:03d}"
            / "candidates"
            / f"candidate_{candidate_id}"
        )

        if not candidate.exists():
            raise FileNotFoundError(
                f"Candidate checkpoint not found: {candidate}"
            )

        return candidate

    def rollback_to_parent(self, state_id: int):
        return self.get_parent(state_id)

    def promote_candidate(
        self,
        current_state_id: int,
        candidate_id: int,
        next_state_id: int
    ):
        candidate = self.get_candidate(
            current_state_id,
            candidate_id
        )

        next_state_dir = self.create_state_dir(next_state_id)
        new_parent = next_state_dir / "parent"

        shutil.copytree(candidate, new_parent)

        return new_parent
