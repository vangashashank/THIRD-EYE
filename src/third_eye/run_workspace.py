import re
from pathlib import Path


def create_trajectory_workspace(
    run_id,
    trajectory_id,
    root="outputs/experiments",
):
    """Create a fresh workspace; refuse to overwrite an existing one."""
    for name, value in (
        ("run_id", run_id),
        ("trajectory_id", trajectory_id),
    ):
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None
        ):
            raise ValueError(
                f"{name} must contain only letters, numbers, "
                "underscores, or hyphens, starting with a letter or number."
            )

    workspace = Path(root) / run_id / trajectory_id

    # Existing trajectories require an explicit resume workflow.
    workspace.mkdir(parents=True, exist_ok=False)

    paths = {"root": workspace}

    for name in (
        "checkpoints",
        "candidate_work",
        "trajectory_cache",
        "metadata",
        "forecasting",
    ):
        paths[name] = workspace / name
        paths[name].mkdir()

    return paths
