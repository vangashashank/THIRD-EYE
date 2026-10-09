#!/bin/bash
#SBATCH --job-name=third-eye-preflight
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/sprint1_preflight_%j.log

set -euo pipefail
test "$(hostname)" != dgx-login01
PROJECT_ROOT="${1:?Pass the server project directory}"
RESULT_DIR="$PROJECT_ROOT/runs/sprint1_preflight_$SLURM_JOB_ID"
mkdir "$RESULT_DIR"
hostname
date -u +%FT%TZ
if command -v quota >/dev/null; then
    quota -s || true
fi
df -h "$PROJECT_ROOT"
du -sx -B1 "$HOME" > "$RESULT_DIR/home-usage.txt"
cat "$RESULT_DIR/home-usage.txt"
HOME_BYTES=$(awk '{print $1}' "$RESULT_DIR/home-usage.txt")
if (( HOME_BYTES > 48000000000 )); then
    echo "Insufficient headroom under the declared 50 GB quota; preserve existing data."
    exit 3
fi
"$PROJECT_ROOT/.venv/bin/python" -m pip freeze > "$RESULT_DIR/environment.txt"
"$PROJECT_ROOT/.venv/bin/python" - "$PROJECT_ROOT" "$RESULT_DIR" <<'PY'
import importlib.util
import json
import platform
import sys
from pathlib import Path

project, result = map(Path, sys.argv[1:])
cache = project / "hf_cache/hub/models--Qwen--Qwen3-4B"
revision = (cache / "refs/main").read_text().strip()
snapshot = cache / "snapshots" / revision
weights = list(snapshot.glob("*.safetensors"))
assert weights and all(path.is_file() for path in weights), "Missing cached 4B model"
record = {
    "python": platform.python_version(),
    "cached_model_revision": revision,
    "cached_weight_files": len(weights),
    "bitsandbytes_available": importlib.util.find_spec("bitsandbytes") is not None,
    "home_usage_bytes": int((result / "home-usage.txt").read_text().split()[0]),
    "declared_quota_bytes": 50_000_000_000,
    "minimum_headroom_bytes": 2_000_000_000,
}
(result / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
echo "SPRINT 1 PREFLIGHT PASSED"
