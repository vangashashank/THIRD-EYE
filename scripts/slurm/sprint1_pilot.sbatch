#!/bin/bash
#SBATCH --job-name=third-eye-sprint1-nf4
#SBATCH --partition=gpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/sprint1_pilot_%j.log

set -euo pipefail
test "$(hostname)" != dgx-login01
PROJECT_ROOT="${1:?Pass the server project directory}"
SOURCE_COMMIT="${2:?Pass the immutable source commit}"
SOURCE_ROOT="$PROJECT_ROOT/.code-snapshots/$SOURCE_COMMIT"
test -f "$SOURCE_ROOT/scripts/run_sprint1_pilot.py"
RESULT_ROOT="$PROJECT_ROOT/runs/sprint1_qlora_$SLURM_JOB_ID"
mkdir "$RESULT_ROOT"
date -u +%FT%TZ
hostname
du -sx -B1 "$HOME" > "$RESULT_ROOT/home-usage-before.txt"
HOME_BYTES=$(awk '{print $1}' "$RESULT_ROOT/home-usage-before.txt")
cat "$RESULT_ROOT/home-usage-before.txt"
if (( HOME_BYTES > 48000000000 )); then
    echo "Insufficient headroom under the 50 GB quota; preserving existing files."
    exit 3
fi
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
OVERLAY="$PROJECT_ROOT/.runtime-deps/bitsandbytes-0.46.1"
if ! test -f "$OVERLAY/.installed"; then
    mkdir -p "$OVERLAY"
    "$PROJECT_ROOT/.venv/bin/python" -m pip install --no-deps --no-cache-dir \
        --target "$OVERLAY" -r "$SOURCE_ROOT/requirements.qlora.txt"
    touch "$OVERLAY/.installed"
fi
export PYTHONPATH="$OVERLAY:$SOURCE_ROOT"
"$PROJECT_ROOT/.venv/bin/python" -m pip freeze > "$RESULT_ROOT/environment.txt"
nvidia-smi > "$RESULT_ROOT/nvidia-smi.txt"
cp "$SOURCE_ROOT/configs/qwen3_4b_qlora.yaml" "$RESULT_ROOT/run_config.yaml"
cd "$SOURCE_ROOT"
"$PROJECT_ROOT/.venv/bin/python" -u -m scripts.run_sprint1_pilot \
    --config configs/qwen3_4b_qlora.yaml --run-root "$RESULT_ROOT" --source-commit "$SOURCE_COMMIT"
du -sx -B1 "$HOME" > "$RESULT_ROOT/home-usage-after.txt"
cat "$RESULT_ROOT/home-usage-after.txt"
date -u +%FT%TZ
