#!/bin/bash
#SBATCH --job-name=third-eye-forecast-eval
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/forecasting_eval_%j.log

# CPU only.  Usage:
#   sbatch scripts/slurm/forecasting_eval.sbatch PROJECT_ROOT RECORDS.jsonl [extra run_forecasting_eval args...]
# Validate the records first (instant):  ... run_forecasting_eval --records X --validate-only
set -euo pipefail
test "$(hostname)" != dgx-login01
PROJECT_ROOT="${1:?Pass the server project directory}"
RECORDS="${2:?Pass the meta-records JSONL}"
shift 2
cd "$PROJECT_ROOT"
OUT="$PROJECT_ROOT/runs/forecasting_eval_$SLURM_JOB_ID"
mkdir "$OUT"
"$PROJECT_ROOT/.venv/bin/python" -m scripts.run_forecasting_eval \
    --records "$RECORDS" --out "$OUT" "$@"
echo "FORECASTING EVAL DONE: $OUT"
