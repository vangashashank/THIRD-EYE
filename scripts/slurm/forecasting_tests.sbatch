#!/bin/bash
#SBATCH --job-name=third-eye-forecast-tests
#SBATCH --partition=cpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/forecasting_tests_%j.log

# CPU only: the forecasting code is tiny and needs no GPU.
# Usage: sbatch scripts/slurm/forecasting_tests.sbatch /path/to/THIRD-EYE
set -euo pipefail
test "$(hostname)" != dgx-login01
PROJECT_ROOT="${1:?Pass the server project directory}"
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" -B -m unittest \
    tests.test_forecasting_core tests.test_forecasting_models tests.test_feature_contract
echo "FORECASTING TESTS PASSED"
