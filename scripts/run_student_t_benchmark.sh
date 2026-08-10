#!/usr/bin/env bash

#SBATCH --job-name=student-t-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

set -euo pipefail

[[ $# -eq 0 ]] || {
    printf 'Usage: sbatch %s\n' "$0" >&2
    exit 2
}

SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPOSITORY_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)
PYTHON_EXECUTABLE=${MVCP_PYTHON_EXECUTABLE:-${REPOSITORY_ROOT}/.venv/bin/python}
CONFIGURATION_ROOT=${MVCP_STUDENT_T_CONFIG_ROOT:-${REPOSITORY_ROOT}/benchmark/configurations/student_t}

[[ -d "$CONFIGURATION_ROOT" ]] || {
    printf 'error: configuration directory not found: %s\n' \
        "$CONFIGURATION_ROOT" >&2
    exit 2
}
[[ -x "$PYTHON_EXECUTABLE" ]] || {
    printf "error: run 'uv sync --frozen --extra tracking' before submitting jobs.\n" >&2
    exit 2
}

printf '[Student-t benchmark] Sequentially running configurations below %s\n' \
    "$CONFIGURATION_ROOT"

cd "$REPOSITORY_ROOT"
export PYTHONUNBUFFERED=1
exec "$PYTHON_EXECUTABLE" \
    "${REPOSITORY_ROOT}/scripts/run_student_t_benchmark.py" \
    "$CONFIGURATION_ROOT"
