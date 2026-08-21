#!/usr/bin/env bash
#SBATCH --job-name=run_benchmark
#SBATCH --gres=gpu:1
#SBATCH --account=eric
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: sbatch %s CONFIGURATION_PATH\n' "$0" >&2
    exit 2
fi

SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPOSITORY_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)
PYTHON_EXECUTABLE=${MVCP_PYTHON_EXECUTABLE:-${REPOSITORY_ROOT}/.venv/bin/python}
CONFIGURATION_PATH=$1

if [[ "$CONFIGURATION_PATH" != /* ]]; then
    CONFIGURATION_PATH="${PWD}/${CONFIGURATION_PATH}"
fi

[[ -e "$CONFIGURATION_PATH" ]] || {
    printf 'error: configuration path not found: %s\n' "$CONFIGURATION_PATH" >&2
    exit 2
}
[[ -x "$PYTHON_EXECUTABLE" ]] || {
    printf "error: run 'uv sync --frozen --extra tracking' before submitting.\n" >&2
    exit 2
}

cd "$REPOSITORY_ROOT"
export PYTHONUNBUFFERED=1
exec "$PYTHON_EXECUTABLE" \
    "${SCRIPT_DIRECTORY}/run_benchmark.py" \
    "$CONFIGURATION_PATH" \
    --priority-method neural_ot_l2 \
    --priority-method realnvp_l2
