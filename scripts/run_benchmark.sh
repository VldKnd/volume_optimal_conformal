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
CONFIGURATION_PATH=$1

if [[ "$CONFIGURATION_PATH" != /* ]]; then
    CONFIGURATION_PATH="${PWD}/${CONFIGURATION_PATH}"
fi

[[ -e "$CONFIGURATION_PATH" ]] || {
    printf 'error: configuration path not found: %s\n' "$CONFIGURATION_PATH" >&2
    exit 2
}
if [[ -n "${MVCP_PYTHON_EXECUTABLE:-}" ]]; then
    [[ -x "$MVCP_PYTHON_EXECUTABLE" ]] || {
        printf 'error: Python executable not found: %s\n' \
            "$MVCP_PYTHON_EXECUTABLE" >&2
        exit 2
    }
    PYTHON_COMMAND=("$MVCP_PYTHON_EXECUTABLE")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_COMMAND=(uv run python3)
elif [[ -x "${REPOSITORY_ROOT}/.venv/bin/python" ]]; then
    PYTHON_COMMAND=("${REPOSITORY_ROOT}/.venv/bin/python")
else
    printf "error: neither 'uv' nor the project Python is available.\n" >&2
    exit 2
fi

cd "$REPOSITORY_ROOT"
export PYTHONUNBUFFERED=1
exec "${PYTHON_COMMAND[@]}" \
    "${SCRIPT_DIRECTORY}/run_benchmark.py" \
    "$CONFIGURATION_PATH" \
    --priority-method neural_ot_l2 \
    --priority-method realnvp_l2
