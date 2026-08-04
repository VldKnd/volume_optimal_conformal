#!/usr/bin/env bash

set -euo pipefail

[[ $# -eq 1 ]] || {
    printf 'Usage: %s CONFIG_FILE\n' "$0" >&2
    exit 2
}

SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPOSITORY_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)
PYTHON_EXECUTABLE=${MVCP_PYTHON_EXECUTABLE:-${REPOSITORY_ROOT}/.venv/bin/python}
CONFIG_PATH=$1

[[ -f "$CONFIG_PATH" ]] || {
    printf 'error: configuration file not found: %s\n' "$CONFIG_PATH" >&2
    exit 2
}
[[ -x "$PYTHON_EXECUTABLE" ]] || {
    printf "error: run 'uv sync --frozen --extra tracking' before submitting jobs.\n" >&2
    exit 2
}

cd "$REPOSITORY_ROOT"
export PYTHONUNBUFFERED=1
exec "$PYTHON_EXECUTABLE" \
    "${REPOSITORY_ROOT}/scripts/run_benchmark.py" \
    "$CONFIG_PATH"
