#!/usr/bin/env bash
#SBATCH --job-name=gaussian_picnn
#SBATCH --gres=shard:1
#SBATCH --account=eric
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: sbatch %s CONFIGURATION_PATH\n' "$0" >&2
    exit 2
fi

if [[ -n "${MVCP_REPOSITORY_ROOT:-}" ]]; then
    REPOSITORY_ROOT=$(cd -- "$MVCP_REPOSITORY_ROOT" && pwd -P)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPOSITORY_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)
else
    SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    REPOSITORY_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)
fi

RUNNER="${REPOSITORY_ROOT}/scripts/run_synthetic_benchmark.sh"
[[ -x "$RUNNER" ]] || {
    printf 'error: synthetic benchmark runner not found: %s\n' "$RUNNER" >&2
    exit 2
}

exec "$RUNNER" "$1"
