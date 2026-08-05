#!/bin/bash
#SBATCH --job-name=run_benchmark
#SBATCH --gres=gpu:1
#SBATCH --account=eric
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 <configuration_path>" >&2
    exit 2
fi

uv run python3 scripts/run_benchmark.py "$1"