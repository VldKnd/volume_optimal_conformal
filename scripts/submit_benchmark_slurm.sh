#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPOSITORY_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd -P)
RUNNER="${SCRIPT_DIRECTORY}/run_experiment_slurm.sh"

usage() {
    printf 'Usage: %s [--dry-run] CONFIG_DIRECTORY\n' "$0"
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi
[[ $# -eq 1 ]] || {
    usage >&2
    exit 2
}

config_directory=$1
if [[ "$config_directory" != /* ]]; then
    config_directory="${PWD}/${config_directory}"
fi
[[ -d "$config_directory" ]] || fail "not a directory: ${config_directory}"
[[ -x "$RUNNER" ]] || fail "runner is not executable: ${RUNNER}"
if [[ "$dry_run" == false ]]; then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable"
fi

config_paths=()
while IFS= read -r config_path; do
    config_paths+=("$config_path")
done < <(
    find "$config_directory" -type f \
        \( -name '*.yaml' -o -name '*.yml' \) -print | LC_ALL=C sort
)
[[ ${#config_paths[@]} -gt 0 ]] || fail \
    "no .yaml or .yml files found below ${config_directory}"

requested_jobs=${#config_paths[@]}
if ((requested_jobs > 20)); then
    fail "found ${requested_jobs} configurations, but the cluster permits at most 20 queued plus running jobs"
fi

# The supplied cluster policy permits at most 20 queued plus running jobs.
if [[ "$dry_run" == false ]] && command -v squeue >/dev/null 2>&1; then
    current_jobs=$(
        squeue --array --noheader --user="$(id -un)" |
            wc -l |
            tr -d '[:space:]'
    )
    if ((current_jobs + requested_jobs > 20)); then
        fail "submitting ${requested_jobs} jobs with ${current_jobs} already active would exceed the cluster limit of 20"
    fi
fi

account=${MVCP_SLURM_ACCOUNT:-}
qos=${MVCP_SLURM_QOS:-}
partition=${MVCP_SLURM_PARTITION:-}
time_limit=${MVCP_SLURM_TIME_LIMIT:-3-00:00:00}
cpus_per_task=${MVCP_SLURM_CPUS_PER_TASK:-1}
max_concurrent=${MVCP_SLURM_MAX_CONCURRENT:-2}
memory=${MVCP_SLURM_MEMORY:-}
log_directory=${MVCP_SLURM_LOG_DIRECTORY:-${REPOSITORY_ROOT}/logs/slurm}
mkdir -p "$log_directory"

case "$max_concurrent" in
    ''|0|0[0-9]*|*[!0-9]*)
        fail "MVCP_SLURM_MAX_CONCURRENT must be a positive integer without leading zeros"
        ;;
esac

submitted_job_ids=()
config_index=0

for config_path in "${config_paths[@]}"; do
    model_name=$(basename "$(dirname "$config_path")")
    config_name=$(basename "${config_path%.*}")
    job_name="${model_name}-${config_name}"
    job_name=${job_name//[^[:alnum:]_.-]/-}
    job_name=${job_name:0:100}

    command=(
        sbatch
        --parsable
        --job-name="$job_name"
        --nodes=1
        --ntasks=1
        --cpus-per-task="$cpus_per_task"
        --gres=gpu:1
        --time="$time_limit"
        --chdir="$REPOSITORY_ROOT"
        --export=ALL
        --open-mode=append
        --output="${log_directory}/%x_%j.out"
        --error="${log_directory}/%x_%j.err"
    )
    [[ -z "$account" ]] || command+=(--account="$account")
    [[ -z "$qos" ]] || command+=(--qos="$qos")
    [[ -z "$partition" ]] || command+=(--partition="$partition")
    [[ -z "$memory" ]] || command+=(--mem="$memory")
    if ((config_index >= max_concurrent)); then
        dependency_job_id=${submitted_job_ids[config_index-max_concurrent]}
        command+=(--dependency="afterany:${dependency_job_id}")
    fi
    command+=("$RUNNER" "$config_path")

    if [[ "$dry_run" == true ]]; then
        printf '[dry-run]'
        printf ' %q' "${command[@]}"
        printf '\n'
        job_id="DRY_RUN_${config_index}"
    else
        job_result=$("${command[@]}")
        job_id=${job_result%%;*}
        printf 'Submitted %-12s job=%s config=%s\n' \
            "$job_name" "$job_id" "$config_path"
    fi
    submitted_job_ids+=("$job_id")
    config_index=$((config_index + 1))
done
