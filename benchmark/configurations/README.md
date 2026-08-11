# Benchmark configurations

The SCM20D benchmark contains one YAML file per method and random seed:

- `residual_rf_global_otcp`
- `residual_rf_local_otcp`
- `residual_rf_elliptic`
- `transport_neural_ot_l2`
- `transport_neural_ot_rearranged_l2`
- `transport_realnvp_l2`
- `transport_realnvp_rearranged_l2`
- `transport_cpflow_l2`
- `transport_cpflow_rearranged_l2`

Each method has ten configurations using seeds `0` through `9`. A single
configuration can be run from Python with:

```python
from experimentation import ExperimentRunner, load_experiment_config

config = load_experiment_config(
    "benchmark/configurations/scm20d/"
    "base_run/"
    "residual_rf_global_otcp/seed_00.yaml"
)
ExperimentRunner(config).run()
```

Run all configurations sequentially from the repository root with:

```bash
uv run python scripts/run_benchmark.py \
  benchmark/configurations/scm20d
```

To run only one method, pass its configuration directory:

```bash
uv run python scripts/run_benchmark.py \
  benchmark/configurations/scm20d/base_run/transport_realnvp_l2
```

A single YAML file can also be selected directly:

```bash
uv run python scripts/run_benchmark.py \
  benchmark/configurations/scm20d/base_run/transport_realnvp_l2/seed_00.yaml
```

## Running a configuration suite on Slurm

Prepare the locked environment once on the login node. Keep the repository on
the cluster's persistent `/shared/home` storage:

```bash
uv sync --frozen --extra tracking
```

The submitter recursively finds every `.yaml` and `.yml` below the path and
submits one independent one-GPU job for each configuration. Set the account
once, then only the configuration directory is needed:

```bash
export MVCP_SLURM_ACCOUNT=nils  # set this once for your group
scripts/submit_benchmark_slurm.sh \
  benchmark/configurations/scm20d/continuing_only_rearranged_realnvp/transport_realnvp_rearranged_l2
```

All jobs are submitted immediately. By default, dependency chains allow at
most two jobs from the submission to run concurrently, while the remaining
jobs stay pending. Change that limit when needed with:

```bash
MVCP_SLURM_MAX_CONCURRENT=4 \
  scripts/submit_benchmark_slurm.sh \
  benchmark/configurations/scm20d/continuing_only_rearranged_realnvp/transport_realnvp_rearranged_l2
```

The worker uses the W&B settings from each YAML file without overriding them,
so configurations with `wandb.mode: online` are tracked live. Optional
environment variables include `MVCP_SLURM_QOS`, `MVCP_SLURM_PARTITION`,
`MVCP_SLURM_TIME_LIMIT`, `MVCP_SLURM_CPUS_PER_TASK`, and
`MVCP_SLURM_MEMORY`. Preview every `sbatch` command with:

```bash
scripts/submit_benchmark_slurm.sh --dry-run \
  benchmark/configurations/scm20d/continuing_only_rearranged_realnvp/transport_realnvp_rearranged_l2
```

Every configuration now receives its own time limit, exit status, log files,
and preemption/requeue lifecycle. Logs are written to `logs/slurm/`. The
cluster permits at most 20 queued plus running jobs per user, including jobs
that are pending on dependencies; the submitter checks this before launching.
Pending time does not consume the job's runtime limit. Ensure every selected
configuration has a unique `save_directory` and `name`, and do not submit
overlapping suites concurrently, or their result files can collide.

## Live W&B tracking

Install the optional integration and authenticate before selecting online
mode:

```bash
uv sync --extra tracking
uv run --extra tracking wandb login
```

Tracking is disabled by default. It can be enabled for all YAML files selected
by a benchmark command without modifying those files:

```bash
uv run --extra tracking python scripts/run_benchmark.py \
  benchmark/configurations/scm20d/continuing_only_realnvp/transport_realnvp_rearranged_l2 \
  --wandb-mode online \
  --wandb-project minimal-volume-conformal-prediction \
  --wandb-group scm20d/transport_realnvp_rearranged_l2 \
  --wandb-tags scm20d realnvp rearranged dopri5
```

Available overrides are `--wandb-mode`, `--wandb-project`,
`--wandb-entity`, `--wandb-group`, `--wandb-name`, and `--wandb-tags`. An
override applies to every selected configuration; prefer each YAML file's
existing run name when selecting multiple seeds. The configuration file path
is also recorded with the W&B run for provenance. When no group is set, the
runner derives one from the configuration's relative parent directory, such as
`scm20d/<suite>/<method>`, which keeps otherwise identical seed names apart.
The rearrangement stage logs lightweight solver NFE and adaptive-step counters
by default; set `wandb.log_solver_diagnostics: false` in YAML if only loss and
optimizer metrics are wanted.

W&B credentials should be supplied by `wandb login` or `WANDB_API_KEY`, never
committed to a YAML file.

Each run writes its configuration, model and trainer checkpoints, histories,
conformal checkpoint, and `metrics.json` under:

```text
benchmark/results/scm20d/<method>/seed_<seed>/
```

`metrics.json` contains marginal coverage, worst-slab coverage, excess
coverage risk, and log-volume-per-dimension summaries.

Volume estimation is the expensive part of these configurations, especially
for neural OT and CPflow. Reduce `volume_mc_samples` for quick smoke runs
before launching the full benchmark.

## Student-t benchmark

The Student-t suite contains 160 configurations below
`benchmark/configurations/student_t`: four target dimensions, four values of
`k`, five seeds, and both base and amortized-rearranged Neural OT runs. All
targets use `nu=3` and the determinant-one diagonal scale matrix implemented by
`StudentTDataset`.

Generate or refresh the suite with:

```bash
uv run python scripts/generate_student_t_benchmark_configs.py
```

Run the complete suite sequentially with the dedicated runner so that the
analytic Student-t HDR comparison is included in `metrics.json`:

```bash
uv run python scripts/run_student_t_benchmark.py \
  benchmark/configurations/student_t
```

The base family sorts before the rearranged family within each parameter
setting. Therefore, sequential execution creates each Neural OT checkpoint
before the matching rearranged configuration loads it. When scheduling runs
independently, complete the `transport_neural_ot_l2` configurations before
submitting their corresponding `transport_neural_ot_rearranged_l2`
configurations.

The supplied Slurm script creates one GPU job and sequentially processes all
160 configurations. The recursively sorted configuration paths run every base
family before the corresponding rearranged family, preserving the checkpoint
dependency:

```bash
sbatch scripts/run_student_t_benchmark.sh \
  benchmark/configurations/student_t
```

Override the partition or time limit through normal `sbatch` options, for
example:

```bash
sbatch --partition=GPU --time=4-00:00:00 \
  scripts/run_student_t_benchmark.sh \
  benchmark/configurations/student_t
```
