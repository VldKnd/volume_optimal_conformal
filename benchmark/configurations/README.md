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
    "comparisont_with_residuals/"
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
  benchmark/configurations/scm20d/comparisont_with_residuals/transport_realnvp_l2
```

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

For disconnected machines, select `--wandb-mode offline` and later run
`uv run --extra tracking wandb sync <run-directory>/wandb` (or pass the exact
`offline-run-*` directory). W&B credentials should be supplied by
`wandb login` or `WANDB_API_KEY`, never committed to a YAML file.

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
