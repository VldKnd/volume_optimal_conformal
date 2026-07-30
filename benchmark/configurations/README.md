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
