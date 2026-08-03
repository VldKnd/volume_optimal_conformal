# Minimal Volume Conformal Prediction

This repository is a modular benchmark for multivariate conformal prediction.
The goal is to compare how different multivariate score constructions and
calibrators affect coverage and prediction-region efficiency, especially
region volume.

## Setup

```bash
uv sync
export PYTHONPATH=src
```

Python 3.11+ is expected. Sandbox notebooks live in `notebooks/sandbox/`.

## Live experiment tracking

Weights & Biases tracking is optional and disabled by default. Install the
tracking dependency and authenticate once before using online mode:

```bash
uv sync --extra tracking
uv run --extra tracking wandb login
```

Alternatively, set `WANDB_API_KEY` in the environment. Do not store API keys
in experiment YAML files. Enable live tracking in an experiment configuration
with:

```yaml
wandb:
  mode: online
  project: minimal-volume-conformal-prediction
  entity: null
  group: scm20d/transport_realnvp_rearranged_l2
  name: seed_03
  tags:
    - scm20d
    - realnvp
    - dopri5
  log_every_n_steps: 20
  log_solver_diagnostics: true
```

Batch metrics are sent every `log_every_n_steps`, while epoch summaries and
final evaluation metrics are sent as soon as they are available. Base
transport and rearrangement training use separate step axes in the same W&B
run. Rearrangement batches also report pre-clipping gradient norm, elapsed
batch time, ODE function evaluations and adaptive-step counts, the learned
output `tanh` scale, and CUDA memory when available. Solver counters are
enabled only during rearrangement training and can be disabled with
`log_solver_diagnostics: false`. Set `mode: offline` to collect a run without
network access for a later `wandb sync`, or leave `mode: disabled` to run
without importing W&B. Initialization and authentication errors fail before
training starts; a later logging-service failure emits a warning and lets the
training run continue.

The benchmark command also provides temporary overrides, so an entire folder
can be tracked without editing its YAML files:

```bash
uv run --extra tracking python scripts/run_benchmark.py \
  benchmark/configurations/scm20d/continuing_only_realnvp/transport_realnvp_rearranged_l2 \
  --wandb-mode online \
  --wandb-project minimal-volume-conformal-prediction \
  --wandb-group scm20d/transport_realnvp_rearranged_l2 \
  --wandb-tags scm20d realnvp rearranged
```

Command-line settings override YAML only for the current invocation. Avoid
`--wandb-name` when selecting multiple configurations, since it deliberately
assigns the same display name to every selected run.

## Repository Structure

- `src/data/datasets/`
  Dataset interfaces and implementations. `XYData` stores `(x, y)` tensors,
  `DatasetSplits` stores train/calibration/test splits, and `BaseDataset`
  defines `prepare()`, `get_splits()`, `x_dim`, and `y_dim`.
- `src/data/datasets/synthetic/`
  Synthetic conditional datasets for experiments:
  Gaussian, banana-shaped, and Student-t targets. They implement sampling,
  splitting, and optional oracle densities or maps when available.
- `src/data/loaders.py`
  Converts `XYData` splits into PyTorch `TensorDataset` / `DataLoader` objects.
- `src/predictors/`
  Predictor interfaces. A predictor maps `(x, y)` to a multivariate score
  `z in R^{y_dim}` through `multivariate_score(x, y)`.
- `src/predictors/transport/`
  Transport predictors with `pushforward(x, u)` and `pullback(x, y)`.
  `FlowMatchingPredictor` uses the pullback as its multivariate score.
- `src/trainers/`
  Optimization logic separated from predictor definitions.
  `FlowMatchingTrainer` fits `FlowMatchingPredictor`.
- `src/conformal/`
  `TransportBasedConformalPredictor` wraps a trained transport predictor,
  constructs the configured calibrator, exposes calibrated containment checks,
  and estimates prediction-region volume from the forward-map Jacobian.
- `src/conformal/calibrators/`
  Scalarization plus conformal thresholding. Current calibrators include
  norm-based, local elliptic/Mahalanobis, and analytic Gaussian-baseline
  calibration.
- `src/configs/`
  Pydantic config objects for datasets, predictors, trainers, and calibrators.

## Intended Pipeline

1. Build a dataset config and dataset.
2. Call `dataset.get_splits()` to obtain train, calibration, and test data.
3. Convert the train and calibration splits with `make_xy_dataloader(...)`.
4. Build a predictor, for example `FlowMatchingPredictor`.
5. Train it with the matching trainer, for example `FlowMatchingTrainer`.
6. Wrap the trained predictor with `TransportBasedConformalPredictor`,
   supplying a `TransportBasedConformalPredictorConfig` containing the desired
   `coverage_mass` and calibrator config.
7. Call `conformal_predictor.fit(calibration_dataloader)` to compute pullback
   scores batch-by-batch and calibrate the region.
8. Check test inclusion with
   `conformal_predictor.contains(x_test, y_test)`.
9. Estimate target-space region volumes, when the calibrator defines a
   Euclidean latent ball, with `conformal_predictor.volume(x_test)`.
10. Report coverage, region volume, and runtime.

## Score And Calibration Convention

Transport predictors use

```python
z = T_x^{-1}(y)
```

Residual predictors, when added, should use

```python
z = y - f(x)
```

Calibrators then map `z` to a scalar score and apply the split-conformal
finite-sample order statistic from
`src/conformal/calibrators/quantile.py`. Coverage is consistently expressed as
`coverage_mass` throughout the conformal API.
