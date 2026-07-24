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
