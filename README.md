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
- `src/calibrators/`
  Scalarization plus conformal thresholding. Current calibrators include
  norm-based and local elliptic/Mahalanobis calibration.
- `src/configs/`
  Pydantic config objects for datasets, predictors, trainers, and calibrators.

## Intended Pipeline

1. Build a dataset config and dataset.
2. Call `dataset.get_splits()` to obtain train, calibration, and test data.
3. Convert the train split with `make_xy_dataloader(...)`.
4. Build a predictor, for example `FlowMatchingPredictor`.
5. Train it with the matching trainer, for example `FlowMatchingTrainer`.
6. Compute calibration scores:
   `z_cal = predictor.multivariate_score(x_cal, y_cal)`.
7. Fit a calibrator with `calibrator.fit(x_cal, z_cal, alpha)`.
8. Compute test scores:
   `z_test = predictor.multivariate_score(x_test, y_test)`.
9. Check inclusion with `calibrator.contains(x_test, z_test)`.
10. Report coverage, size or volume proxy, and runtime.

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
finite-sample quantile from `src/calibrators/quantile.py`.