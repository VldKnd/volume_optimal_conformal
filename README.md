# Minimal-Volume Conformal Prediction

Research code for multivariate conformal prediction with learned transport
maps. The project trains conditional transports, conformalizes their latent
regions, and compares coverage and log-volume on real and synthetic datasets.
It also studies Gaussian-preserving rearrangements intended to reduce the
volume of transported prediction regions.

The main workflow is:

```text
dataset -> transport or regression predictor -> conformal calibration -> evaluation
```

## Setup and execution

The project targets Python 3.11 and uses `uv`:

```bash
uv sync
export PYTHONPATH=src
uv run python scripts/run_benchmark.py <configuration-or-folder>
```

Specialized Student-t and synthetic entry points, shell wrappers, and Slurm
submission helpers are also available in `scripts/`. Run the test suite with:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

## Repository layout

- `src/` — reusable implementation.
  - `configs/` and `data/` define validated configurations and dataset loaders.
  - `predictors/`, `networks/`, and `trainers/` contain regression, transport,
    Neural OT, flow, and rearrangement models and their optimization code.
  - `conformal/` calibrates prediction regions; `evaluation/` provides coverage
    and log-volume metrics; `experimentation/` runs complete experiments.
- `benchmark/configurations/` — YAML experiment grids grouped by dataset and
  method. `benchmark/results/` and `benchmark/downloads/` hold run artifacts and
  downloaded results.
- `scripts/` — configuration generators and local, shell, or Slurm benchmark
  entry points.
- `notebooks/` — dataset exploration, theorem illustrations, result analysis,
  plotting, and small sandbox experiments.
- `profiling/` — targeted numerical, runtime, and model-diagnostic studies.
- `tests/` — unit and integration tests for datasets, transports, conformal
  predictors, metrics, configurations, and runners.
- `data/` — local raw datasets; `weights/` — standalone pretrained checkpoints.
- `figures/`, `plots/`, and `logs/` — generated visual and execution artifacts.

Weights & Biases tracking is optional and can be installed with
`uv sync --extra tracking`; individual YAML configurations control whether it
is disabled, offline, or online.
