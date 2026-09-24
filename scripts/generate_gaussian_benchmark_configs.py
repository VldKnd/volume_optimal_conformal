"""Generate PICNN Gaussian-to-Gaussian benchmark configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = REPOSITORY_ROOT / "benchmark/configurations/gaussian"

K_VALUES = (1, 2, 4, 8, 16, 32)
Y_DIMS = (2, 4, 8, 16, 32)
SEEDS = range(5)
FAMILY = "transport_neural_ot_l2"

DEVICE = "cuda"
DTYPE = "float32"
OT_WARMUP_ITERATIONS = 10

YAML_BLOCKS = (
    ("name", "seed", "save_directory"),
    ("dataset_config", ),
    ("predictor_config", ),
    ("trainer_config", ),
    ("conformal_config", ),
    (
        "train_batch_size",
        "calibration_batch_size",
        "test_batch_size",
        "compute_volume",
        "metrics_verbose",
    ),
    ("wandb", ),
)


class IndentedSafeDumper(yaml.SafeDumper):

    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


def dump_config(config: dict[str, Any]) -> str:
    blocks = []
    written_keys: set[str] = set()
    for keys in YAML_BLOCKS:
        block = {key: config[key] for key in keys if key in config}
        if block:
            blocks.append(
                yaml.dump(
                    block,
                    Dumper=IndentedSafeDumper,
                    sort_keys=False,
                    default_flow_style=False,
                ).rstrip()
            )
            written_keys.update(block)

    unexpected_keys = set(config) - written_keys
    if unexpected_keys:
        raise ValueError(
            "Configuration keys are missing from YAML_BLOCKS: "
            f"{sorted(unexpected_keys)}"
        )
    return "\n\n".join(blocks) + "\n"


def parameter_slug(*, k: int, y_dim: int) -> str:
    if k not in K_VALUES or y_dim not in Y_DIMS:
        raise ValueError(f"Unsupported Gaussian setting: k={k}, y_dim={y_dim}.")
    return f"dim_{y_dim}/k_{k}"


def make_config(*, k: int, y_dim: int, seed: int) -> dict[str, Any]:
    """Build one Gaussian target experiment without rearrangement training."""
    setting = parameter_slug(k=k, y_dim=y_dim)
    if seed not in SEEDS:
        raise ValueError(f"Gaussian benchmark seed must be in 0..4, got {seed}.")

    result_root = f"benchmark/results/gaussian/{setting}"
    return {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": f"{result_root}/{FAMILY}",
        "dataset_config": {
            "type": "elliptical_gaussian_dataset",
            "n_train": 20_000,
            "n_calibration": 1_000,
            "n_test": 1_000,
            "x_dim": 1,
            "y_dim": y_dim,
            "k": float(k),
            "seed": seed,
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "predictor_config": {
            "type": "neural_optimal_transport",
            "x_dim": 1,
            "y_dim": y_dim,
            "hidden_dim": 32,
            "num_hidden_layers": 2,
            "potential_type": "u",
            "standardize_y": False,
            "c_transform_lr": 1.0,
            "c_transform_max_iter": 1_000,
            "seed": seed,
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "trainer_config": {
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "warmup_iterations": OT_WARMUP_ITERATIONS,
            "grad_clip_norm": 1.0,
            "use_cosine_scheduler": True,
            "verbose": True,
        },
        "conformal_config": {
            "type": "transport_based",
            "coverage_mass": 0.9,
            "calibrator": {
                "type": "norm",
                "p": 2.0,
            },
            "volume_mc_samples": 1_000,
            "volume_batch_size": 1_024,
            "volume_seed": seed,
        },
        "train_batch_size": 1_024,
        "calibration_batch_size": 512,
        "test_batch_size": 512,
        "compute_volume": True,
        "metrics_verbose": True,
        "wandb": {
            "mode":
            "online",
            "project":
            "minimal-volume-conformal-prediction",
            "group":
            f"gaussian/{setting}/{FAMILY}",
            "name":
            f"{setting.replace('/', '_')}_{FAMILY}_seed_{seed:02d}",
            "tags": [
                "table_1",
                "gaussian",
                "synthetic",
                "neural-ot",
                "picnn",
                "transport",
                "l2",
                setting,
            ],
            "job_type":
            FAMILY,
            "log_every_n_steps":
            20,
            "log_solver_diagnostics":
            False,
        },
    }


def main() -> None:
    expected_paths: set[Path] = set()
    for y_dim in Y_DIMS:
        for k in K_VALUES:
            family_directory = (
                CONFIGURATION_ROOT / parameter_slug(k=k, y_dim=y_dim) / FAMILY
            )
            family_directory.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                path = family_directory / f"seed_{seed:02d}.yaml"
                path.write_text(
                    dump_config(make_config(k=k, y_dim=y_dim, seed=seed)),
                    encoding="utf-8",
                )
                expected_paths.add(path)

    stale_paths = set(CONFIGURATION_ROOT.rglob("*.yaml")) - expected_paths
    for path in stale_paths:
        path.unlink()
    for directory in sorted(
        (path for path in CONFIGURATION_ROOT.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    print(
        f"Generated {len(expected_paths)} Gaussian benchmark configurations "
        f"and removed {len(stale_paths)} stale configurations."
    )


if __name__ == "__main__":
    main()
