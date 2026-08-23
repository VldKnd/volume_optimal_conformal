"""Generate pairwise synthetic dimension-scaling benchmark configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = REPOSITORY_ROOT / "benchmark/configurations"

SUITE = "synthetic_dimension_scaling"
DIMENSIONS = (2, 4, 8, 16, 32, 64, 128)
SEEDS = range(10)

NEURAL_OT_L2 = "neural_ot_l2"
NEURAL_OT_REARRANGED_L2 = "neural_ot_rearranged_l2"
METHODS = (NEURAL_OT_L2, NEURAL_OT_REARRANGED_L2)
DATASETS = (
    "banana",
    "sinusoidal_transport",
    "star_shaped_gaussian",
)

YAML_BLOCKS = (
    ("name", "seed", "save_directory"),
    ("dataset_config", ),
    ("predictor_config", "predictor_checkpoint"),
    ("trainer_config", ),
    ("rearrangement_config", ),
    ("rearrangement_trainer_config", ),
    ("supervised_rearrangement", ),
    ("conformal_config", ),
    (
        "train_batch_size",
        "rearrangement_train_batch_size",
        "calibration_batch_size",
        "test_batch_size",
        "compute_volume",
        "metrics_verbose",
    ),
    ("wandb", ),
)


class IndentedSafeDumper(yaml.SafeDumper):
    """Indent sequence items beneath their mapping key."""

    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


def dump_config(config: dict[str, Any]) -> str:
    """Render top-level configuration sections as separated YAML blocks."""
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


def dimension_directory(dimension: int) -> str:
    """Return a lexically sortable directory name for one target dimension."""
    return f"dim_{dimension:03d}"


def _dataset_config(dataset: str, dimension: int, seed: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "type": dataset,
        "n_train": 100_000,
        "n_calibration": 1_000,
        "n_test": 1_000,
        "x_dim": 1,
        "y_dim": dimension,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }
    if dataset == "sinusoidal_transport":
        config.update(
            x_low=0.0,
            x_high=0.0,
            amplitude=1.0,
            amplitude_x_scale=0.0,
            frequency=2.0,
            phase=0.0,
            vertical_scale=1.0,
            vertical_scale_x_scale=0.0,
        )
    elif dataset == "star_shaped_gaussian":
        config.update(
            x_low=0.0,
            x_high=0.0,
            petal_amplitude=0.45,
            rotation_fraction=0.0,
        )
    return config


def _predictor_config(dimension: int, seed: int) -> dict[str, Any]:
    return {
        "type": "neural_optimal_transport",
        "x_dim": 1,
        "y_dim": dimension,
        "hidden_dim": 64,
        "num_hidden_layers": 8,
        "potential_type": "u",
        "c_transform_lr": 1.0,
        "c_transform_max_iter": 1_000,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def _trainer_config() -> dict[str, Any]:
    return {
        "epochs": 250,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "warmup_iterations": 50,
        "grad_clip_norm": 10.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }


def _rearrangement_config(dimension: int, seed: int) -> dict[str, Any]:
    return {
        "type": "rearranged_transport",
        "x_dim": 1,
        "y_dim": dimension,
        "hidden_dimension": 256,
        "number_of_hidden_layers": 1,
        "time_dependent": True,
        "vector_field_implementation": "sparse",
        "activation": "silu",
        "use_adjoint": False,
        "method": "dopri5",
        "rtol": 1.0e-5,
        "atol": 1.0e-6,
        "number_of_steps": None,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def _rearrangement_trainer_config() -> dict[str, Any]:
    return {
        "epochs": 100,
        "mc_samples_per_x": 32,
        "train_transport_map": False,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip_norm": 10.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }


def _conformal_config(seed: int) -> dict[str, Any]:
    return {
        "type": "transport_based",
        "coverage_mass": 0.9,
        "calibrator": {
            "type": "norm",
            "p": 2.0,
        },
        "volume_mc_samples": 1_000,
        "volume_batch_size": 1_024,
        "volume_seed": seed,
    }


def _wandb_config(
    dataset: str,
    dimension: int,
    method: str,
    seed: int,
) -> dict[str, Any]:
    is_rearranged = method == NEURAL_OT_REARRANGED_L2
    tags = [
        "synthetic",
        SUITE,
        dataset,
        f"dimension-{dimension}",
        "transport",
        "neural-ot",
        "l2",
    ]
    if is_rearranged:
        tags.extend(["rearranged", "non-amortized", "sparse", "dopri5"])

    dimension_name = dimension_directory(dimension)
    return {
        "mode": "online",
        "project": "minimal-volume-conformal-prediction",
        "entity": None,
        "group": f"{SUITE}/{dataset}/{dimension_name}/{method}",
        "name": f"{method}_{dimension_name}_seed_{seed:02d}",
        "tags": tags,
        "job_type": method,
        "log_every_n_steps": 20,
        "log_solver_diagnostics": is_rearranged,
    }


def make_config(
    *,
    dataset: str,
    dimension: int,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Build one synthetic dimension-scaling configuration."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unsupported target dimension: {dimension}.")
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method!r}.")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be between 0 and 9, got {seed}.")

    dimension_name = dimension_directory(dimension)
    result_directory = (
        f"benchmark/results/{SUITE}/{dataset}/{dimension_name}/{method}"
    )
    config: dict[str, Any] = {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": result_directory,
        "dataset_config": _dataset_config(dataset, dimension, seed),
        "predictor_config": _predictor_config(dimension, seed),
        "trainer_config": _trainer_config(),
    }

    if method == NEURAL_OT_REARRANGED_L2:
        config.update(
            predictor_checkpoint=(
                f"benchmark/results/{SUITE}/{dataset}/{dimension_name}/"
                f"{NEURAL_OT_L2}/seed_{seed:02d}/base/predictor.pt"
            ),
            rearrangement_config=_rearrangement_config(dimension, seed),
            rearrangement_trainer_config=_rearrangement_trainer_config(),
            supervised_rearrangement=False,
        )

    config.update(
        conformal_config=_conformal_config(seed),
        train_batch_size=1_024,
        **(
            {
                "rearrangement_train_batch_size": 1_024
            } if method == NEURAL_OT_REARRANGED_L2 else {}
        ),
        calibration_batch_size=512,
        test_batch_size=512,
        compute_volume=True,
        metrics_verbose=True,
        wandb=_wandb_config(dataset, dimension, method, seed),
    )
    return config


def main() -> None:
    count = 0
    for dataset in DATASETS:
        for dimension in DIMENSIONS:
            dimension_name = dimension_directory(dimension)
            for method in METHODS:
                directory = (
                    CONFIGURATION_ROOT / SUITE / dataset / dimension_name / method
                )
                directory.mkdir(parents=True, exist_ok=True)
                for seed in SEEDS:
                    path = directory / f"seed_{seed:02d}.yaml"
                    path.write_text(
                        dump_config(
                            make_config(
                                dataset=dataset,
                                dimension=dimension,
                                method=method,
                                seed=seed,
                            )
                        ),
                        encoding="utf-8",
                    )
                    count += 1

    print(f"Generated {count} {SUITE} configurations.")


if __name__ == "__main__":
    main()
