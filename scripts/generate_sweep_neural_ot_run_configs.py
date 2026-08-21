"""Generate the huge NeuralOT and rearranged NeuralOT real-data sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = REPOSITORY_ROOT / "benchmark/configurations"

DATASETS: dict[str, dict[str, Any]] = {
    "scm20d": {
        "file_path": "data/scm20d/file1730492b4408.arff",
        "x_dim": 61,
        "y_dim": 16,
        "train_batch_size": 256,
        "rearrangement_train_batch_size": 256,
        "grad_clip_norm": 20.0,
    },
    "bio": {
        "file_path": "data/bio/CASP.csv",
        "x_dim": 8,
        "y_dim": 2,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
        "grad_clip_norm": 10.0,
    },
    "blog": {
        "file_path": "data/blog/blogData_train.csv",
        "x_dim": 279,
        "y_dim": 2,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
        "grad_clip_norm": 10.0,
    },
    "sgemm": {
        "file_path": "data/sgemm/sgemm_product.csv",
        "x_dim": 14,
        "y_dim": 4,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
        "grad_clip_norm": 10.0,
    },
    "qm9": {
        "file_path": "data/qm9/qm9_mbtr_100.npz",
        "raw_root": "data/qm9/pyg",
        "x_dim": 100,
        "y_dim": 4,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
        "grad_clip_norm": 10.0,
    },
}

SEEDS = range(5)
SUITE = "sweep_neural_ot_run"
BASE_FAMILY = "transport_neural_ot_l2_huge"
REARRANGED_FAMILY = "transport_neural_ot_rearranged_l2_huge"
FAMILIES = (BASE_FAMILY, REARRANGED_FAMILY)
DEVICE = "cuda"
DTYPE = "float32"
HIDDEN_DIMENSION = 64
NUMBER_OF_HIDDEN_LAYERS = 8
BASE_EPOCHS = 250
BASE_WARMUP_ITERATIONS = 50
REARRANGEMENT_EPOCHS = 50

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


def make_config(*, dataset: str, family: str, seed: int) -> dict[str, Any]:
    """Build one huge NeuralOT sweep configuration."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")
    if family not in FAMILIES:
        raise ValueError(f"Unsupported family: {family!r}.")
    if seed not in SEEDS:
        raise ValueError(f"Sweep seed must be between 0 and 4, got {seed}.")

    spec = DATASETS[dataset]
    is_rearranged = family == REARRANGED_FAMILY
    dataset_config = {
        "type":
        dataset,
        "file_path":
        spec["file_path"],
        **({
            "raw_root": spec["raw_root"]
        } if "raw_root" in spec else {}),
        "x_dim":
        spec["x_dim"],
        "y_dim":
        spec["y_dim"],
        "train_fraction":
        0.6,
        "calibration_fraction":
        0.2,
        "test_fraction":
        0.2,
        "seed":
        seed,
        "device":
        DEVICE,
        "dtype":
        DTYPE,
    }
    predictor_config = {
        "type": "neural_optimal_transport",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        # This selects the repository's standard PISCNN architecture.
        "hidden_dim": HIDDEN_DIMENSION,
        "num_hidden_layers": NUMBER_OF_HIDDEN_LAYERS,
        "potential_type": "u",
        "c_transform_lr": 1.0,
        "c_transform_max_iter": 1_000,
        "seed": seed,
        "device": DEVICE,
        "dtype": DTYPE,
    }
    config: dict[str, Any] = {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": f"benchmark/results/{dataset}/{SUITE}/{family}",
        "dataset_config": dataset_config,
        "predictor_config": predictor_config,
        "trainer_config": {
            "epochs": BASE_EPOCHS,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "warmup_iterations": BASE_WARMUP_ITERATIONS,
            "grad_clip_norm": spec["grad_clip_norm"],
            "use_cosine_scheduler": True,
            "verbose": True,
        },
    }

    if is_rearranged:
        config.update(
            {
                "predictor_checkpoint": (
                    f"benchmark/results/{dataset}/{SUITE}/{BASE_FAMILY}/"
                    f"seed_{seed:02d}/base/predictor.pt"
                ),
                "rearrangement_config": {
                    "type": "amortized_rearranged_transport",
                    "x_dim": spec["x_dim"],
                    "y_dim": spec["y_dim"],
                    "hidden_dimension": HIDDEN_DIMENSION,
                    "number_of_hidden_layers": NUMBER_OF_HIDDEN_LAYERS,
                    "time_dependent": True,
                    "vector_field_implementation": "sparse",
                    "activation": "silu",
                    "activation_power": 2.0,
                    "use_adjoint": False,
                    "method": "dopri5",
                    "rtol": 0.0001,
                    "atol": 0.00001,
                    "number_of_steps": None,
                    "seed": seed,
                    "device": DEVICE,
                    "dtype": DTYPE,
                },
                "rearrangement_trainer_config": {
                    "epochs": REARRANGEMENT_EPOCHS,
                    "mc_samples_per_x": 32,
                    "train_transport_map": False,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "grad_clip_norm": spec["grad_clip_norm"],
                    "use_cosine_scheduler": True,
                    "verbose": True,
                },
                "supervised_rearrangement":
                False,
            }
        )

    config.update(
        {
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
            "train_batch_size":
            spec["train_batch_size"],
            **(
                {
                    "rearrangement_train_batch_size": spec["rearrangement_train_batch_size"]
                } if is_rearranged else {}
            ),
            "calibration_batch_size":
            512,
            "test_batch_size":
            512,
            "compute_volume":
            True,
            "metrics_verbose":
            True,
            "wandb": {
                "mode":
                "online",
                "project":
                "minimal-volume-conformal-prediction",
                "entity":
                None,
                "group":
                f"{dataset}/{SUITE}/{family}",
                "name":
                f"{family}_seed_{seed:02d}",
                "tags": [
                    dataset,
                    SUITE,
                    "model_sweep",
                    "huge",
                    "neural-ot",
                    "transport",
                    "l2",
                    *(
                        ["rearranged", "amortized", "sparse", "dopri5"]
                        if is_rearranged else []
                    ),
                ],
                "job_type":
                family,
                "log_every_n_steps":
                20,
                "log_solver_diagnostics":
                is_rearranged,
            },
        }
    )
    return config


def main() -> None:
    count = 0
    for dataset in DATASETS:
        for family in FAMILIES:
            directory = CONFIGURATION_ROOT / dataset / SUITE / family
            directory.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                path = directory / f"seed_{seed:02d}.yaml"
                path.write_text(
                    dump_config(
                        make_config(
                            dataset=dataset,
                            family=family,
                            seed=seed,
                        )
                    ),
                    encoding="utf-8",
                )
                count += 1

    print(f"Generated {count} {SUITE} configurations.")


if __name__ == "__main__":
    main()
