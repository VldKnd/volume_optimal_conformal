"""Generate Neural OT architecture-sweep configurations for real datasets."""

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
        "grad_clip_norm": 20.0,
    },
    "bio": {
        "file_path": "data/bio/CASP.csv",
        "x_dim": 8,
        "y_dim": 2,
        "train_batch_size": 4_096,
        "grad_clip_norm": 10.0,
    },
    "blog": {
        "file_path": "data/blog/blogData_train.csv",
        "x_dim": 279,
        "y_dim": 2,
        "train_batch_size": 4_096,
        "grad_clip_norm": 10.0,
    },
    "sgemm": {
        "file_path": "data/sgemm/sgemm_product.csv",
        "x_dim": 14,
        "y_dim": 4,
        "train_batch_size": 4_096,
        "grad_clip_norm": 10.0,
    },
    "qm9": {
        "file_path": "data/qm9/qm9_mbtr_100.npz",
        "raw_root": "data/qm9/pyg",
        "x_dim": 100,
        "y_dim": 4,
        "train_batch_size": 4_096,
        "grad_clip_norm": 10.0,
    },
}

MODEL_SIZES: dict[str, dict[str, int]] = {
    "small": {
        "hidden_dim": 8,
        "num_hidden_layers": 4
    },
    "medium": {
        "hidden_dim": 16,
        "num_hidden_layers": 4
    },
    "big": {
        "hidden_dim": 32,
        "num_hidden_layers": 8
    },
    "huge": {
        "hidden_dim": 64,
        "num_hidden_layers": 8
    },
}

SEEDS = range(2)
DEVICE = "cuda"
DTYPE = "float32"
FAMILY = "transport_neural_ot_l2"
SUITE = "model_sweep"

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


def make_config(*, dataset: str, model_size: str, seed: int) -> dict[str, Any]:
    """Build one Neural OT model-sweep experiment mapping."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unsupported model size: {model_size!r}.")
    if seed not in SEEDS:
        raise ValueError(f"Model-sweep seed must be 0 or 1, got {seed}.")

    spec = DATASETS[dataset]
    architecture = MODEL_SIZES[model_size]
    variant = f"{FAMILY}_{model_size}"

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

    return {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": f"benchmark/results/{dataset}/{SUITE}/{variant}",
        "dataset_config": dataset_config,
        "predictor_config": {
            "type": "neural_optimal_transport",
            "x_dim": spec["x_dim"],
            "y_dim": spec["y_dim"],
            **architecture,
            "potential_type": "u",
            "c_transform_lr": 1.0,
            "c_transform_max_iter": 1_000,
            "seed": seed,
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "trainer_config": {
            "epochs": 250,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "warmup_iterations": 10,
            "grad_clip_norm": spec["grad_clip_norm"],
            "use_cosine_scheduler": True,
            "verbose": True,
        },
        "conformal_config": {
            "type": "transport_based",
            "coverage_mass": 0.9,
            "calibrator": {
                "type": "norm",
                "p": 2.0
            },
            "volume_mc_samples": 1_000,
            "volume_batch_size": 1_024,
            "volume_seed": seed,
        },
        "train_batch_size": spec["train_batch_size"],
        "calibration_batch_size": 512,
        "test_batch_size": 512,
        "compute_volume": True,
        "metrics_verbose": True,
        "wandb": {
            "mode": "online",
            "project": "minimal-volume-conformal-prediction",
            "entity": None,
            "group": f"{dataset}/{SUITE}/{variant}",
            "name": f"{variant}_seed_{seed:02d}",
            "tags": [
                dataset,
                "model_sweep",
                model_size,
                "neural-ot",
                "transport",
                "l2",
            ],
            "job_type": variant,
            "log_every_n_steps": 20,
            "log_solver_diagnostics": False,
        },
    }


def main() -> None:
    count = 0
    for dataset in DATASETS:
        for model_size in MODEL_SIZES:
            variant = f"{FAMILY}_{model_size}"
            directory = CONFIGURATION_ROOT / dataset / SUITE / variant
            directory.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                path = directory / f"seed_{seed:02d}.yaml"
                path.write_text(
                    dump_config(
                        make_config(
                            dataset=dataset,
                            model_size=model_size,
                            seed=seed,
                        )
                    ),
                    encoding="utf-8",
                )
                count += 1

    print(f"Generated {count} Neural OT model-sweep configurations.")


if __name__ == "__main__":
    main()
