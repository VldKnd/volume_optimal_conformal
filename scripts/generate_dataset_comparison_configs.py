"""Generate the Bio, Blog, and SGEMM benchmark comparison suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = REPOSITORY_ROOT / "benchmark/configurations"

DATASETS: dict[str, dict[str, Any]] = {
    "bio": {
        "file_path": "data/bio/CASP.csv",
        "x_dim": 8,
        "y_dim": 2,
        "normalizing_flow_hidden_dim": 16,
        "normalizing_flow_hidden_layers": 4,
        "normalizing_flow_layers": 4,
        "neural_ot_hidden_dim": 14,
        "neural_ot_hidden_layers": 5,
        "rearrangement_hidden_dimension": 32,
        "rearrangement_hidden_layers": 5,
    },
    "blog": {
        "file_path": "data/blog/blogData_train.csv",
        "x_dim": 279,
        "y_dim": 2,
        "normalizing_flow_hidden_dim": 8,
        "normalizing_flow_hidden_layers": 4,
        "normalizing_flow_layers": 2,
        "neural_ot_hidden_dim": 7,
        "neural_ot_hidden_layers": 4,
        "rearrangement_hidden_dimension": 16,
        "rearrangement_hidden_layers": 4,
    },
    "sgemm": {
        "file_path": "data/sgemm/sgemm_product.csv",
        "x_dim": 14,
        "y_dim": 4,
        "normalizing_flow_hidden_dim": 32,
        "normalizing_flow_hidden_layers": 6,
        "normalizing_flow_layers": 4,
        "neural_ot_hidden_dim": 32,
        "neural_ot_hidden_layers": 6,
        "rearrangement_hidden_dimension": 64,
        "rearrangement_hidden_layers": 8,
    },
}

FAMILIES = (
    "residual_rf_elliptic",
    "residual_rf_global_otcp",
    "residual_rf_local_otcp",
    "transport_realnvp_l2",
    "transport_neural_ot_l2",
    "transport_realnvp_rearranged_l2",
    "transport_neural_ot_rearranged_l2",
)

YAML_BLOCKS = (
    ("name", "seed", "save_directory"),
    ("dataset_config",),
    ("predictor_config", "predictor_checkpoint"),
    ("trainer_config",),
    ("rearrangement_config", "rearrangement_checkpoint"),
    ("rearrangement_trainer_config",),
    ("supervised_rearrangement",),
    ("conformal_config", "conformal_checkpoint"),
    (
        "train_batch_size",
        "rearrangement_train_batch_size",
        "calibration_batch_size",
        "test_batch_size",
        "compute_volume",
        "metrics_verbose",
    ),
    ("wandb",),
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


def dataset_config(dataset: str, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": dataset,
        "file_path": spec["file_path"],
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "train_fraction": 0.6,
        "calibration_fraction": 0.2,
        "test_fraction": 0.2,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def random_forest_config(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "random_forest",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "n_estimators": 100,
        "max_depth": None,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def normalizing_flow_config(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "normalizing_flow",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "hidden_dim": spec["normalizing_flow_hidden_dim"],
        "num_hidden_layers": spec["normalizing_flow_hidden_layers"],
        "num_flow_layers": spec["normalizing_flow_layers"],
        "log_scale_bound": 3.0,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def neural_ot_config(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "neural_optimal_transport",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "hidden_dim": spec["neural_ot_hidden_dim"],
        "num_hidden_layers": spec["neural_ot_hidden_layers"],
        "potential_type": "u",
        "c_transform_lr": 1.0,
        "c_transform_max_iter": 1_000,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def base_trainer_config(predictor_type: str) -> dict[str, Any]:
    if predictor_type == "random_forest":
        return {"epochs": 1}

    config = {
        "epochs": 250,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip_norm": 1.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }
    if predictor_type == "neural_optimal_transport":
        config = {
            "epochs": 250,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "warmup_iterations": 1,
            "grad_clip_norm": 1.0,
            "use_cosine_scheduler": True,
            "verbose": True,
        }
    return config


def rearrangement_config(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "amortized_rearranged_transport",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "hidden_dimension": spec["rearrangement_hidden_dimension"],
        "number_of_hidden_layers": spec["rearrangement_hidden_layers"],
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
        "device": "cuda",
        "dtype": "float32",
    }


def rearrangement_trainer_config() -> dict[str, Any]:
    return {
        "epochs": 100,
        "mc_samples_per_x": 32,
        "train_transport_map": False,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip_norm": 1.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }


def residual_conformal_config(family: str, seed: int) -> dict[str, Any]:
    if family == "residual_rf_elliptic":
        calibrator = {
            "type": "elliptic",
            "n_neighbors": 100,
            "regularization": 0.0001,
            "local_weight": 0.8,
        }
    elif family == "residual_rf_global_otcp":
        calibrator = {"type": "global_otcp", "seed": seed}
    else:
        calibrator = {
            "type": "local_otcp",
            "n_neighbors": 100,
            "seed": seed,
        }
    return {
        "type": "residual",
        "coverage_mass": 0.9,
        "calibrator": calibrator,
        "volume_mc_samples": 1_000,
        "volume_n_neighbors": 100,
        "volume_seed": seed,
    }


def transport_conformal_config(seed: int) -> dict[str, Any]:
    return {
        "type": "transport_based",
        "coverage_mass": 0.9,
        "calibrator": {"type": "norm", "p": 2.0},
        "volume_mc_samples": 1_000,
        "volume_batch_size": 1_024,
        "volume_seed": seed,
    }


def wandb_config(dataset: str, family: str, seed: int) -> dict[str, Any]:
    tags = [dataset, "benchmark-comparison"]
    if family.startswith("residual_rf"):
        tags.extend(["random-forest", "residual"])
        tags.append(family.removeprefix("residual_rf_").replace("_", "-"))
    elif "realnvp" in family:
        tags.extend(["normalizing-flow", "realnvp", "transport", "l2"])
    else:
        tags.extend(["neural-ot", "transport", "l2"])
    if "rearranged" in family:
        tags.extend(["rearranged", "amortized", "sparse", "dopri5"])

    return {
        "mode": "online",
        "project": "minimal-volume-conformal-prediction",
        "entity": None,
        "group": f"{dataset}/{family}",
        "name": f"{family}_seed_{seed:02d}",
        "tags": tags,
        "job_type": family,
        "log_every_n_steps": 20,
        "log_solver_diagnostics": "rearranged" in family,
    }


def make_config(dataset: str, family: str, seed: int) -> dict[str, Any]:
    spec = DATASETS[dataset]
    is_residual = family.startswith("residual_rf")
    is_neural_ot = "neural_ot" in family
    is_rearranged = "rearranged" in family

    if is_residual:
        predictor = random_forest_config(spec, seed)
    elif is_neural_ot:
        predictor = neural_ot_config(spec, seed)
    else:
        predictor = normalizing_flow_config(spec, seed)

    config: dict[str, Any] = {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": f"benchmark/results/{dataset}/base_run/{family}",
        "dataset_config": dataset_config(dataset, spec, seed),
        "predictor_config": predictor,
        "trainer_config": base_trainer_config(predictor["type"]),
    }

    if is_rearranged:
        base_family = family.replace("_rearranged", "")
        config["predictor_checkpoint"] = (
            f"benchmark/results/{dataset}/base_run/{base_family}/seed_{seed:02d}"
            "/base/predictor.pt"
        )
        config["rearrangement_config"] = rearrangement_config(spec, seed)
        config["rearrangement_trainer_config"] = rearrangement_trainer_config()
        config["supervised_rearrangement"] = False

    config["conformal_config"] = (
        residual_conformal_config(family, seed)
        if is_residual
        else transport_conformal_config(seed)
    )
    config.update(
        {
            "train_batch_size": 4_096,
            **(
                {"rearrangement_train_batch_size": 512}
                if is_rearranged
                else {}
            ),
            "calibration_batch_size": 512,
            "test_batch_size": 512,
            "compute_volume": True,
            "metrics_verbose": True,
            "wandb": wandb_config(dataset, family, seed),
        }
    )
    return config


def main() -> None:
    for dataset in DATASETS:
        suite_directory = (
            CONFIGURATION_ROOT / dataset / "base_run"
        )
        for family in FAMILIES:
            family_directory = suite_directory / family
            family_directory.mkdir(parents=True, exist_ok=True)
            for seed in range(5):
                path = family_directory / f"seed_{seed:02d}.yaml"
                path.write_text(
                    dump_config(make_config(dataset, family, seed)),
                    encoding="utf-8",
                )

    count = len(DATASETS) * len(FAMILIES) * 5
    print(f"Generated {count} benchmark configurations.")


if __name__ == "__main__":
    main()
