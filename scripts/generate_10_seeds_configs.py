"""Generate the final 10-seed benchmark suite for selected real datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = REPOSITORY_ROOT / "benchmark/configurations"

SUITE = "10_seeds"
SEEDS = range(10)

RF_ELLIPTIC = "rf_elliptic"
GLOBAL_OTCP = "global_otcp"
LOCAL_OTCP = "local_otcp"
NEURAL_OT_L2 = "neural_ot_l2"
NEURAL_OT_REARRANGED_L2 = "neural_ot_rearranged_l2"
REALNVP_L2 = "realnvp_l2"
REALNVP_CDF = "realnvp_cdf"
REALNVP_LOG_PROB = "realnvp_log_prob"
REALNVP_REARRANGED = "realnvp_rearranged"

METHODS = (
    NEURAL_OT_L2,
    REALNVP_L2,
    RF_ELLIPTIC,
    GLOBAL_OTCP,
    LOCAL_OTCP,
    NEURAL_OT_REARRANGED_L2,
    REALNVP_CDF,
    REALNVP_LOG_PROB,
    REALNVP_REARRANGED,
)

RF_METHODS = frozenset({RF_ELLIPTIC, GLOBAL_OTCP, LOCAL_OTCP})
NEURAL_OT_METHODS = frozenset({NEURAL_OT_L2, NEURAL_OT_REARRANGED_L2})
REARRANGED_METHODS = frozenset({NEURAL_OT_REARRANGED_L2, REALNVP_REARRANGED})
CHECKPOINT_METHODS = frozenset(
    {
        NEURAL_OT_REARRANGED_L2,
        REALNVP_CDF,
        REALNVP_LOG_PROB,
        REALNVP_REARRANGED,
    }
)

CDF_SAMPLES_BY_Y_DIM = {2: 32, 4: 64, 16: 512}

DATASETS: dict[str, dict[str, Any]] = {
    "bio": {
        "file_path": "data/bio/CASP.csv",
        "x_dim": 8,
        "y_dim": 2,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
    },
    "blog": {
        "file_path": "data/blog/blogData_train.csv",
        "x_dim": 279,
        "y_dim": 2,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
    },
    "qm9": {
        "file_path": "data/qm9/qm9_mbtr_100.npz",
        "raw_root": "data/qm9/pyg",
        "x_dim": 100,
        "y_dim": 4,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
    },
    "scm20d": {
        "file_path": "data/scm20d/file1730492b4408.arff",
        "x_dim": 61,
        "y_dim": 16,
        "train_batch_size": 256,
        "rearrangement_train_batch_size": 256,
    },
    "sgemm": {
        "file_path": "data/sgemm/sgemm_product.csv",
        "x_dim": 14,
        "y_dim": 4,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
    },
}

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


def _dataset_config(
    dataset: str,
    spec: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, Any]:
    return {
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
        device,
        "dtype":
        "float32",
    }


def _random_forest_predictor(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "random_forest",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "n_estimators": 100,
        "max_depth": None,
        "seed": seed,
        "device": "cpu",
        "dtype": "float32",
    }


def _neural_ot_predictor(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "neural_optimal_transport",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "hidden_dim": 64,
        "num_hidden_layers": 8,
        "potential_type": "u",
        "c_transform_lr": 1.0,
        "c_transform_max_iter": 1_000,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def _realnvp_predictor(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "normalizing_flow",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "hidden_dim": 64,
        "num_hidden_layers": 4,
        "num_flow_layers": 4,
        "log_scale_bound": 3.0,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def _trainer_config(method: str, spec: dict[str, Any]) -> dict[str, Any]:
    if method in RF_METHODS:
        return {"epochs": 1}

    trainer = {
        "epochs": 250,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip_norm": 10.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }
    if method in NEURAL_OT_METHODS:
        trainer["warmup_iterations"] = 50
        trainer["grad_clip_norm"] = 20.0 if spec["y_dim"] == 16 else 10.0
    return trainer


def _rearrangement_config(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "rearranged_transport",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
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
        "grad_clip_norm": 1.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }


def _residual_conformal_config(method: str, spec: dict[str, Any],
                               seed: int) -> dict[str, Any]:
    calibrator_n_neighbors = spec.get("calibrator_n_neighbors", 100)
    if method == RF_ELLIPTIC:
        calibrator = {
            "type": "elliptic",
            "n_neighbors": calibrator_n_neighbors,
            "regularization": 0.0001,
            "local_weight": 0.8,
        }
    elif method == GLOBAL_OTCP:
        calibrator = {"type": "global_otcp", "seed": seed}
    else:
        calibrator = {
            "type": "local_otcp",
            "n_neighbors": calibrator_n_neighbors,
            "seed": seed,
        }
    return {
        "type": "residual",
        "coverage_mass": 0.9,
        "calibrator": calibrator,
        "volume_mc_samples": 1_000,
        "volume_n_neighbors": spec.get("volume_n_neighbors", 100),
        "volume_seed": seed,
    }


def _transport_conformal_config(method: str, spec: dict[str, Any],
                                seed: int) -> dict[str, Any]:
    if method == REALNVP_CDF:
        calibrator = {
            "type": "cdf_calibrator",
            "n_cdf_samples": CDF_SAMPLES_BY_Y_DIM[spec["y_dim"]],
            "cdf_batch_size": 65_536,
        }
    elif method == REALNVP_LOG_PROB:
        calibrator = {"type": "log_probability"}
    else:
        calibrator = {"type": "norm", "p": 2.0}

    return {
        "type": "transport_based",
        "coverage_mass": 0.9,
        "calibrator": calibrator,
        "volume_mc_samples": 1_000,
        "volume_batch_size": 1_024,
        "volume_n_neighbors": spec.get("volume_n_neighbors", 100),
        "volume_seed": seed,
    }


def _wandb_config(dataset: str, method: str, seed: int) -> dict[str, Any]:
    tags = [dataset, SUITE, "final-benchmark"]
    if method in RF_METHODS:
        tags.extend(["random-forest", "residual", method.replace("_", "-")])
    elif method in NEURAL_OT_METHODS:
        tags.extend(["transport", "neural-ot", "l2"])
    else:
        tags.extend(["transport", "normalizing-flow", "realnvp"])
        if method in {REALNVP_L2, REALNVP_REARRANGED}:
            tags.append("l2")
        elif method == REALNVP_CDF:
            tags.append("cdf")
        else:
            tags.append("log-probability")

    if method in REARRANGED_METHODS:
        tags.extend(["rearranged", "non-amortized", "sparse", "dopri5"])

    return {
        "mode": "online",
        "project": "minimal-volume-conformal-prediction",
        "entity": None,
        "group": f"{SUITE}/{dataset}/{method}",
        "name": f"{method}_seed_{seed:02d}",
        "tags": tags,
        "job_type": method,
        "log_every_n_steps": 20,
        "log_solver_diagnostics": method in REARRANGED_METHODS,
    }


def make_config(*, dataset: str, method: str, seed: int) -> dict[str, Any]:
    """Build one final-benchmark experiment configuration."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method!r}.")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be between 0 and 9, got {seed}.")

    spec = DATASETS[dataset]
    device = "cpu" if method in RF_METHODS else "cuda"
    if method in RF_METHODS:
        predictor = _random_forest_predictor(spec, seed)
    elif method in NEURAL_OT_METHODS:
        predictor = _neural_ot_predictor(spec, seed)
    else:
        predictor = _realnvp_predictor(spec, seed)

    config: dict[str, Any] = {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": f"benchmark/results/{SUITE}/{dataset}/{method}",
        "dataset_config": _dataset_config(dataset, spec, seed, device),
        "predictor_config": predictor,
        "trainer_config": _trainer_config(method, spec),
    }

    if method in CHECKPOINT_METHODS:
        base_method = (
            NEURAL_OT_L2 if method == NEURAL_OT_REARRANGED_L2 else REALNVP_L2
        )
        config["predictor_checkpoint"] = (
            f"benchmark/results/{SUITE}/{dataset}/{base_method}/"
            f"seed_{seed:02d}/base/predictor.pt"
        )

    if method in REARRANGED_METHODS:
        config.update(
            rearrangement_config=_rearrangement_config(spec, seed),
            rearrangement_trainer_config=_rearrangement_trainer_config(),
            supervised_rearrangement=False,
        )

    config.update(
        conformal_config=(
            _residual_conformal_config(method, spec, seed) if method in RF_METHODS else
            _transport_conformal_config(method, spec, seed)
        ),
        train_batch_size=spec["train_batch_size"],
        **(
            {
                "rearrangement_train_batch_size": spec["rearrangement_train_batch_size"]
            } if method in REARRANGED_METHODS else {}
        ),
        calibration_batch_size=512,
        test_batch_size=512,
        compute_volume=True,
        metrics_verbose=True,
        wandb=_wandb_config(dataset, method, seed),
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    arguments = parser.parse_args()
    datasets = (
        (arguments.dataset, ) if arguments.dataset is not None else tuple(DATASETS)
    )

    count = 0
    for dataset in datasets:
        for method in METHODS:
            directory = CONFIGURATION_ROOT / SUITE / dataset / method
            directory.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                path = directory / f"seed_{seed:02d}.yaml"
                path.write_text(
                    dump_config(make_config(dataset=dataset, method=method, seed=seed)),
                    encoding="utf-8",
                )
                count += 1

    print(f"Generated {count} {SUITE} configurations.")


if __name__ == "__main__":
    main()
