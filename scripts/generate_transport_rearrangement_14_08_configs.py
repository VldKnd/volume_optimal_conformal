"""Generate the 14 August transport/rearrangement real-data suite."""

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
        "neural_ot_grad_clip_norm": 20.0,
    },
    "bio": {
        "file_path": "data/bio/CASP.csv",
        "x_dim": 8,
        "y_dim": 2,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
        "neural_ot_grad_clip_norm": 10.0,
    },
    "blog": {
        "file_path": "data/blog/blogData_train.csv",
        "x_dim": 279,
        "y_dim": 2,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
        "neural_ot_grad_clip_norm": 10.0,
    },
    "sgemm": {
        "file_path": "data/sgemm/sgemm_product.csv",
        "x_dim": 14,
        "y_dim": 4,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
        "neural_ot_grad_clip_norm": 10.0,
    },
    "qm9": {
        "file_path": "data/qm9/qm9_mbtr_100.npz",
        "raw_root": "data/qm9/pyg",
        "x_dim": 100,
        "y_dim": 4,
        "train_batch_size": 4_096,
        "rearrangement_train_batch_size": 512,
        "neural_ot_grad_clip_norm": 10.0,
    },
}

SEEDS = range(5)
SUITE = "transport_rearrangement_14_08"
NEURAL_OT_REARRANGED_FAMILY = "transport_neural_ot_rearranged_l2"
REALNVP_BASE_FAMILY = "transport_realnvp_l2"
REALNVP_REARRANGED_FAMILY = "transport_realnvp_rearranged_l2"
REALNVP_CDF_FAMILY = "transport_realnvp_cdf"
REALNVP_LOG_PROBABILITY_FAMILY = "transport_realnvp_log_probability"
FAMILIES = (
    NEURAL_OT_REARRANGED_FAMILY,
    REALNVP_BASE_FAMILY,
    REALNVP_REARRANGED_FAMILY,
    REALNVP_CDF_FAMILY,
    REALNVP_LOG_PROBABILITY_FAMILY,
)

YAML_BLOCKS = (
    ("name", "seed", "save_directory"),
    ("dataset_config",),
    ("predictor_config", "predictor_checkpoint"),
    ("trainer_config",),
    ("rearrangement_config",),
    ("rearrangement_trainer_config",),
    ("supervised_rearrangement",),
    ("conformal_config",),
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


def _dataset_config(dataset: str, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": dataset,
        "file_path": spec["file_path"],
        **({"raw_root": spec["raw_root"]} if "raw_root" in spec else {}),
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "train_fraction": 0.6,
        "calibration_fraction": 0.2,
        "test_fraction": 0.2,
        "seed": seed,
        "device": "cuda",
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


def _rearrangement_config(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "amortized_rearranged_transport",
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


def _realnvp_trainer_config() -> dict[str, Any]:
    return {
        "epochs": 250,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip_norm": 10.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }


def _neural_ot_trainer_config(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "epochs": 250,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "warmup_iterations": 50,
        "grad_clip_norm": spec["neural_ot_grad_clip_norm"],
        "use_cosine_scheduler": True,
        "verbose": True,
    }


def _conformal_config(family: str, seed: int) -> dict[str, Any]:
    if family == REALNVP_CDF_FAMILY:
        calibrator = {
            "type": "cdf_calibrator",
            "n_cdf_samples": 32,
            "cdf_batch_size": 65_536,
        }
    elif family == REALNVP_LOG_PROBABILITY_FAMILY:
        calibrator = {"type": "log_probability"}
    else:
        calibrator = {"type": "norm", "p": 2.0}

    return {
        "type": "transport_based",
        "coverage_mass": 0.9,
        "calibrator": calibrator,
        "volume_mc_samples": 1_000,
        "volume_batch_size": 1_024,
        **(
            {"volume_n_neighbors": 100}
            if family in {REALNVP_CDF_FAMILY, REALNVP_LOG_PROBABILITY_FAMILY}
            else {}
        ),
        "volume_seed": seed,
    }


def _tags(dataset: str, family: str, is_rearranged: bool) -> list[str]:
    tags = [dataset, SUITE, "transport"]
    if family == NEURAL_OT_REARRANGED_FAMILY:
        tags.extend(["neural-ot", "l2"])
    else:
        tags.extend(["normalizing-flow", "realnvp"])
        if family in {REALNVP_BASE_FAMILY, REALNVP_REARRANGED_FAMILY}:
            tags.append("l2")
        elif family == REALNVP_CDF_FAMILY:
            tags.append("cdf")
        else:
            tags.append("log-probability")
    if is_rearranged:
        tags.extend(["rearranged", "amortized", "sparse", "dopri5"])
    return tags


def make_config(*, dataset: str, family: str, seed: int) -> dict[str, Any]:
    """Build one transport/rearrangement experiment configuration."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")
    if family not in FAMILIES:
        raise ValueError(f"Unsupported family: {family!r}.")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be between 0 and 4, got {seed}.")

    spec = DATASETS[dataset]
    is_neural_ot = family == NEURAL_OT_REARRANGED_FAMILY
    is_rearranged = family in {
        NEURAL_OT_REARRANGED_FAMILY,
        REALNVP_REARRANGED_FAMILY,
    }
    predictor_config = (
        _neural_ot_predictor(spec, seed)
        if is_neural_ot
        else _realnvp_predictor(spec, seed)
    )
    config: dict[str, Any] = {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": f"benchmark/results/{dataset}/{SUITE}/{family}",
        "dataset_config": _dataset_config(dataset, spec, seed),
        "predictor_config": predictor_config,
        "trainer_config": (
            _neural_ot_trainer_config(spec)
            if is_neural_ot
            else _realnvp_trainer_config()
        ),
    }

    if is_neural_ot:
        config["predictor_checkpoint"] = (
            f"benchmark/results/{dataset}/sweep_neural_ot_run/"
            f"transport_neural_ot_l2_huge/seed_{seed:02d}/base/predictor.pt"
        )
    elif family != REALNVP_BASE_FAMILY:
        config["predictor_checkpoint"] = (
            f"benchmark/results/{dataset}/{SUITE}/{REALNVP_BASE_FAMILY}/"
            f"seed_{seed:02d}/base/predictor.pt"
        )

    if is_rearranged:
        config.update(
            rearrangement_config=_rearrangement_config(spec, seed),
            rearrangement_trainer_config=_rearrangement_trainer_config(),
            supervised_rearrangement=False,
        )

    config.update(
        conformal_config=_conformal_config(family, seed),
        train_batch_size=spec["train_batch_size"],
        **(
            {
                "rearrangement_train_batch_size": spec[
                    "rearrangement_train_batch_size"
                ]
            }
            if is_rearranged
            else {}
        ),
        calibration_batch_size=512,
        test_batch_size=512,
        compute_volume=True,
        metrics_verbose=True,
        wandb={
            "mode": "online",
            "project": "minimal-volume-conformal-prediction",
            "entity": None,
            "group": f"{dataset}/{SUITE}/{family}",
            "name": f"{family}_seed_{seed:02d}",
            "tags": _tags(dataset, family, is_rearranged),
            "job_type": family,
            "log_every_n_steps": 20,
            "log_solver_diagnostics": is_rearranged,
        },
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
