"""Generate Neural OT Student-t benchmark configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = REPOSITORY_ROOT / "benchmark/configurations/student_t"

NU = 3
K_VALUES = (2, 4, 8, 16)
Y_DIMS = (2, 4, 8, 16)
SEEDS = range(5)
FAMILIES = (
    "transport_neural_ot_l2",
    "transport_neural_ot_rearranged_l2",
)

DEVICE = "cuda"
DTYPE = "float32"
OT_WARMUP_ITERATIONS = 10

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


def parameter_slug(nu: int, k: int, y_dim: int) -> str:
    """Return the directory name for one Student-t parameter setting."""
    return f"nu_{nu}_k_{k}_y_dim_{y_dim}"


def make_config(
    *,
    nu: int,
    k: int,
    y_dim: int,
    family: str,
    seed: int,
) -> dict[str, Any]:
    """Build one validated-run-compatible Student-t experiment mapping."""
    if family not in FAMILIES:
        raise ValueError(f"Unsupported Student-t benchmark family: {family!r}.")
    if k not in K_VALUES or y_dim not in Y_DIMS or nu != NU:
        raise ValueError(
            f"Unsupported Student-t parameters: nu={nu}, k={k}, y_dim={y_dim}."
        )
    if seed not in SEEDS:
        raise ValueError(f"Student-t benchmark seed must be in 0..4, got {seed}.")

    setting = parameter_slug(nu=nu, k=k, y_dim=y_dim)
    result_root = f"benchmark/results/student_t/{setting}"
    is_rearranged = family == "transport_neural_ot_rearranged_l2"

    config: dict[str, Any] = {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": f"{result_root}/{family}",
        "dataset_config": {
            "type": "student_t_dataset",
            "n_train": 20_000,
            "n_calibration": 1_000,
            "n_test": 1_000,
            "x_dim": 1,
            "y_dim": y_dim,
            "nu": float(nu),
            "k": float(k),
            "seed": seed,
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "predictor_config": {
            "type": "neural_optimal_transport",
            "x_dim": 1,
            "y_dim": y_dim,
            "hidden_dim": 16,
            "num_hidden_layers": 8,
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
    }

    if is_rearranged:
        config["predictor_checkpoint"] = (
            f"{result_root}/transport_neural_ot_l2/seed_{seed:02d}"
            "/base/predictor.pt"
        )
        config["rearrangement_config"] = {
            "type": "amortized_rearranged_transport",
            "x_dim": 1,
            "y_dim": y_dim,
            "hidden_dimension": 32,
            "number_of_hidden_layers": 9,
            "use_adjoint": False,
            "method": "dopri5",
            "rtol": 0.0001,
            "atol": 0.00001,
            "vector_field_implementation": "sparse",
            "activation": "silu",
            "seed": seed,
            "device": DEVICE,
            "dtype": DTYPE,
        }
        config["rearrangement_trainer_config"] = {
            "epochs": 100,
            "train_transport_map": False,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "mc_samples_per_x": 16,
            "grad_clip_norm": 1.0,
            "use_cosine_scheduler": True,
            "verbose": True,
        }
        config["supervised_rearrangement"] = False

    config["conformal_config"] = {
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
    config.update(
        {
            "train_batch_size":
            1_024,
            **({
                "rearrangement_train_batch_size": 1_024
            } if is_rearranged else {}),
            "calibration_batch_size":
            512,
            "test_batch_size":
            512,
            "compute_volume":
            True,
            "metrics_verbose":
            True,
            "wandb":
            _wandb_config(
                setting=setting,
                family=family,
                seed=seed,
                is_rearranged=is_rearranged,
            ),
        }
    )
    return config


def _wandb_config(
    *,
    setting: str,
    family: str,
    seed: int,
    is_rearranged: bool,
) -> dict[str, Any]:
    tags = [
        "student-t",
        "synthetic",
        "neural-ot",
        "transport",
        "l2",
        setting,
    ]
    if is_rearranged:
        tags.extend(["rearranged", "amortized", "sparse", "dopri5"])

    return {
        "mode": "online",
        "project": "minimal-volume-conformal-prediction",
        "group": f"student_t/{setting}/{family}",
        "name": f"{setting}_{family}_seed_{seed:02d}",
        "tags": tags,
        "job_type": family,
        "log_every_n_steps": 20,
        "log_solver_diagnostics": is_rearranged,
    }


def main() -> None:
    count = 0
    for y_dim in Y_DIMS:
        for k in K_VALUES:
            setting = parameter_slug(nu=NU, k=k, y_dim=y_dim)
            for family in FAMILIES:
                family_directory = CONFIGURATION_ROOT / setting / family
                family_directory.mkdir(parents=True, exist_ok=True)
                for seed in SEEDS:
                    config = make_config(
                        nu=NU,
                        k=k,
                        y_dim=y_dim,
                        family=family,
                        seed=seed,
                    )
                    path = family_directory / f"seed_{seed:02d}.yaml"
                    path.write_text(dump_config(config), encoding="utf-8")
                    count += 1

    print(f"Generated {count} Student-t benchmark configurations.")


if __name__ == "__main__":
    main()
