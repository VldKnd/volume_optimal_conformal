"""Generate the transport benchmark suite (5 seeds, five real datasets).

Three transport bases (neural OT, RealNVP, flow matching) and their
non-amortized rearrangements on bio, blog, qm9, scm20d and onp. The RealNVP
and neural OT settings come from the CPU pilots in
``profiling/realnvp_diagnostics`` (``FINDINGS.md``): training-time target and
covariate noise, the covariate-only scale conditioning of the coupling layers
and batch 512 instead of 4096. Flow matching has no pilot history and uses the
predictor defaults with the same training schedule as the other bases.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = REPOSITORY_ROOT / "benchmark/configurations"

SUITE = "transport_benchmark"
SEEDS = range(5)

NEURAL_OT_L2 = "neural_ot_l2"
REALNVP_L2 = "realnvp_l2"
FLOW_MATCHING_L2 = "flow_matching_l2"
NEURAL_OT_REARRANGED_L2 = "neural_ot_rearranged_l2"
REALNVP_REARRANGED_L2 = "realnvp_rearranged_l2"
FLOW_MATCHING_REARRANGED_L2 = "flow_matching_rearranged_l2"

BASE_METHODS = (NEURAL_OT_L2, REALNVP_L2, FLOW_MATCHING_L2)
REARRANGED_METHODS = (
    NEURAL_OT_REARRANGED_L2,
    REALNVP_REARRANGED_L2,
    FLOW_MATCHING_REARRANGED_L2,
)
METHODS = BASE_METHODS + REARRANGED_METHODS

BASE_METHOD_OF = {
    NEURAL_OT_L2: NEURAL_OT_L2,
    REALNVP_L2: REALNVP_L2,
    FLOW_MATCHING_L2: FLOW_MATCHING_L2,
    NEURAL_OT_REARRANGED_L2: NEURAL_OT_L2,
    REALNVP_REARRANGED_L2: REALNVP_L2,
    FLOW_MATCHING_REARRANGED_L2: FLOW_MATCHING_L2,
}

NEURAL_OT_FAMILY = frozenset({NEURAL_OT_L2, NEURAL_OT_REARRANGED_L2})
REALNVP_FAMILY = frozenset({REALNVP_L2, REALNVP_REARRANGED_L2})
FLOW_MATCHING_FAMILY = frozenset({FLOW_MATCHING_L2, FLOW_MATCHING_REARRANGED_L2})

DATASETS: dict[str, dict[str, Any]] = {
    "bio": {
        "file_path": "data/bio/CASP.csv",
        "x_dim": 8,
        "y_dim": 2,
    },
    "blog": {
        "file_path": "data/blog/blogData_train.csv",
        "x_dim": 279,
        "y_dim": 2,
    },
    "qm9": {
        "file_path": "data/qm9/qm9_mbtr_100.npz",
        "raw_root": "data/qm9/pyg",
        "x_dim": 100,
        "y_dim": 4,
    },
    "scm20d": {
        "file_path": "data/scm20d/file1730492b4408.arff",
        "x_dim": 61,
        "y_dim": 16,
    },
    "onp": {
        "file_path": "data/onp/OnlineNewsPopularity.csv",
        "x_dim": 29,
        "y_dim": 8,
    },
}

# RealNVP: the per-dataset recipe from the two-seed CPU pilots
# (profiling/realnvp_diagnostics/FINDINGS.md, pilot_summary.md). Target noise
# is in units of each target's standard deviation, covariate noise in
# standardised covariate units; "x_only" conditions every coupling's
# log-scale on the covariates alone so the log-determinant is constant over
# the latent ball. Batch 512 replaces 4096 wherever the pilots showed the
# 4096 fits to be under-trained (bio, blog, qm9, onp).
REALNVP_SETTINGS: dict[str, dict[str, Any]] = {
    # bio: batch 512 + target noise 0.1 gives 4.355 (two seeds) against
    # 4.47 for the retrained benchmark setting; covariate noise and the
    # x-only scale cost 0.04 to 0.09 there.
    "bio": {
        "train_batch_size": 512,
        "predictor": {},
        "trainer": {
            "target_noise_std": 0.1
        },
    },
    # blog: 279 covariates and 31k rows, overfits within 50 epochs at every
    # milder setting; covariate noise 0.5 with target noise 0.1 at batch 512
    # gives 2.76 and 2.84 (two seeds) against 3.26 for the benchmark setting
    # (neural OT 2.86), and the x-only scale does not help there.
    "blog": {
        "train_batch_size": 512,
        "predictor": {},
        "trainer": {
            "covariate_noise_std": 0.5,
            "target_noise_std": 0.1
        },
    },
    # qm9: batch 512 alone gives -0.14 (two seeds) against -0.05 for the
    # benchmark setting; every noise variant over-smooths it (target noise
    # 0.1: +0.12, covariate noise 0.2: -0.07).
    "qm9": {
        "train_batch_size": 512,
        "predictor": {},
        "trainer": {},
    },
    # scm20d: x-only scale with both noises, 5.73 on three seeds against
    # 6.84 (neural OT 5.84).
    "scm20d": {
        "train_batch_size": 256,
        "predictor": {
            "scale_conditioning": "x_only"
        },
        "trainer": {
            "covariate_noise_std": 0.2,
            "target_noise_std": 0.1
        },
    },
    # onp: batch 512 with the x-only scale and both noises, 0.970 and 0.974
    # on two seeds against 1.04 to 1.16 for batch 512 alone (neural OT 1.04).
    "onp": {
        "train_batch_size": 512,
        "predictor": {
            "scale_conditioning": "x_only"
        },
        "trainer": {
            "covariate_noise_std": 0.2,
            "target_noise_std": 0.1
        },
    },
}

# Neural OT: the 10-seed benchmark settings unchanged. The noise pilot
# (seed 0, scm20d and onp) found every noise variant 0.00 to 0.07 nats per
# dimension worse than the retrained baseline: neural OT has no Jensen gap
# to remove, so the noise only inflates its map.
NEURAL_OT_SETTINGS: dict[str, dict[str, Any]] = {
    "bio": {
        "train_batch_size": 4_096,
        "trainer": {
            "grad_clip_norm": 10.0
        }
    },
    "blog": {
        "train_batch_size": 4_096,
        "trainer": {
            "grad_clip_norm": 10.0
        }
    },
    "qm9": {
        "train_batch_size": 4_096,
        "trainer": {
            "grad_clip_norm": 10.0
        }
    },
    "scm20d": {
        "train_batch_size": 256,
        "trainer": {
            "grad_clip_norm": 20.0
        }
    },
    "onp": {
        "train_batch_size": 4_096,
        "trainer": {
            "grad_clip_norm": 10.0
        }
    },
}

# Flow matching: predictor defaults (128 x 3 vector field, 100 RK4 steps,
# exact divergence) with the RealNVP batch size of each dataset.
FLOW_MATCHING_SETTINGS: dict[str, dict[str, Any]] = {
    dataset: {
        "train_batch_size": settings["train_batch_size"]
    }
    for dataset, settings in REALNVP_SETTINGS.items()
}

REARRANGEMENT_TRAIN_BATCH_SIZE = {
    "bio": 512,
    "blog": 512,
    "qm9": 512,
    "scm20d": 256,
    "onp": 512,
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


def _dataset_config(dataset: str, spec: dict[str, Any], seed: int) -> dict[str, Any]:
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
        "cuda",
        "dtype":
        "float32",
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


def _realnvp_predictor(
    dataset: str,
    spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "type": "normalizing_flow",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "hidden_dim": 64,
        "num_hidden_layers": 4,
        "num_flow_layers": 4,
        "log_scale_bound": 3.0,
        "scale_conditioning": "x_and_u",
        **REALNVP_SETTINGS[dataset].get("predictor", {}),
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def _flow_matching_predictor(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "type": "flow_matching",
        "x_dim": spec["x_dim"],
        "y_dim": spec["y_dim"],
        "hidden_dim": 128,
        "num_hidden_layers": 3,
        "ode_steps": 100,
        "use_hutchinson_trace_estimator": False,
        "seed": seed,
        "device": "cuda",
        "dtype": "float32",
    }


def _trainer_config(dataset: str, method: str) -> dict[str, Any]:
    trainer: dict[str, Any] = {
        "epochs": 250,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip_norm": 10.0,
        "covariate_noise_std": 0.0,
        "target_noise_std": 0.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }
    if method in NEURAL_OT_FAMILY:
        trainer["warmup_iterations"] = 50
        trainer.update(NEURAL_OT_SETTINGS[dataset].get("trainer", {}))
    elif method in REALNVP_FAMILY:
        trainer.update(REALNVP_SETTINGS[dataset].get("trainer", {}))
    else:
        trainer.update(FLOW_MATCHING_SETTINGS[dataset].get("trainer", {}))
    return trainer


def _train_batch_size(dataset: str, method: str) -> int:
    if method in NEURAL_OT_FAMILY:
        return NEURAL_OT_SETTINGS[dataset]["train_batch_size"]
    if method in REALNVP_FAMILY:
        return REALNVP_SETTINGS[dataset]["train_batch_size"]
    return FLOW_MATCHING_SETTINGS[dataset]["train_batch_size"]


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


def flow_matching_log_det_options(y_dim: int) -> dict[str, Any]:
    """Training-time log-det estimate for a flow matching base.

    The rearrangement loss differentiates log |det D T_x| through the whole
    ODE solve, so the 100 exact RK4 steps used at evaluation are replaced by
    20 steps during training; above two target dimensions the exact
    divergence (one reverse-mode call per dimension) is replaced by a single
    Rademacher Hutchinson probe, which is unbiased for the loss. Evaluation
    keeps the predictor's configured exact solve.
    """
    options: dict[str, Any] = {"ode_steps": 20}
    if y_dim > 2:
        options["use_hutchinson_trace_estimator"] = True
    return options


def _rearrangement_trainer_config(method: str, spec: dict[str, Any]) -> dict[str, Any]:
    trainer = {
        "epochs": 100,
        "mc_samples_per_x": 32,
        "train_transport_map": False,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip_norm": 1.0,
        "use_cosine_scheduler": True,
        "verbose": True,
    }
    if method == FLOW_MATCHING_REARRANGED_L2:
        trainer["transport_log_det_options"] = flow_matching_log_det_options(
            spec["y_dim"]
        )
    return trainer


def _conformal_config(seed: int) -> dict[str, Any]:
    return {
        "type": "transport_based",
        "coverage_mass": 0.9,
        "calibrator": {
            "type": "norm",
            "p": 2.0
        },
        "volume_mc_samples": 1_000,
        "volume_batch_size": 1_024,
        "volume_n_neighbors": 100,
        "volume_seed": seed,
    }


def _wandb_config(dataset: str, method: str, seed: int) -> dict[str, Any]:
    tags = [dataset, SUITE, "transport"]
    if method in NEURAL_OT_FAMILY:
        tags.extend(["neural-ot", "l2"])
    elif method in REALNVP_FAMILY:
        tags.extend(["normalizing-flow", "realnvp", "l2"])
    else:
        tags.extend(["flow-matching", "l2"])

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
    """Build one transport benchmark experiment configuration."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method!r}.")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be between 0 and 4, got {seed}.")

    spec = DATASETS[dataset]
    if method in NEURAL_OT_FAMILY:
        predictor = _neural_ot_predictor(spec, seed)
    elif method in REALNVP_FAMILY:
        predictor = _realnvp_predictor(dataset, spec, seed)
    else:
        predictor = _flow_matching_predictor(spec, seed)

    config: dict[str, Any] = {
        "name": f"seed_{seed:02d}",
        "seed": seed,
        "save_directory": f"benchmark/results/{SUITE}/{dataset}/{method}",
        "dataset_config": _dataset_config(dataset, spec, seed),
        "predictor_config": predictor,
        "trainer_config": _trainer_config(dataset, method),
    }

    if method in REARRANGED_METHODS:
        config["predictor_checkpoint"] = (
            f"benchmark/results/{SUITE}/{dataset}/{BASE_METHOD_OF[method]}/"
            f"seed_{seed:02d}/base/predictor.pt"
        )
        config.update(
            rearrangement_config=_rearrangement_config(spec, seed),
            rearrangement_trainer_config=_rearrangement_trainer_config(method, spec),
            supervised_rearrangement=False,
        )

    config.update(
        conformal_config=_conformal_config(seed),
        train_batch_size=_train_batch_size(dataset, method),
        **(
            {
                "rearrangement_train_batch_size":
                REARRANGEMENT_TRAIN_BATCH_SIZE[dataset]
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
