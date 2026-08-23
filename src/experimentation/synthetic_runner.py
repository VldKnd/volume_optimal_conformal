from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from scipy.stats import chi2

from configs.calibrators import NoCalibratorConfig
from configs.conformal import TransportBasedConformalPredictorConfig
from conformal import TransportBasedConformalPredictor
from experimentation.config import ExperimentConfig
from experimentation.runner import ExperimentRunner

SUPPORTED_SYNTHETIC_DATASET_TYPES = frozenset(
    {
        "banana",
        "sinusoidal_transport",
        "star_shaped_gaussian",
        "star_shaped",
    }
)


def validate_synthetic_experiment_config(config: Any) -> None:
    """Validate the assumptions required by the analytic HDR comparison."""
    dataset_config = config.dataset_config
    dataset_type = dataset_config.type
    if dataset_type not in SUPPORTED_SYNTHETIC_DATASET_TYPES:
        supported = ", ".join(sorted(SUPPORTED_SYNTHETIC_DATASET_TYPES))
        raise ValueError(
            "Synthetic benchmark dataset must be one of "
            f"{{{supported}}}, got {dataset_type!r}."
        )
    if dataset_config.y_dim < 2 or dataset_config.y_dim % 2 != 0:
        raise ValueError(
            "The pairwise synthetic HDR metric requires a positive even "
            "y_dim, got "
            f"{dataset_config.y_dim}."
        )
    if config.predictor_config.type == "random_forest":
        raise ValueError("The synthetic HDR metric requires a transport predictor.")
    if config.conformal_config.type != "transport_based":
        raise ValueError(
            "The synthetic HDR metric requires transport-based conformal prediction."
        )


def compute_hdr_volume_ratio(
    predictor: Any,
    condition: torch.Tensor,
    coverage_mass: float,
    *,
    number_of_samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Compare the analytic HDR volume with ``Vol(T(B(0, r)))``.

    The supported datasets are even-dimensional pushforwards of a standard
    Gaussian by pairwise unit-determinant ground-truth maps. Their
    probability-mass ``coverage_mass`` HDR therefore has the same volume as
    the Gaussian ball whose squared radius is the corresponding chi-square
    quantile. The learned transport volume is estimated by Monte Carlo
    integration of its forward Jacobian over that same latent ball.
    """
    if not 0.0 < coverage_mass < 1.0:
        raise ValueError("coverage_mass must be in (0, 1).")
    dimension = getattr(predictor, "y_dim", None)
    if (
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2
        or dimension % 2 != 0
    ):
        raise ValueError(
            "The pairwise synthetic HDR metric requires a positive even y_dim."
        )
    if not isinstance(condition, torch.Tensor):
        raise TypeError("condition must be a torch.Tensor.")
    expected_shape = (1, getattr(predictor, "x_dim", None))
    if tuple(condition.shape) != expected_shape:
        raise ValueError(
            f"Expected condition shape {expected_shape}, got "
            f"{tuple(condition.shape)}."
        )

    radius_squared = float(chi2.ppf(coverage_mass, df=dimension))
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise RuntimeError("The Gaussian HDR Chi-square quantile is invalid.")
    radius = math.sqrt(radius_squared)
    log_hdr_volume = (
        0.5 * dimension * math.log(math.pi) + dimension * math.log(radius) -
        math.lgamma(0.5 * dimension + 1.0)
    )
    hdr_volume = math.exp(log_hdr_volume)

    analytic_region = TransportBasedConformalPredictor(
        predictor=predictor,
        config=TransportBasedConformalPredictorConfig(
            coverage_mass=coverage_mass,
            calibrator=NoCalibratorConfig(),
            volume_mc_samples=number_of_samples,
            volume_batch_size=batch_size,
            volume_seed=seed,
        ),
    )
    log_transport_volume = float(
        analytic_region.estimate_log_volume(
            condition,
            number_of_samples=number_of_samples,
            batch_size=batch_size,
            seed=seed,
        )[0].detach().cpu()
    )
    transport_volume = math.exp(log_transport_volume)
    ratio = math.exp(log_hdr_volume - log_transport_volume)

    return {
        "mean": ratio,
        "std": None,
        "hdr_volume": hdr_volume,
        "transport_ball_volume": transport_volume,
        "gaussian_ball_radius": radius,
        "coverage_mass": coverage_mass,
        "volume_mc_samples": number_of_samples,
    }


class SyntheticExperimentRunner(ExperimentRunner):
    """Regular experiment runner with an analytic synthetic-volume metric."""

    def __init__(
        self,
        config: ExperimentConfig,
        source_config_path: str | Path | None = None,
    ):
        validate_synthetic_experiment_config(config)
        super().__init__(config=config, source_config_path=source_config_path)

    def _run(self) -> SyntheticExperimentRunner:
        super()._run()

        final_predictor = (
            self.rearrangement if self.rearrangement is not None else self.predictor
        )
        if final_predictor is None or self.dataset is None:
            raise RuntimeError("The trained transport and dataset are unavailable.")

        conformal_config = self.config.conformal_config
        comparison = compute_hdr_volume_ratio(
            predictor=final_predictor,
            condition=self.dataset.sample_x(1),
            coverage_mass=conformal_config.coverage_mass,
            number_of_samples=conformal_config.volume_mc_samples,
            batch_size=conformal_config.volume_batch_size,
            seed=conformal_config.volume_seed,
        )
        comparison["transport_stage"] = (
            "rearrangement" if self.rearrangement is not None else "base"
        )
        self.metrics["hdr_volume_ratio"] = comparison
        self._write_json(self.run_directory / "metrics.json", self.metrics)
        self._log_tracking_metrics(
            "evaluation",
            {"hdr_volume_ratio": comparison},
        )

        if self.config.metrics_verbose:
            print(
                "[metrics] Analytic HDR / transport-ball volume: "
                f"{comparison['mean']:.6f} "
                f"({comparison['hdr_volume']:.6f} / "
                f"{comparison['transport_ball_volume']:.6f}).",
                flush=True,
            )

        return self
