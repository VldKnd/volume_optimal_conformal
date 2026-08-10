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


def validate_student_t_experiment_config(config: Any) -> None:
    """Validate the assumptions required by the Student-t benchmark."""
    if config.dataset_config.type != "student_t_dataset":
        raise ValueError(
            "Student-t benchmark requires dataset type 'student_t_dataset', "
            f"got {config.dataset_config.type!r}."
        )
    if config.predictor_config.type == "random_forest":
        raise ValueError("Student-t benchmark requires a transport predictor.")
    if config.conformal_config.type != "transport_based":
        raise ValueError(
            "Student-t benchmark requires transport-based conformal prediction."
        )


def compute_student_t_volume_comparison(
    predictor: Any,
    dataset: Any,
    coverage_mass: float,
    *,
    number_of_samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Compare the Student-t HDR with a learned Gaussian-ball pushforward.

    The learned transport is evaluated on the latent standard-Gaussian ball
    containing ``coverage_mass`` probability. Its image volume is estimated
    by Monte Carlo integration of the forward Jacobian. The target reference
    is the analytic Student-t ``coverage_mass`` HDR returned by
    ``dataset.hdr(alpha=1 - coverage_mass)``.
    """
    if not 0.0 < coverage_mass < 1.0:
        raise ValueError("coverage_mass must be in (0, 1).")
    if getattr(predictor, "y_dim", None) != getattr(dataset, "y_dim", None):
        raise ValueError("Predictor and Student-t target dimensions must match.")
    if getattr(predictor, "x_dim", None) != getattr(dataset, "x_dim", None):
        raise ValueError("Predictor and Student-t condition dimensions must match.")
    hdr_method = getattr(dataset, "hdr", None)
    if not callable(hdr_method):
        raise TypeError("Student-t dataset must implement callable hdr(alpha).")

    condition = dataset.sample_x(1)
    expected_shape = (1, predictor.x_dim)
    if not isinstance(condition, torch.Tensor):
        raise TypeError("dataset.sample_x must return a torch.Tensor.")
    if tuple(condition.shape) != expected_shape:
        raise ValueError(
            f"Expected condition shape {expected_shape}, got "
            f"{tuple(condition.shape)}."
        )

    significance_level = 1.0 - coverage_mass
    hdr = hdr_method(alpha=significance_level)
    hdr_volume = float(hdr.volume)
    if not math.isfinite(hdr_volume) or hdr_volume <= 0.0:
        raise RuntimeError("The analytic Student-t HDR volume is invalid.")

    gaussian_radius_squared = float(chi2.ppf(coverage_mass, df=predictor.y_dim))
    if not math.isfinite(gaussian_radius_squared) or gaussian_radius_squared <= 0.0:
        raise RuntimeError("The Gaussian Chi-square quantile is invalid.")

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
    transport_log_volume = float(
        analytic_region.estimate_log_volume(
            condition,
            number_of_samples=number_of_samples,
            batch_size=batch_size,
            seed=seed,
        )[0].detach().cpu()
    )
    transport_volume = math.exp(transport_log_volume)
    hdr_log_volume = math.log(hdr_volume)
    hdr_to_transport_ratio = math.exp(hdr_log_volume - transport_log_volume)

    return {
        "mean": hdr_to_transport_ratio,
        "std": None,
        "hdr_to_transport_volume_ratio": hdr_to_transport_ratio,
        "student_t_hdr_volume": hdr_volume,
        "student_t_hdr_log_volume": hdr_log_volume,
        "student_t_radius": float(hdr.radius),
        "student_t_semi_axis_lengths": (hdr.semi_axis_lengths.detach().cpu().tolist()),
        "transport_ball_volume": transport_volume,
        "transport_ball_log_volume": transport_log_volume,
        "gaussian_ball_radius": math.sqrt(gaussian_radius_squared),
        "coverage_mass": coverage_mass,
        "significance_level": significance_level,
        "volume_mc_samples": number_of_samples,
    }


class StudentTExperimentRunner(ExperimentRunner):
    """Regular experiment runner with an analytic Student-t volume metric."""

    def __init__(
        self,
        config: ExperimentConfig,
        source_config_path: str | Path | None = None,
    ):
        validate_student_t_experiment_config(config)
        if not config.compute_volume:
            config = config.model_copy(update={"compute_volume": True})
        super().__init__(config=config, source_config_path=source_config_path)

    def _run(self) -> StudentTExperimentRunner:
        super()._run()

        final_predictor = (
            self.rearrangement if self.rearrangement is not None else self.predictor
        )
        if final_predictor is None or self.dataset is None:
            raise RuntimeError("The trained transport and dataset are unavailable.")

        conformal_config = self.config.conformal_config
        comparison = compute_student_t_volume_comparison(
            predictor=final_predictor,
            dataset=self.dataset,
            coverage_mass=conformal_config.coverage_mass,
            number_of_samples=conformal_config.volume_mc_samples,
            batch_size=conformal_config.volume_batch_size,
            seed=conformal_config.volume_seed,
        )
        comparison["transport_stage"] = (
            "rearrangement" if self.rearrangement is not None else "base"
        )
        self.metrics["student_t_hdr_volume_ratio"] = comparison
        self._write_json(self.run_directory / "metrics.json", self.metrics)
        self._log_tracking_metrics(
            "evaluation",
            {"student_t_hdr_volume_ratio": comparison},
        )

        if self.config.metrics_verbose:
            print(
                "[metrics] Student-t HDR / transport-ball volume: "
                f"{comparison['hdr_to_transport_volume_ratio']:.6f} "
                f"({comparison['student_t_hdr_volume']:.6f} / "
                f"{comparison['transport_ball_volume']:.6f}).",
                flush=True,
            )

        return self
