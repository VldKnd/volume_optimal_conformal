from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

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
    conformal_predictor: Any,
    dataset: Any,
    *,
    number_of_samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Compare the Student-t HDR with the fitted conformal transport region.

    The learned transport is evaluated on the empirically calibrated latent
    ball stored by ``conformal_predictor``. Its image volume is estimated by
    Monte Carlo integration of the forward Jacobian. The target reference is
    the analytic Student-t HDR at the conformal predictor's requested coverage.
    """
    coverage_mass = float(getattr(conformal_predictor, "coverage_mass", math.nan))
    if not 0.0 < coverage_mass < 1.0:
        raise ValueError("conformal_predictor.coverage_mass must be in (0, 1).")
    if getattr(conformal_predictor, "y_dim", None) != getattr(
        dataset,
        "y_dim",
        None,
    ):
        raise ValueError(
            "Conformal predictor and Student-t target dimensions must match."
        )
    if getattr(conformal_predictor, "x_dim", None) != getattr(
        dataset,
        "x_dim",
        None,
    ):
        raise ValueError(
            "Conformal predictor and Student-t condition dimensions must match."
        )
    estimate_log_volume = getattr(
        conformal_predictor,
        "estimate_log_volume",
        None,
    )
    if not callable(estimate_log_volume):
        raise TypeError(
            "conformal_predictor must implement estimate_log_volume(x, ...)."
        )

    threshold = getattr(conformal_predictor, "threshold", None)
    if threshold is None:
        raise RuntimeError(
            "The transport conformal predictor must be calibrated before the "
            "Student-t volume comparison."
        )
    threshold = torch.as_tensor(threshold).detach().reshape(-1)
    if threshold.numel() != 1:
        raise ValueError("The calibrated Student-t latent-ball radius must be scalar.")
    conformal_ball_radius = float(threshold.cpu()[0])
    if not math.isfinite(conformal_ball_radius) or conformal_ball_radius <= 0.0:
        raise RuntimeError("The calibrated Student-t latent-ball radius is invalid.")

    hdr_method = getattr(dataset, "hdr", None)
    if not callable(hdr_method):
        raise TypeError("Student-t dataset must implement callable hdr(alpha).")

    condition = dataset.sample_x(1)
    expected_shape = (1, conformal_predictor.x_dim)
    if not isinstance(condition, torch.Tensor):
        raise TypeError("dataset.sample_x must return a torch.Tensor.")
    if tuple(condition.shape) != expected_shape:
        raise ValueError(
            f"Expected condition shape {expected_shape}, got "
            f"{tuple(condition.shape)}."
        )

    significance_level = 1.0 - coverage_mass
    hdr = hdr_method(alpha=significance_level)
    hdr_log_volume = float(hdr.log_volume)
    if not math.isfinite(hdr_log_volume):
        raise RuntimeError("The analytic Student-t HDR log-volume is invalid.")

    transport_log_volume = float(
        estimate_log_volume(
            condition,
            number_of_samples=number_of_samples,
            batch_size=batch_size,
            seed=seed,
        )[0].detach().cpu()
    )
    if not math.isfinite(transport_log_volume):
        raise RuntimeError("The conformal region log-volume estimate is invalid.")

    normalized_volume_fraction = (
        transport_log_volume - hdr_log_volume
    ) / conformal_predictor.y_dim

    return {
        "mean": normalized_volume_fraction,
        "std": None,
        "normalized_log_volume_fraction": normalized_volume_fraction,
        "student_t_hdr_log_volume": hdr_log_volume,
        "conformal_region_log_volume": transport_log_volume,
        "conformal_ball_radius": conformal_ball_radius,
        "volume_region": "conformal",
        "dimension": conformal_predictor.y_dim,
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

        if self.conformal_predictor is None or self.dataset is None:
            raise RuntimeError(
                "The fitted conformal predictor and dataset are unavailable."
            )

        conformal_config = self.config.conformal_config
        comparison = compute_student_t_volume_comparison(
            conformal_predictor=self.conformal_predictor,
            dataset=self.dataset,
            number_of_samples=conformal_config.volume_mc_samples,
            batch_size=conformal_config.volume_batch_size,
            seed=conformal_config.volume_seed,
        )
        comparison["transport_stage"] = (
            "rearrangement" if self.rearrangement is not None else "base"
        )
        self.metrics["student_t_normalized_volume_fraction"] = comparison
        self._write_json(self.run_directory / "metrics.json", self.metrics)
        self._log_tracking_metrics(
            "evaluation",
            {"student_t_normalized_volume_fraction": comparison},
        )

        if self.config.metrics_verbose:
            print(
                "[metrics] Student-t normalized log-volume fraction: "
                f"{comparison['normalized_log_volume_fraction']:.6f} "
                "((log Vol(conformal) - log Vol(HDR)) / dimension).",
                flush=True,
            )

        return self
