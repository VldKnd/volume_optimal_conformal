from __future__ import annotations

import torch

from configs.trainers.rearranged_transport.experimental_amortized_rearrangement import (
    ExperimentalAmortizedRearrangementTrainerConfig,
)
from predictors.rearranged_transport.amortized_rearranged_transport import (
    AmortizedRearrangedTransport,
)
from trainers.rearranged_transport.amortized_rearranged_transport import (
    AmortizedRearrangedTransportTrainer,
)


class ExperimentalAmortizedRearrangementTrainer(
    AmortizedRearrangedTransportTrainer
):
    """Train an amortized rearrangement over truncated Gaussian balls.

    For each selected coverage mass, samples follow the standard Gaussian
    conditioned on its corresponding chi ball. The per-context objective is
    the direct Monte Carlo mean of the full composition log-Jacobian

        log |det D (T_x o S_x)(u)|.
    """

    config_class = ExperimentalAmortizedRearrangementTrainerConfig
    trainer_type = "experimental_amortized_rearrangement_trainer"

    def __init__(
        self,
        config: ExperimentalAmortizedRearrangementTrainerConfig,
    ):
        if not isinstance(
            config,
            ExperimentalAmortizedRearrangementTrainerConfig,
        ):
            raise TypeError(
                "config must be an "
                "ExperimentalAmortizedRearrangementTrainerConfig instance."
            )
        super().__init__(config)

    def estimate_log_volumes(
        self,
        predictor: AmortizedRearrangedTransport,
        x: torch.Tensor,
        u: torch.Tensor,
        coverage_mass: torch.Tensor | float,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        weights = predictor.log_det(
            x=x,
            u=u,
            coverage_mass=coverage_mass,
        )
        return self._grouped_mean(
            weights=weights,
            mc_samples_per_x=mc_samples_per_x,
        )

    def _sample_latent_points(
        self,
        batch_size: int,
        dimension: int,
        coverage_masses: torch.Tensor,
        maximum_radii: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        directions = torch.randn(
            batch_size,
            dimension,
            device=device,
            dtype=dtype,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(dtype).eps
        )

        conditional_masses = torch.rand(
            batch_size,
            device=device,
            dtype=dtype,
        ) * coverage_masses
        sampled_radii = self._ball_radii(
            coverage_masses=conditional_masses,
            dimension=dimension,
        )
        sampled_radii = torch.minimum(sampled_radii, maximum_radii)
        return directions * sampled_radii.unsqueeze(-1)

    @staticmethod
    def _grouped_mean(
        weights: torch.Tensor,
        mc_samples_per_x: int,
    ) -> torch.Tensor:
        if mc_samples_per_x < 1:
            raise ValueError("mc_samples_per_x must be positive.")

        weights = weights.reshape(-1)
        if weights.numel() % mc_samples_per_x != 0:
            raise ValueError(
                "Number of log-det weights must be divisible by "
                f"mc_samples_per_x={mc_samples_per_x}, got {weights.numel()}."
            )

        return weights.reshape(-1, mc_samples_per_x).mean(dim=1)
