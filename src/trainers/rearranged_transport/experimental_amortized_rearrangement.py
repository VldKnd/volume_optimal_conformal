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
    """Train an amortized rearrangement using mean log-volume weights.

    This replaces the standard per-context ``logsumexp(weights) - log(n)``
    estimator with the direct arithmetic mean of the log-det weights.
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
