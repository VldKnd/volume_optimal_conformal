from __future__ import annotations

import torch

from configs.trainers.rearranged_transport.amortized_rearranged_transport import (
    AmortizedRearrangedTransportTrainerConfig,
)
from predictors.rearranged_transport.amortized_rearranged_transport import (
    AmortizedRearrangedTransport,
)
from trainers.base import BaseTrainer
from trainers.rearranged_transport.rearranged_transport import (
    RearrangedTransportTrainer,
)


class AmortizedRearrangedTransportTrainer(RearrangedTransportTrainer):
    """Train one rearrangement jointly over all coverage levels.

    One coverage mass is sampled uniformly from ``(0, 1)`` for every context
    point. The same value determines both the latent chi-ball radius and the
    coverage context supplied to the rearrangement vector field. Latent points
    follow the standard Gaussian conditioned on that ball. The objective is
    the direct sample mean of the full composition log-Jacobian.
    """

    config_class = AmortizedRearrangedTransportTrainerConfig
    trainer_type = "amortized_rearranged_transport_trainer"
    _progress_description = "Amortized Rearranged Transport"
    _missing_dataloader_message = (
        "dataloader must be provided to train amortized rearranged transport."
    )
    _non_finite_loss_message = "Non-finite amortized rearranged transport loss."

    def __init__(
        self,
        config: AmortizedRearrangedTransportTrainerConfig,
    ):
        if not isinstance(config, AmortizedRearrangedTransportTrainerConfig):
            raise TypeError(
                "config must be an AmortizedRearrangedTransportTrainerConfig "
                "instance."
            )

        super().__init__(config)

    def fit(
        self,
        predictor: AmortizedRearrangedTransport,
        dataloader: torch.utils.data.DataLoader,
        transport_trainer: BaseTrainer | None = None,
        max_epochs: int | None = None,
    ) -> AmortizedRearrangedTransport:
        return super().fit(
            predictor=predictor,
            dataloader=dataloader,
            transport_trainer=transport_trainer,
            max_epochs=max_epochs,
        )

    def _validate_fit_predictor(
        self,
        predictor: AmortizedRearrangedTransport,
    ) -> None:
        if not isinstance(predictor, AmortizedRearrangedTransport):
            raise TypeError(
                "predictor must be an AmortizedRearrangedTransport instance."
            )

    def _estimate_sampled_coverage_training_losses(
        self,
        predictor: AmortizedRearrangedTransport,
        x: torch.Tensor,
        u: torch.Tensor,
        coverage_masses: torch.Tensor,
        mc_samples_per_x: int,
    ) -> torch.Tensor:
        return self.estimate_training_losses(
            predictor=predictor,
            x=x,
            u=u,
            coverage_mass=coverage_masses,
            mc_samples_per_x=mc_samples_per_x,
        )

    def estimate_log_volume(
        self,
        predictor: AmortizedRearrangedTransport,
        x: torch.Tensor,
        u: torch.Tensor,
        coverage_mass: torch.Tensor | float,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        """Estimate log-volume from points sampled uniformly in each ball."""
        return self.estimate_log_volumes(
            predictor=predictor,
            x=x,
            u=u,
            coverage_mass=coverage_mass,
            mc_samples_per_x=mc_samples_per_x,
        ).mean()

    def estimate_log_volumes(
        self,
        predictor: AmortizedRearrangedTransport,
        x: torch.Tensor,
        u: torch.Tensor,
        coverage_mass: torch.Tensor | float,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        """Return per-context uniform-ball log-mean-exp estimates."""
        weights = predictor.log_det(
            x=x,
            u=u,
            coverage_mass=coverage_mass,
        )

        return self._grouped_log_mean_exp(
            weights=weights,
            mc_samples_per_x=mc_samples_per_x,
        )

    def estimate_training_loss(
        self,
        predictor: AmortizedRearrangedTransport,
        x: torch.Tensor,
        u: torch.Tensor,
        coverage_mass: torch.Tensor | float,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        """Estimate the loss from truncated-Gaussian latent points."""
        return self.estimate_training_losses(
            predictor=predictor,
            x=x,
            u=u,
            coverage_mass=coverage_mass,
            mc_samples_per_x=mc_samples_per_x,
        ).mean()

    def estimate_training_losses(
        self,
        predictor: AmortizedRearrangedTransport,
        x: torch.Tensor,
        u: torch.Tensor,
        coverage_mass: torch.Tensor | float,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        """Return per-context means of composition log-Jacobians."""
        weights = predictor.log_det(
            x=x,
            u=u,
            coverage_mass=coverage_mass,
        )
        return self._grouped_mean(
            weights=weights,
            mc_samples_per_x=mc_samples_per_x,
        )
