from __future__ import annotations

import time

import torch
from tqdm import trange

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

    One coverage mass is sampled uniformly from ``(0, 1)`` for every
    minibatch. The same value determines both the latent chi-ball radius and
    the coverage context supplied to the rearrangement vector field.
    """

    config_class = AmortizedRearrangedTransportTrainerConfig
    trainer_type = "amortized_rearranged_transport_trainer"

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
        if dataloader is None:
            raise ValueError(
                "dataloader must be provided to train amortized rearranged "
                "transport."
            )

        if not isinstance(predictor, AmortizedRearrangedTransport):
            raise TypeError(
                "predictor must be an AmortizedRearrangedTransport instance."
            )

        end_epoch = self._fit_end_epoch(max_epochs)
        if end_epoch <= self.completed_epochs:
            predictor.eval()
            return predictor

        steps_per_epoch = len(dataloader)
        self._validate_steps_per_epoch(steps_per_epoch)
        self._restore_rng_state()

        if not self.initialization_complete:
            self._fit_transport_map_if_requested(
                predictor=predictor,
                dataloader=dataloader,
                transport_trainer=transport_trainer,
            )
            self.initialization_complete = True

        predictor.train()
        predictor.transport_predictor.eval()

        optimizer, scheduler = self._setup_optimization(
            predictor.rearrangement_flow.named_parameters(prefix="rearrangement_flow"),
            steps_per_epoch=steps_per_epoch,
            predictor=predictor,
        )

        progress = trange(
            self.completed_epochs,
            end_epoch,
            disable=not self.config.verbose,
            desc="Amortized Rearranged Transport",
        )

        for epoch in progress:
            start = time.perf_counter()
            epoch_losses: list[float] = []
            coverage_levels: list[float] = []
            radii: list[float] = []

            for batch in dataloader:
                x_batch = self._extract_x_batch(batch)
                x_batch = predictor.to_device(x_batch)
                x = self._repeat_context(
                    x=x_batch,
                    mc_samples_per_x=self.config.mc_samples_per_x,
                )

                alpha = self._sample_coverage_level(predictor)
                alpha_value = float(alpha.item())
                radius = self._ball_radius(
                    alpha=alpha_value,
                    dimension=predictor.y_dim,
                )
                u = self._sample_uniform_ball(
                    batch_size=x.shape[0],
                    dimension=predictor.y_dim,
                    radius=radius,
                    device=predictor.device,
                    dtype=predictor.dtype,
                )

                loss = self.estimate_log_volume(
                    predictor=predictor,
                    x=x,
                    u=u,
                    alpha=alpha,
                    mc_samples_per_x=self.config.mc_samples_per_x,
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite amortized rearranged transport loss."
                    )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.rearrangement_flow.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

                optimizer.step()
                self.global_step += 1

                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(float(loss.detach().cpu()))
                coverage_levels.append(alpha_value)
                radii.append(radius)

            epoch_loss = float(torch.tensor(epoch_losses).mean())
            coverage_tensor = torch.tensor(coverage_levels)
            radius_tensor = torch.tensor(radii)

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "log_volume_loss": epoch_loss,
                    "coverage_level_mean": float(coverage_tensor.mean()),
                    "coverage_level_min": float(coverage_tensor.min()),
                    "coverage_level_max": float(coverage_tensor.max()),
                    "radius_mean": float(radius_tensor.mean()),
                    "radius_min": float(radius_tensor.min()),
                    "radius_max": float(radius_tensor.max()),
                    "mc_samples_per_x": self.config.mc_samples_per_x,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            self.completed_epochs = epoch + 1

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch + 1} | Log-volume {epoch_loss:.4f} "
                    f"| Mean coverage {float(coverage_tensor.mean()):.3f}"
                )

        predictor.eval()
        return predictor

    def estimate_log_volume(
        self,
        predictor: AmortizedRearrangedTransport,
        x: torch.Tensor,
        u: torch.Tensor,
        alpha: torch.Tensor | float,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        weights = predictor.log_det(
            x=x,
            u=u,
            alpha=alpha,
        )

        return self._grouped_log_mean_exp(
            weights=weights,
            mc_samples_per_x=mc_samples_per_x,
        ).mean()

    @staticmethod
    def _sample_coverage_level(
        predictor: AmortizedRearrangedTransport,
    ) -> torch.Tensor:
        epsilon = torch.finfo(predictor.dtype).eps
        return torch.empty(
            (),
            device=predictor.device,
            dtype=predictor.dtype,
        ).uniform_(epsilon, 1.0 - epsilon)
