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

    One coverage mass is sampled uniformly from ``(0, 1)`` for every context
    point. The same value determines both the latent chi-ball radius and the
    coverage context supplied to the rearrangement vector field. Latent points
    follow the standard Gaussian conditioned on that ball. The objective is
    the direct sample mean of the full composition log-Jacobian.
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
            epoch_training_losses: list[torch.Tensor] = []
            coverage_mass_batches: list[torch.Tensor] = []
            radius_batches: list[torch.Tensor] = []

            for batch in dataloader:
                batch_start = (
                    time.perf_counter() if self.live_metrics_enabled else None
                )
                self._reset_solver_diagnostics(predictor)
                x_batch = self._extract_x_batch(batch)
                x_batch = predictor.to_device(x_batch)
                coverage_masses = self._sample_coverage_masses(
                    predictor=predictor,
                    batch_size=x_batch.shape[0],
                )
                radii = self._ball_radii(
                    coverage_masses=coverage_masses,
                    dimension=predictor.y_dim,
                )
                x = self._repeat_context(
                    x=x_batch,
                    mc_samples_per_x=self.config.mc_samples_per_x,
                )
                repeated_coverage_masses = coverage_masses.repeat_interleave(
                    self.config.mc_samples_per_x,
                )
                repeated_radii = radii.repeat_interleave(self.config.mc_samples_per_x, )
                u = self._sample_training_points(
                    batch_size=x.shape[0],
                    dimension=predictor.y_dim,
                    coverage_masses=repeated_coverage_masses,
                    maximum_radii=repeated_radii,
                    device=predictor.device,
                    dtype=predictor.dtype,
                )

                training_losses = self.estimate_training_losses(
                    predictor=predictor,
                    x=x,
                    u=u,
                    coverage_mass=repeated_coverage_masses,
                    mc_samples_per_x=self.config.mc_samples_per_x,
                )
                loss = training_losses.mean()

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite amortized rearranged transport loss."
                    )

                optimizer.zero_grad()
                loss.backward()

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    predictor.rearrangement_flow.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

                optimizer.step()
                self.global_step += 1

                if scheduler is not None:
                    scheduler.step()

                loss_value = float(loss.detach().cpu())
                epoch_training_losses.append(training_losses.detach().cpu())
                coverage_mass_batches.append(coverage_masses.detach().cpu())
                radius_batches.append(radii.detach().cpu())
                if batch_start is not None:
                    batch_metrics = {
                        "epoch": epoch + 1,
                        "loss": loss_value,
                        "mean_log_det_loss": loss_value,
                        "gradient_norm": gradient_norm,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "coverage_mass": float(coverage_masses.mean()),
                        "coverage_mass_min": float(coverage_masses.min()),
                        "coverage_mass_max": float(coverage_masses.max()),
                        "radius": float(radii.mean()),
                        "radius_min": float(radii.min()),
                        "radius_max": float(radii.max()),
                        "mc_samples_per_x": self.config.mc_samples_per_x,
                        "batch_time": time.perf_counter() - batch_start,
                    }
                    batch_metrics.update(self._rearrangement_flow_metrics(predictor))
                    self._record_batch(batch_metrics)

            training_loss_tensor = torch.cat(epoch_training_losses)
            coverage_tensor = torch.cat(coverage_mass_batches)
            radius_tensor = torch.cat(radius_batches)
            epoch_training_loss = float(training_loss_tensor.mean())

            self._record_epoch(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_training_loss,
                    "mean_log_det_loss": epoch_training_loss,
                    "coverage_mass_mean": float(coverage_tensor.mean()),
                    "coverage_mass_min": float(coverage_tensor.min()),
                    "coverage_mass_max": float(coverage_tensor.max()),
                    "radius_mean": float(radius_tensor.mean()),
                    "radius_min": float(radius_tensor.min()),
                    "radius_max": float(radius_tensor.max()),
                    "mc_samples_per_x": self.config.mc_samples_per_x,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch + 1} | Mean log-det {epoch_training_loss:.4f} "
                    f"| Mean coverage {float(coverage_tensor.mean()):.3f}"
                )

        predictor.eval()
        return predictor

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
