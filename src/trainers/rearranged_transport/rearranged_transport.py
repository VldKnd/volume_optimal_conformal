# src/trainers/rearranged_transport/dense.py

import math
import time

import torch
from scipy.stats import chi
from tqdm import trange

from configs.trainers.rearranged_transport.dense import (
    RearrangedTransportTrainerConfig,
    SupervisedRearrangedTransportTrainerConfig,
)
from predictors.rearranged_transport.rearranged_transport import (
    RearrangedTransportPredictor,
)
from trainers.base import BaseTrainer


class RearrangedTransportTrainer(BaseTrainer):
    """Train one unconditioned rearrangement over all coverage levels.

    Coverage masses select truncated-Gaussian latent training points but are
    not supplied to the rearrangement vector field.
    """

    config_class = RearrangedTransportTrainerConfig
    trainer_type = "rearranged_transport_trainer"
    _progress_description = "Rearranged Transport"
    _missing_dataloader_message = (
        "dataloader must be provided to train rearranged transport."
    )
    _non_finite_loss_message = "Non-finite rearranged transport loss."

    def __init__(
        self,
        config: RearrangedTransportTrainerConfig,
    ):
        super().__init__(config)

    def fit(
        self,
        predictor: RearrangedTransportPredictor,
        dataloader: torch.utils.data.DataLoader,
        transport_trainer: BaseTrainer | None = None,
        max_epochs: int | None = None,
    ) -> RearrangedTransportPredictor:
        if dataloader is None:
            raise ValueError(self._missing_dataloader_message)

        self._validate_fit_predictor(predictor)

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
            desc=self._progress_description,
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
                repeated_radii = radii.repeat_interleave(self.config.mc_samples_per_x)
                u = self._sample_training_points(
                    batch_size=x.shape[0],
                    dimension=predictor.y_dim,
                    coverage_masses=repeated_coverage_masses,
                    maximum_radii=repeated_radii,
                    device=predictor.device,
                    dtype=predictor.dtype,
                )

                training_losses = self._estimate_sampled_coverage_training_losses(
                    predictor=predictor,
                    x=x,
                    u=u,
                    coverage_masses=repeated_coverage_masses,
                    mc_samples_per_x=self.config.mc_samples_per_x,
                )
                loss = training_losses.mean()

                if not torch.isfinite(loss):
                    raise FloatingPointError(self._non_finite_loss_message)

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
                    f"Epoch {epoch + 1} | Mean log-det "
                    f"{epoch_training_loss:.4f} | Mean coverage "
                    f"{float(coverage_tensor.mean()):.3f}"
                )

        predictor.eval()
        return predictor

    def _validate_fit_predictor(
        self,
        predictor: RearrangedTransportPredictor,
    ) -> None:
        pass

    def _estimate_sampled_coverage_training_losses(
        self,
        predictor: RearrangedTransportPredictor,
        x: torch.Tensor,
        u: torch.Tensor,
        coverage_masses: torch.Tensor,
        mc_samples_per_x: int,
    ) -> torch.Tensor:
        del coverage_masses
        return self.estimate_training_losses(
            predictor=predictor,
            x=x,
            u=u,
            mc_samples_per_x=mc_samples_per_x,
        )

    def estimate_log_volume(
        self,
        predictor: RearrangedTransportPredictor,
        x: torch.Tensor,
        u: torch.Tensor,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        weights = predictor.log_det(
            x=x,
            u=u,
        )

        return self._grouped_log_mean_exp(
            weights=weights,
            mc_samples_per_x=mc_samples_per_x,
        ).mean()

    def estimate_training_loss(
        self,
        predictor: RearrangedTransportPredictor,
        x: torch.Tensor,
        u: torch.Tensor,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        """Estimate the loss from truncated-Gaussian latent points."""
        return self.estimate_training_losses(
            predictor=predictor,
            x=x,
            u=u,
            mc_samples_per_x=mc_samples_per_x,
        ).mean()

    def estimate_training_losses(
        self,
        predictor: RearrangedTransportPredictor,
        x: torch.Tensor,
        u: torch.Tensor,
        mc_samples_per_x: int = 1,
    ) -> torch.Tensor:
        """Return per-context means without conditioning the rearrangement."""
        weights = predictor.log_det(
            x=x,
            u=u,
        )
        return self._grouped_mean(
            weights=weights,
            mc_samples_per_x=mc_samples_per_x,
        )

    def _grouped_log_mean_exp(
        self,
        weights: torch.Tensor,
        mc_samples_per_x: int,
    ) -> torch.Tensor:
        grouped_weights = self._group_weights(
            weights=weights,
            mc_samples_per_x=mc_samples_per_x,
        )
        return (torch.logsumexp(grouped_weights, dim=1) - math.log(mc_samples_per_x))

    def _fit_transport_map_if_requested(
        self,
        predictor: RearrangedTransportPredictor,
        dataloader: torch.utils.data.DataLoader | None,
        transport_trainer: BaseTrainer | None,
    ) -> None:
        if not self.config.train_transport_map:
            return

        if dataloader is None:
            raise ValueError(
                "dataloader must be provided when train_transport_map=True."
            )

        if transport_trainer is None:
            raise ValueError(
                "transport_trainer must be provided when train_transport_map=True."
            )

        transport_trainer.fit(predictor.transport_predictor, dataloader)
        predictor._move_transport_predictor_to_device()

    def _extract_x_batch(self, batch) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            return batch[0]

        raise ValueError("Expected a tensor batch or a non-empty tuple/list batch.")

    def _repeat_context(
        self,
        x: torch.Tensor,
        mc_samples_per_x: int,
    ) -> torch.Tensor:
        if mc_samples_per_x == 1:
            return x

        return x.repeat_interleave(mc_samples_per_x, dim=0)

    @staticmethod
    def _sample_coverage_masses(
        predictor: RearrangedTransportPredictor,
        batch_size: int,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        epsilon = torch.finfo(predictor.dtype).eps
        return torch.empty(
            batch_size,
            device=predictor.device,
            dtype=predictor.dtype,
        ).uniform_(epsilon, 1.0 - epsilon)

    @staticmethod
    def _ball_radii(
        coverage_masses: torch.Tensor,
        dimension: int,
    ) -> torch.Tensor:
        radii = chi.ppf(
            coverage_masses.detach().cpu().numpy(),
            df=dimension,
        )
        return torch.as_tensor(
            radii,
            device=coverage_masses.device,
            dtype=coverage_masses.dtype,
        )

    def _sample_training_points(
        self,
        batch_size: int,
        dimension: int,
        coverage_masses: torch.Tensor,
        maximum_radii: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        directions = self._sample_unit_directions(
            batch_size=batch_size,
            dimension=dimension,
            device=device,
            dtype=dtype,
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
    def _sample_unit_directions(
        batch_size: int,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        directions = torch.randn(
            batch_size,
            dimension,
            device=device,
            dtype=dtype,
        )
        return directions / directions.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(dtype).eps
        )

    def _sample_uniform_ball(
        self,
        batch_size: int,
        dimension: int,
        radius: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        directions = self._sample_unit_directions(
            batch_size=batch_size,
            dimension=dimension,
            device=device,
            dtype=dtype,
        )
        radial = torch.rand(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        ).pow(1.0 / dimension)
        return radius * radial * directions

    def _grouped_mean(
        self,
        weights: torch.Tensor,
        mc_samples_per_x: int,
    ) -> torch.Tensor:
        return self._group_weights(
            weights=weights,
            mc_samples_per_x=mc_samples_per_x,
        ).mean(dim=1)

    @staticmethod
    def _group_weights(
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

        return weights.reshape(-1, mc_samples_per_x)

    @staticmethod
    def _rearrangement_flow_metrics(
        predictor: RearrangedTransportPredictor,
    ) -> dict[str, float | int | torch.Tensor]:
        """Return inexpensive scalar diagnostics for live tracking."""
        flow = predictor.rearrangement_flow
        metrics: dict[str, float | int | torch.Tensor] = {
            "integration_end_time": flow.last_end_time,
        }

        if predictor.device.type == "cuda":
            metrics.update(
                cuda_memory_allocated=torch.cuda.memory_allocated(predictor.device),
                cuda_max_memory_allocated=torch.cuda.max_memory_allocated(
                    predictor.device
                ),
                cuda_memory_reserved=torch.cuda.memory_reserved(predictor.device),
                cuda_max_memory_reserved=torch.cuda.max_memory_reserved(
                    predictor.device
                ),
            )

        vector_field = getattr(flow, "vector_field", None)
        network = getattr(vector_field, "network", None)
        layers = getattr(network, "net", None)
        if layers is not None and len(layers) > 0:
            output_activation = layers[-1]
            alpha = getattr(output_activation, "alpha", None)
            scale = getattr(output_activation, "scale", None)
            if isinstance(alpha, torch.Tensor) and alpha.numel() == 1:
                metrics["output_alpha"] = alpha.detach()
            if isinstance(scale, torch.Tensor) and scale.numel() == 1:
                metrics["output_scale"] = scale.detach()

        solver_diagnostics = flow.solver_diagnostics_summary(reset=True)
        if solver_diagnostics is not None:
            metrics.update(
                {
                    f"solver_{key}": value
                    for key, value in solver_diagnostics.items()
                }
            )
        return metrics

    def _reset_solver_diagnostics(
        self,
        predictor: RearrangedTransportPredictor,
    ) -> None:
        if (
            self.live_metrics_enabled
            and predictor.rearrangement_flow.solver_diagnostics_enabled
        ):
            predictor.rearrangement_flow.reset_solver_diagnostics()


class SupervisedRearrangedTransportTrainer(RearrangedTransportTrainer):
    """
    Rearranged transport trainer using observed target samples for support.

    Each y from the dataloader is pulled back through the wrapped transport and
    then through the current rearrangement flow without gradient tracking. The
    resulting latent point is accepted only if it lies inside the chi-radius
    ball specified by config.coverage_mass. Accepted points use the same
    log-volume loss as RearrangedTransportTrainer.
    """

    config_class = SupervisedRearrangedTransportTrainerConfig
    trainer_type = "supervised_rearranged_transport_trainer"

    @staticmethod
    def _ball_radius(
        coverage_mass: float,
        dimension: int,
    ) -> float:
        return float(chi.ppf(coverage_mass, df=dimension))

    def fit(
        self,
        predictor: RearrangedTransportPredictor,
        dataloader: torch.utils.data.DataLoader,
        transport_trainer: BaseTrainer | None = None,
        max_epochs: int | None = None,
    ) -> RearrangedTransportPredictor:
        if dataloader is None:
            raise ValueError(
                "dataloader must be provided to train supervised "
                "rearranged transport."
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

        radius = self._ball_radius(
            coverage_mass=self.config.coverage_mass,
            dimension=predictor.y_dim,
        )

        optimizer, scheduler = self._setup_optimization(
            predictor.rearrangement_flow.named_parameters(prefix="rearrangement_flow"),
            steps_per_epoch=steps_per_epoch,
            predictor=predictor,
        )

        progress = trange(
            self.completed_epochs,
            end_epoch,
            disable=not self.config.verbose,
            desc="Supervised Rearranged Transport",
        )

        for epoch in progress:
            start = time.perf_counter()
            epoch_losses: list[float] = []
            accepted_samples = 0
            seen_samples = 0

            for batch in dataloader:
                batch_start = (
                    time.perf_counter() if self.live_metrics_enabled else None
                )
                self._reset_solver_diagnostics(predictor)
                x_batch, y_batch = self._extract_xy_batch(batch)
                x_batch = predictor.to_device(x_batch)
                y_batch = predictor.to_device(y_batch)

                with torch.no_grad():
                    transport_u = predictor.transport_pullback(
                        x=x_batch,
                        y=y_batch,
                    )
                    u = predictor.rearrangement_pullback(
                        x=x_batch,
                        u=transport_u,
                    )
                    inside_ball = u.norm(dim=-1) <= radius
                    batch_seen_samples = int(inside_ball.numel())
                    batch_accepted_samples = int(inside_ball.sum().item())
                    seen_samples += batch_seen_samples
                    accepted_samples += batch_accepted_samples

                    if not inside_ball.any():
                        continue

                    x = x_batch[inside_ball].detach()
                    u = u[inside_ball].detach()

                loss = self.estimate_log_volume(
                    predictor=predictor,
                    x=x,
                    u=u,
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite supervised rearranged transport loss."
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
                epoch_losses.append(loss_value)
                if batch_start is not None:
                    batch_metrics = {
                        "epoch": epoch + 1,
                        "loss": loss_value,
                        "log_volume_loss": loss_value,
                        "gradient_norm": gradient_norm,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "radius": radius,
                        "coverage_mass": self.config.coverage_mass,
                        "accepted_samples": batch_accepted_samples,
                        "seen_samples": batch_seen_samples,
                        "acceptance_rate":
                        (batch_accepted_samples / batch_seen_samples),
                        "batch_time": time.perf_counter() - batch_start,
                    }
                    batch_metrics.update(self._rearrangement_flow_metrics(predictor))
                    self._record_batch(batch_metrics)

            if not epoch_losses:
                raise RuntimeError(
                    "No dataloader samples were accepted inside the latent ball. "
                    "Increase config.coverage_mass or use a larger "
                    "batch/dataset."
                )

            epoch_loss = float(torch.tensor(epoch_losses).mean())
            acceptance_rate = accepted_samples / max(seen_samples, 1)

            self._record_epoch(
                {
                    "epoch": epoch + 1,
                    "log_volume_loss": epoch_loss,
                    "radius": radius,
                    "coverage_mass": self.config.coverage_mass,
                    "accepted_samples": accepted_samples,
                    "seen_samples": seen_samples,
                    "acceptance_rate": acceptance_rate,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch + 1} | Log-volume {epoch_loss:.4f} "
                    f"| Accepted {acceptance_rate:.2%}"
                )

        predictor.eval()
        return predictor

    def _extract_xy_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise ValueError(
            "Expected dataloader batches to be non-empty tuple/list pairs "
            "(x_batch, y_batch)."
        )
