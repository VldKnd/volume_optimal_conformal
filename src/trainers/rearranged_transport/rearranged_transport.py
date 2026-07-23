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

    config_class = RearrangedTransportTrainerConfig
    trainer_type = "rearranged_transport_trainer"

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
            raise ValueError(
                "dataloader must be provided to train rearranged transport."
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
            desc="Rearranged Transport",
        )

        for epoch in progress:
            start = time.perf_counter()
            epoch_losses: list[float] = []

            for batch in dataloader:
                x_batch = self._extract_x_batch(batch)
                x_batch = predictor.to_device(x_batch)
                x = self._repeat_context(
                    x=x_batch,
                    mc_samples_per_x=self.config.mc_samples_per_x,
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
                    mc_samples_per_x=self.config.mc_samples_per_x,
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite rearranged transport loss.")

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

            epoch_loss = float(torch.tensor(epoch_losses).mean())

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "log_volume_loss": epoch_loss,
                    "radius": radius,
                    "coverage_mass": self.config.coverage_mass,
                    "mc_samples_per_x": self.config.mc_samples_per_x,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            self.completed_epochs = epoch + 1

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch + 1} | Log-volume {epoch_loss:.4f}"
                )

        predictor.eval()
        return predictor

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

    def _grouped_log_mean_exp(
        self,
        weights: torch.Tensor,
        mc_samples_per_x: int,
    ) -> torch.Tensor:
        if mc_samples_per_x < 1:
            raise ValueError("mc_samples_per_x must be positive.")

        if weights.ndim != 1:
            weights = weights.reshape(-1)

        if weights.numel() % mc_samples_per_x != 0:
            raise ValueError(
                "Number of log-det weights must be divisible by "
                f"mc_samples_per_x={mc_samples_per_x}, got {weights.numel()}."
            )

        grouped_weights = weights.reshape(-1, mc_samples_per_x)
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

    def _sample_uniform_ball(
        self,
        batch_size: int,
        dimension: int,
        radius: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        direction = torch.randn(
            batch_size,
            dimension,
            device=device,
            dtype=dtype,
        )
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(dtype).eps
        )

        radial = torch.rand(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        ).pow(1.0 / dimension)

        return radius * radial * direction

    def _ball_radius(
        self,
        coverage_mass: float,
        dimension: int,
    ) -> float:
        return float(chi.ppf(coverage_mass, df=dimension))


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
                    seen_samples += int(inside_ball.numel())
                    accepted_samples += int(inside_ball.sum().item())

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

                torch.nn.utils.clip_grad_norm_(
                    predictor.rearrangement_flow.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

                optimizer.step()
                self.global_step += 1

                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(float(loss.detach().cpu()))

            if not epoch_losses:
                raise RuntimeError(
                    "No dataloader samples were accepted inside the latent ball. "
                    "Increase config.coverage_mass or use a larger "
                    "batch/dataset."
                )

            epoch_loss = float(torch.tensor(epoch_losses).mean())
            acceptance_rate = accepted_samples / max(seen_samples, 1)

            self.training_history.append(
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
            self.completed_epochs = epoch + 1

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
