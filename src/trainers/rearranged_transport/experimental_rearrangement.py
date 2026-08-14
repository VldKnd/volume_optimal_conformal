from __future__ import annotations

import math
import time

import torch
from tqdm import trange

from configs.trainers.rearranged_transport.experimental_rearrangement import (
    ExperimentalRearrangementTrainerConfig,
)
from predictors.rearranged_transport.amortized_rearranged_transport import (
    AmortizedRearrangedTransport,
)
from predictors.rearranged_transport.rearranged_transport import (
    RearrangedTransportPredictor,
)
from trainers.base import BaseTrainer
from trainers.rearranged_transport.rearranged_transport import (
    RearrangedTransportTrainer,
)


class ExperimentalRearrangementTrainer(RearrangedTransportTrainer):
    """Train a fixed rearrangement with a paired shell-consistency objective.

    For each context, two latent points with independent directions share a
    radius sampled from the Chi distribution. For ``s = S_theta(u, x)`` and
    frozen base transport ``T_x``, each point contributes

        -log phi(s) + log |det D T_x(s)|,

    where ``phi`` is the standard Gaussian density. The loss is the mean
    per-context L2 difference between the two contributions.
    """

    config_class = ExperimentalRearrangementTrainerConfig
    trainer_type = "experimental_rearrangement_trainer"

    def __init__(self, config: ExperimentalRearrangementTrainerConfig):
        if not isinstance(config, ExperimentalRearrangementTrainerConfig):
            raise TypeError(
                "config must be an ExperimentalRearrangementTrainerConfig "
                "instance."
            )
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
                "dataloader must be provided to train experimental "
                "rearranged transport."
            )
        if not isinstance(predictor, RearrangedTransportPredictor):
            raise TypeError(
                "predictor must be a RearrangedTransportPredictor instance."
            )
        if isinstance(predictor, AmortizedRearrangedTransport):
            raise TypeError(
                "ExperimentalRearrangementTrainer requires a non-amortized "
                "rearrangement."
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
        if isinstance(predictor.transport_predictor, torch.nn.Module):
            predictor.transport_predictor.requires_grad_(False)
        optimizer, scheduler = self._setup_optimization(
            predictor.rearrangement_flow.named_parameters(
                prefix="rearrangement_flow"
            ),
            steps_per_epoch=steps_per_epoch,
            predictor=predictor,
        )

        progress = trange(
            self.completed_epochs,
            end_epoch,
            disable=not self.config.verbose,
            desc="Experimental Rearrangement",
        )

        for epoch in progress:
            start = time.perf_counter()
            epoch_losses: list[torch.Tensor] = []
            epoch_negative_log_probabilities: list[torch.Tensor] = []
            epoch_transport_log_dets: list[torch.Tensor] = []

            for batch in dataloader:
                batch_start = (
                    time.perf_counter() if self.live_metrics_enabled else None
                )
                self._reset_solver_diagnostics(predictor)
                x = predictor.to_device(self._extract_x_batch(batch))
                paired_u = self._sample_same_radius_pairs(
                    batch_size=x.shape[0],
                    dimension=predictor.y_dim,
                    device=predictor.device,
                    dtype=predictor.dtype,
                )
                repeated_x = x.repeat_interleave(2, dim=0)

                point_losses, negative_log_probabilities, transport_log_dets = (
                    self.pointwise_loss_components(
                        predictor=predictor,
                        x=repeated_x,
                        u=paired_u.flatten(0, 1),
                    )
                )
                paired_point_losses = point_losses.reshape(x.shape[0], 2)
                paired_l2_differences = self._paired_l2_differences(
                    paired_point_losses
                )
                loss = paired_l2_differences.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite experimental rearrangement loss."
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

                epoch_losses.append(paired_l2_differences.detach().cpu())
                epoch_negative_log_probabilities.append(
                    negative_log_probabilities.detach().cpu()
                )
                epoch_transport_log_dets.append(transport_log_dets.detach().cpu())

                if batch_start is not None:
                    batch_metrics = {
                        "epoch": epoch + 1,
                        "loss": loss.detach(),
                        "ell_0": paired_point_losses[:, 0].detach().mean(),
                        "ell_1": paired_point_losses[:, 1].detach().mean(),
                        "negative_log_probability": (
                            negative_log_probabilities.detach().mean()
                        ),
                        "transport_log_det": transport_log_dets.detach().mean(),
                        "gradient_norm": gradient_norm,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "batch_time": time.perf_counter() - batch_start,
                    }
                    batch_metrics.update(self._rearrangement_flow_metrics(predictor))
                    self._record_batch(batch_metrics)

            epoch_loss = float(torch.cat(epoch_losses).mean())
            epoch_negative_log_probability = float(
                torch.cat(epoch_negative_log_probabilities).mean()
            )
            epoch_transport_log_det = float(
                torch.cat(epoch_transport_log_dets).mean()
            )
            self._record_epoch(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "negative_log_probability": epoch_negative_log_probability,
                    "transport_log_det": epoch_transport_log_det,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch + 1} | Paired loss {epoch_loss:.4f}"
                )

        predictor.eval()
        return predictor

    @staticmethod
    def _sample_same_radius_pairs(
        batch_size: int,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        directions = torch.randn(
            batch_size,
            2,
            dimension,
            device=device,
            dtype=dtype,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(dtype).eps
        )
        radii = torch.randn(
            batch_size,
            dimension,
            device=device,
            dtype=dtype,
        ).norm(dim=-1, keepdim=True)
        return directions * radii.unsqueeze(-1)

    @staticmethod
    def _paired_l2_differences(
        paired_point_losses: torch.Tensor,
    ) -> torch.Tensor:
        if paired_point_losses.ndim != 2 or paired_point_losses.shape[1] != 2:
            raise ValueError(
                "paired_point_losses must have shape (batch_size, 2), got "
                f"{tuple(paired_point_losses.shape)}."
            )
        differences = paired_point_losses[:, 0] - paired_point_losses[:, 1]
        return torch.linalg.vector_norm(differences.unsqueeze(-1), dim=-1)

    def pointwise_loss(
        self,
        predictor: RearrangedTransportPredictor,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        losses, _, _ = self.pointwise_loss_components(
            predictor=predictor,
            x=x,
            u=u,
        )
        return losses

    def pointwise_loss_components(
        self,
        predictor: RearrangedTransportPredictor,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rearranged_u = predictor.rearrangement_pushforward(x=x, u=u)
        negative_log_probabilities = 0.5 * (
            rearranged_u.square().sum(dim=-1)
            + predictor.y_dim * math.log(2.0 * math.pi)
        )
        transport_log_dets = predictor.transport_log_det(
            x=x,
            u=rearranged_u,
        ).reshape(-1)
        negative_log_probabilities = negative_log_probabilities.reshape(-1)

        if transport_log_dets.shape != negative_log_probabilities.shape:
            raise ValueError(
                "transport_log_det must return one value per rearranged point; "
                f"got {tuple(transport_log_dets.shape)} for "
                f"{negative_log_probabilities.numel()} points."
            )

        return (
            negative_log_probabilities + transport_log_dets,
            negative_log_probabilities,
            transport_log_dets,
        )
