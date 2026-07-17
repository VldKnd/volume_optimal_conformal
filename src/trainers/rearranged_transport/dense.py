# src/trainers/rearranged_transport/dense.py

import math
import time

import torch
from scipy.stats import chi
from tqdm import trange

from configs.trainers.rearranged_transport.dense import (
    DenseRearrangedTransportTrainerConfig,
)
from predictors.rearranged_transport.dense import (
    DenseRearrangedTransportPredictor,
)
from trainers.base import BaseTrainer


class DenseRearrangedTransportTrainer(BaseTrainer):

    def __init__(
        self,
        config: DenseRearrangedTransportTrainerConfig,
    ):
        self.config = config
        self.training_history: list[dict] = []

    def fit(
        self,
        predictor: DenseRearrangedTransportPredictor,
        dataloader: torch.utils.data.DataLoader,
        transport_trainer: BaseTrainer | None = None,
    ) -> DenseRearrangedTransportPredictor:
        if dataloader is None:
            raise ValueError(
                "dataloader must be provided to train dense rearranged transport."
            )

        self._fit_transport_map_if_requested(
            predictor=predictor,
            dataloader=dataloader,
            transport_trainer=transport_trainer,
        )

        predictor.transport_predictor.eval()
        predictor.rearrangement_flow.train()

        radius = self._ball_radius(
            alpha=self.config.alpha,
            dimension=predictor.y_dim,
        )

        optimizer = torch.optim.AdamW(
            predictor.rearrangement_flow.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = None
        if self.config.use_cosine_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs * len(dataloader),
            )

        progress = trange(
            self.config.epochs,
            disable=not self.config.verbose,
            desc="Dense Rearranged Transport",
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
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite dense rearranged transport loss."
                    )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.rearrangement_flow.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(float(loss.detach().cpu()))

            epoch_loss = float(torch.tensor(epoch_losses).mean())

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "log_volume_loss": epoch_loss,
                    "radius": radius,
                    "alpha": self.config.alpha,
                    "mc_samples_per_x": self.config.mc_samples_per_x,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch + 1} | Log-volume {epoch_loss:.4f}"
                )

        predictor.eval()
        return predictor

    def estimate_log_volume(
        self,
        predictor: DenseRearrangedTransportPredictor,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        weights = predictor.log_det(
            x=x,
            u=u,
        )

        return torch.logsumexp(weights, dim=0) - math.log(weights.numel())

    def _fit_transport_map_if_requested(
        self,
        predictor: DenseRearrangedTransportPredictor,
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
        alpha: float,
        dimension: int,
    ) -> float:
        return float(chi.ppf(alpha, df=dimension))
