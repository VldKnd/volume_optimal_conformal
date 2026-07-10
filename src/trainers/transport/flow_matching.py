# src/trainers/flow_matching.py

import time

import torch
from tqdm import trange

from configs.trainers.transport.flow_matching import FlowMatchingTrainerConfig
from predictors.transport.flow_matching import FlowMatchingPredictor
from trainers.base import BaseTrainer


class FlowMatchingTrainer(BaseTrainer):
    def __init__(
        self,
        config: FlowMatchingTrainerConfig,
    ):
        self.config = config
        self.training_history: list[dict] = []

    def fit(
        self,
        predictor: FlowMatchingPredictor,
        dataloader: torch.utils.data.DataLoader,
    ) -> FlowMatchingPredictor:

        predictor.warmup_y_scaler(dataloader)
        predictor.y_scaler.eval()
        predictor.vector_field.train()

        optimizer = torch.optim.AdamW(
            predictor.vector_field.parameters(),
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
            desc="Flow Matching",
        )

        for epoch in progress:

            start = time.perf_counter()
            epoch_losses = []

            for x_batch, y_batch in dataloader:

                x_batch = predictor.to_device(x_batch)
                y_batch = predictor.to_device(y_batch)

                y_scaled = predictor.scale_y(y_batch)
                u = torch.randn_like(y_scaled)

                t = torch.rand(
                    y_scaled.shape[0],
                    1,
                    device=y_scaled.device,
                    dtype=y_scaled.dtype,
                )

                state = (1.0 - t) * u + t * y_scaled
                target_velocity = y_scaled - u

                prediction = predictor.predict_vector_field(
                    state=state,
                    x=x_batch,
                    t=t,
                )

                loss = (
                    (prediction - target_velocity)
                    .square()
                    .sum(dim=-1)
                    .mean()
                )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.parameters(),
                    self.config.grad_clip_norm,
                )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(loss.item())

            epoch_loss = float(torch.tensor(epoch_losses).mean())

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "flow_matching_loss": epoch_loss,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch+1} | Loss {epoch_loss:.4f}"
                )

        predictor.eval()

        return predictor