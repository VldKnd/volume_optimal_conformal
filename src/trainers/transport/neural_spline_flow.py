# src/trainers/transport/neural_spline_flow.py

import time

import torch
from tqdm import trange

from configs.trainers.transport.neural_spline_flow import (
    NeuralSplineFlowTrainerConfig,
)
from predictors.transport.neural_spline_flow import NeuralSplineFlowPredictor
from trainers.base import BaseTrainer


class NeuralSplineFlowTrainer(BaseTrainer):

    def __init__(
        self,
        config: NeuralSplineFlowTrainerConfig,
    ):
        self.config = config
        self.training_history: list[dict] = []

    def fit(
        self,
        predictor: NeuralSplineFlowPredictor,
        dataloader: torch.utils.data.DataLoader,
    ) -> NeuralSplineFlowPredictor:

        predictor.warmup_y_scaler(dataloader)
        predictor.y_scaler.eval()
        predictor.flow_layers.train()

        optimizer = torch.optim.AdamW(
            predictor.flow_layers.parameters(),
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
            desc="Neural Spline Flow",
        )

        for epoch in progress:
            start = time.perf_counter()
            epoch_losses: list[float] = []

            for x_batch, y_batch in dataloader:
                x_batch = predictor.to_device(x_batch)
                y_batch = predictor.to_device(y_batch)

                log_prob = predictor.log_prob(x=x_batch, y=y_batch)
                loss = -log_prob.mean()

                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite neural spline flow loss.")

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.flow_layers.parameters(),
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
                    "negative_log_likelihood": epoch_loss,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if self.config.verbose:
                progress.set_description(f"Epoch {epoch + 1} | NLL {epoch_loss:.4f}")

        predictor.eval()

        return predictor
