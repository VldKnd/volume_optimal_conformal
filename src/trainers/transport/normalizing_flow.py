# src/trainers/transport/normalizing_flow.py

import time

import torch
from tqdm import trange

from configs.trainers.transport.normalizing_flow import (
    NormalizingFlowTrainerConfig,
)
from predictors.transport.normalizing_flow import NormalizingFlowPredictor
from trainers.base import BaseTrainer


class NormalizingFlowTrainer(BaseTrainer):
    config_class = NormalizingFlowTrainerConfig
    trainer_type = "normalizing_flow_trainer"

    def __init__(
        self,
        config: NormalizingFlowTrainerConfig,
    ):
        super().__init__(config)

    def fit(
        self,
        predictor: NormalizingFlowPredictor,
        dataloader: torch.utils.data.DataLoader,
        max_epochs: int | None = None,
    ) -> NormalizingFlowPredictor:
        end_epoch = self._fit_end_epoch(max_epochs)
        if self.completed_epochs >= end_epoch:
            predictor.eval()
            return predictor

        steps_per_epoch = len(dataloader)
        self._validate_steps_per_epoch(steps_per_epoch)
        self._restore_rng_state()

        if not self.initialization_complete:
            predictor.warmup_y_scaler(dataloader)
            self.initialization_complete = True

        predictor.y_scaler.eval()
        predictor.flow_layers.train()

        optimizer, scheduler = self._setup_optimization(
            predictor.flow_layers.named_parameters(prefix="flow_layers"),
            steps_per_epoch=steps_per_epoch,
            predictor=predictor,
        )

        progress = trange(
            self.completed_epochs,
            end_epoch,
            disable=not self.config.verbose,
            desc="Normalizing Flow",
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
                    raise FloatingPointError("Non-finite normalizing flow loss.")

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.flow_layers.parameters(),
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
                    "negative_log_likelihood": epoch_loss,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            self.completed_epochs = epoch + 1

            if self.config.verbose:
                progress.set_description(f"Epoch {epoch + 1} | NLL {epoch_loss:.4f}")

        predictor.eval()

        return predictor
