# src/trainers/flow_matching.py

import time

import torch
from tqdm import trange

from configs.trainers.transport.flow_matching import FlowMatchingTrainerConfig
from predictors.transport.flow_matching import FlowMatchingPredictor
from trainers.base import BaseTrainer


class FlowMatchingTrainer(BaseTrainer):
    config_class = FlowMatchingTrainerConfig
    trainer_type = "flow_matching_trainer"

    def __init__(
        self,
        config: FlowMatchingTrainerConfig,
    ):
        super().__init__(config)

    def fit(
        self,
        predictor: FlowMatchingPredictor,
        dataloader: torch.utils.data.DataLoader,
        max_epochs: int | None = None,
    ) -> FlowMatchingPredictor:
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
        predictor.vector_field.train()

        optimizer, scheduler = self._setup_optimization(
            predictor.vector_field.named_parameters(prefix="vector_field"),
            steps_per_epoch=steps_per_epoch,
            predictor=predictor,
        )

        progress = trange(
            self.completed_epochs,
            end_epoch,
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

                loss = ((prediction - target_velocity).square().sum(dim=-1).mean())

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.parameters(),
                    self.config.grad_clip_norm,
                )

                optimizer.step()
                self.global_step += 1

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
            self.completed_epochs = epoch + 1

            if self.config.verbose:
                progress.set_description(f"Epoch {epoch+1} | Loss {epoch_loss:.4f}")

        predictor.eval()

        return predictor
