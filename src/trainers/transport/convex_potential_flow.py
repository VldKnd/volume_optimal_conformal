import time

import torch
from tqdm import trange

from configs.trainers.transport.convex_potential_flow import (
    ConvexPotentialFlowTrainerConfig,
)
from predictors.transport.convex_potential_flow import (
    ConvexPotentialFlowPredictor,
)
from trainers.base import BaseTrainer


class ConvexPotentialFlowTrainer(BaseTrainer):
    """Maximum-likelihood trainer for a convex-potential flow."""

    config_class = ConvexPotentialFlowTrainerConfig
    trainer_type = "convex_potential_flow_trainer"

    def __init__(self, config: ConvexPotentialFlowTrainerConfig):
        super().__init__(config)

    def fit(
        self,
        predictor: ConvexPotentialFlowPredictor,
        dataloader: torch.utils.data.DataLoader,
        max_epochs: int | None = None,
    ) -> ConvexPotentialFlowPredictor:
        end_epoch = self._fit_end_epoch(max_epochs)
        if end_epoch <= self.completed_epochs:
            predictor.eval()
            return predictor

        steps_per_epoch = len(dataloader)
        self._validate_steps_per_epoch(steps_per_epoch)
        self._restore_rng_state()

        predictor.train()
        optimizer, scheduler = self._setup_optimization(
            predictor.flow.named_parameters(prefix="flow"),
            steps_per_epoch=steps_per_epoch,
            predictor=predictor,
        )

        progress = trange(
            self.completed_epochs,
            end_epoch,
            disable=not self.config.verbose,
            desc="Convex Potential Flow",
        )

        for epoch in progress:
            start = time.perf_counter()
            epoch_losses: list[float] = []

            for x_batch, y_batch in dataloader:
                x_batch = predictor.to_device(x_batch)
                y_batch = predictor.to_device(y_batch)

                loss = -predictor.log_prob(x=x_batch, y=y_batch).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite convex-potential flow loss.")

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    predictor.flow.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )
                optimizer.step()

                self.global_step += 1
                self.initialization_complete = True

                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(float(loss.detach().cpu()))

            epoch_loss = float(torch.tensor(epoch_losses).mean())
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "surrogate_negative_log_likelihood": epoch_loss,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            self.completed_epochs = epoch + 1

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch + 1} | Surrogate NLL {epoch_loss:.4f}"
                )

        predictor.eval()
        return predictor
