import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from configs.trainers.regression import (
    MLPTrainerConfig,
    NearestNeighborsTrainerConfig,
    RandomForestTrainerConfig,
)
from predictors.regression import (
    MLPPredictor,
    NearestNeighborsPredictor,
    RandomForestPredictor,
)
from trainers.base import BaseTrainer


def _fit_sklearn_trainer(
    trainer,
    predictor,
    dataloader,
    max_epochs,
):
    if trainer._fit_end_epoch(max_epochs) <= trainer.completed_epochs:
        return predictor

    trainer._validate_steps_per_epoch(len(dataloader))
    trainer._restore_rng_state()
    start = time.perf_counter()

    x_batches = []
    y_batches = []
    for x_batch, y_batch in dataloader:
        x_batches.append(x_batch.detach().cpu())
        y_batches.append(y_batch.detach().cpu())

    x = torch.cat(x_batches).numpy()
    y = torch.cat(y_batches).numpy()
    target = y[:, 0] if predictor.y_dim == 1 else y

    predictor.model.fit(x, target)
    prediction = predictor.model.predict(x).reshape(-1, predictor.y_dim)
    mse = float(np.mean((prediction - y)**2))

    trainer.steps_per_epoch = len(dataloader)
    trainer.completed_epochs = 1
    trainer.global_step = 1
    trainer.initialization_complete = True
    trainer.training_history.append(
        {
            "epoch": 1,
            "mean_squared_error": mse,
            "training_time": time.perf_counter() - start,
        }
    )
    return predictor


class RandomForestTrainer(BaseTrainer):
    config_class = RandomForestTrainerConfig
    trainer_type = "random_forest_trainer"

    def fit(
        self,
        predictor: RandomForestPredictor,
        dataloader: torch.utils.data.DataLoader,
        max_epochs: int | None = None,
    ) -> RandomForestPredictor:
        return _fit_sklearn_trainer(
            self,
            predictor,
            dataloader,
            max_epochs,
        )


class MLPTrainer(BaseTrainer):
    config_class = MLPTrainerConfig
    trainer_type = "mlp_trainer"

    def fit(
        self,
        predictor: MLPPredictor,
        dataloader: torch.utils.data.DataLoader,
        max_epochs: int | None = None,
    ) -> MLPPredictor:
        end_epoch = self._fit_end_epoch(max_epochs)
        if end_epoch <= self.completed_epochs:
            predictor.eval()
            return predictor

        steps_per_epoch = len(dataloader)
        self._validate_steps_per_epoch(steps_per_epoch)
        self._restore_rng_state()
        self.initialization_complete = True

        predictor.train()
        optimizer, scheduler = self._setup_optimization(
            predictor.network.named_parameters(prefix="network"),
            steps_per_epoch=steps_per_epoch,
            predictor=predictor,
        )

        progress = trange(
            self.completed_epochs,
            end_epoch,
            disable=not self.config.verbose,
            desc="MLP Regression",
        )

        for epoch in progress:
            start = time.perf_counter()
            losses = []

            for x_batch, y_batch in dataloader:
                x_batch = predictor.to_device(x_batch)
                y_batch = predictor.to_device(y_batch)

                optimizer.zero_grad()
                loss = F.mse_loss(predictor.network(x_batch), y_batch)
                loss.backward()
                optimizer.step()

                self.global_step += 1
                if scheduler is not None:
                    scheduler.step()
                losses.append(float(loss.detach().cpu()))

            epoch_loss = float(torch.tensor(losses).mean())
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "mean_squared_error": epoch_loss,
                    "training_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            self.completed_epochs = epoch + 1

            if self.config.verbose:
                progress.set_description(f"Epoch {epoch + 1} | MSE {epoch_loss:.4f}")

        predictor.eval()
        return predictor


class NearestNeighborsTrainer(BaseTrainer):
    config_class = NearestNeighborsTrainerConfig
    trainer_type = "nearest_neighbors_trainer"

    def fit(
        self,
        predictor: NearestNeighborsPredictor,
        dataloader: torch.utils.data.DataLoader,
        max_epochs: int | None = None,
    ) -> NearestNeighborsPredictor:
        return _fit_sklearn_trainer(
            self,
            predictor,
            dataloader,
            max_epochs,
        )
