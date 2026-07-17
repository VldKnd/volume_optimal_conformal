# src/trainers/neural_quantile.py

import time

import torch
from tqdm import trange

from configs.trainers.transport.neural_optimal_transport import NeuralOptimalTransportTrainerConfig
from predictors.transport.neural_optimal_transport import NeuralOptimalTransportPredictor
from trainers.base import BaseTrainer


class NeuralQuantileTrainer(BaseTrainer):
    def __init__(self, config: NeuralOptimalTransportTrainerConfig):
        self.config = config
        self.training_history: list[dict] = []

    def fit(
        self,
        predictor: NeuralOptimalTransportPredictor,
        dataloader: torch.utils.data.DataLoader,
    ) -> NeuralOptimalTransportPredictor:
        predictor.warmup_y_scaler(dataloader)
        predictor.y_scaler.eval()

        self._warmup_network(predictor, dataloader)

        predictor.potential_network.train()

        optimizer = torch.optim.AdamW(
            predictor.potential_network.parameters(),
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
            1,
            self.config.epochs + 1,
            disable=not self.config.verbose,
            desc="Training Neural Quantile",
        )

        for epoch in progress:
            start = time.perf_counter()
            epoch_losses: list[float] = []

            for x_batch, y_batch in dataloader:
                x_batch = predictor.to_device(x_batch)
                y_batch = predictor.to_device(y_batch)

                y_scaled = predictor.scale_y(y_batch)
                u = torch.randn_like(y_scaled)

                if predictor.potential_type == "y":
                    inverse = predictor.c_transform_inverse(
                        x=x_batch,
                        point=u,
                    )
                    y_for_phi = inverse
                    u_for_psi = None

                else:
                    inverse = predictor.c_transform_inverse(
                        x=x_batch,
                        point=y_scaled,
                    )
                    y_for_phi = None
                    u_for_psi = inverse

                psi = predictor.estimate_psi(
                    x=x_batch,
                    y_scaled=y_scaled,
                    u=u_for_psi,
                )

                phi = predictor.estimate_phi(
                    x=x_batch,
                    u=u,
                    y_scaled=y_for_phi,
                )

                loss = phi.mean() + psi.mean()

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.potential_network.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(float(loss.detach().cpu()))

            epoch_loss = float(torch.tensor(epoch_losses).mean())

            self.training_history.append(
                {
                    "epoch": epoch,
                    "potential_loss": epoch_loss,
                    "epoch_time": time.perf_counter() - start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if self.config.verbose:
                progress.set_description(
                    f"Epoch {epoch} | Potential loss {epoch_loss:.4f}"
                )

        predictor.potential_network.eval()
        return predictor

    def _warmup_network(
        self,
        predictor: NeuralOptimalTransportPredictor,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        if self.config.warmup_iterations <= 0:
            return

        predictor.potential_network.train()

        optimizer = torch.optim.AdamW(
            predictor.potential_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        progress = trange(
            1,
            self.config.warmup_iterations + 1,
            disable=not self.config.verbose,
            desc="Warming Neural Quantile",
        )

        for iteration in progress:
            losses = []

            for x_batch, y_batch in dataloader:
                x_batch = predictor.to_device(x_batch)
                y_batch = predictor.to_device(y_batch)

                y_scaled = predictor.scale_y(y_batch)
                u = torch.randn_like(y_scaled)

                optimizer.zero_grad()

                if predictor.potential_type == "y":
                    point = y_scaled.detach().clone().requires_grad_(True)
                else:
                    point = u.detach().clone().requires_grad_(True)

                potential = predictor.potential_network(
                    condition=x_batch,
                    tensor=point,
                )

                grad = torch.autograd.grad(
                    potential.sum(),
                    point,
                    create_graph=True,
                )[0]

                loss = (grad - point).norm(dim=-1).mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.potential_network.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            if self.config.verbose:
                progress.set_description(
                    f"Warmup {iteration} | Loss {torch.tensor(losses).mean():.4f}"
                )