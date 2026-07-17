# src/predictors/transport/neural_optimal_transport.py

from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn

from configs.predictors.transport.neural_optimal_transport import (
    NeuralOptimalTransportPredictorConfig,
)
from predictors.transport.base import BaseTransportPredictor
from networks.picnn import PISCNN


class NeuralOptimalTransportPredictor(nn.Module, BaseTransportPredictor):
    """
    Conditional neural quantile regression predictor.

    It learns a conditional convex potential and defines

        pushforward(x, u) = T_x(u),
        pullback(x, y)    = T_x^{-1}(y).

    The multivariate score is

        S_multi(x, y) = pullback(x, y).
    """

    def __init__(self, config: NeuralOptimalTransportPredictorConfig):
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        torch.manual_seed(config.seed)

        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self.potential_type = config.potential_type

        self.potential_network = PISCNN(
            feature_dimension=config.x_dim,
            response_dimension=config.y_dim,
            hidden_dimension=config.hidden_dim,
            number_of_hidden_layers=config.num_hidden_layers,
        ).to(device=self.device, dtype=self.dtype)

        self.y_scaler = nn.BatchNorm1d(
            config.y_dim,
            affine=False,
        ).to(device=self.device, dtype=self.dtype)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def warmup_y_scaler(self, dataloader) -> None:
        self.y_scaler.train()

        for _, y_batch in dataloader:
            y_batch = self.to_device(y_batch)
            _ = self.y_scaler(y_batch)

        self.y_scaler.eval()

    def scale_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.y_scaler(y)

    def unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return (
            y_scaled
            * torch.sqrt(self.y_scaler.running_var + self.y_scaler.eps)
            + self.y_scaler.running_mean
        )

    @torch.enable_grad()
    def gradient_inverse(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes gradient of the learned potential wrt point.
        """
        x = self.to_device(x)
        point = self.to_device(point).detach().clone().requires_grad_(True)

        potential = self.potential_network.forward(
            condition=x,
            tensor=point,
        ).sum()

        grad = torch.autograd.grad(
            potential,
            point,
            create_graph=False,
        )[0]

        return grad.detach()

    def c_transform_inverse(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximately solves

            argmin_z phi_x(z) - <point, z>

        using LBFGS, with Adam fallback.
        """
        x = self.to_device(x)
        point = self.to_device(point)

        inverse = torch.nn.Parameter(
            point.detach().clone().contiguous()
        )

        optimizer = torch.optim.LBFGS(
            [inverse],
            lr=self.config.c_transform_lr,
            max_iter=self.config.c_transform_max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-7,
        )

        def closure():
            optimizer.zero_grad()

            inner = (point * inverse).sum(dim=-1, keepdim=True)
            potential = self.potential_network.forward(
                condition=x,
                tensor=inverse,
            )

            objective = (potential - inner).mean()

            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite c-transform objective.")

            objective.backward()
            torch.nn.utils.clip_grad_norm_([inverse], max_norm=10.0)

            return objective

        try:
            optimizer.step(closure)

        except (FloatingPointError, RuntimeError):
            inverse = torch.nn.Parameter(
                point.detach().clone().contiguous()
            )

            fallback = torch.optim.Adam([inverse], lr=1e-2)

            for _ in range(self.config.c_transform_max_iter):
                fallback.zero_grad()

                inner = (point * inverse).sum(dim=-1, keepdim=True)
                potential = self.potential_network.forward(
                    condition=x,
                    tensor=inverse,
                )

                objective = (potential - inner).mean()

                if not torch.isfinite(objective):
                    break

                objective.backward()
                torch.nn.utils.clip_grad_norm_([inverse], max_norm=10.0)
                fallback.step()

        return inverse.detach()

    def estimate_psi(
        self,
        x: torch.Tensor,
        y_scaled: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.potential_type == "y":
            return self.potential_network(
                condition=x,
                tensor=y_scaled,
            )

        if u is None:
            raise ValueError("u must be provided when potential_type='u'.")

        return (
            (y_scaled * u).sum(dim=-1, keepdim=True)
            - self.potential_network(condition=x, tensor=u)
        )

    def estimate_phi(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y_scaled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.potential_type == "u":
            return self.potential_network(
                condition=x,
                tensor=u,
            )

        if y_scaled is None:
            raise ValueError("y_scaled must be provided when potential_type='y'.")

        return (
            (y_scaled * u).sum(dim=-1, keepdim=True)
            - self.potential_network(condition=x, tensor=y_scaled)
        )

    @torch.enable_grad()
    def pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pull y back to latent u.

        Args:
            x: (batch, x_dim)
            y: (batch, y_dim)

        Returns:
            u: (batch, y_dim)
        """
        self.eval()

        x = self.to_device(x)
        y_scaled = self.scale_y(self.to_device(y))

        if self.potential_type == "y":
            u = self.gradient_inverse(x=x, point=y_scaled)
        else:
            u = self.c_transform_inverse(x=x, point=y_scaled)

        return u.detach()

    @torch.enable_grad()
    def pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Push latent u to y.

        Args:
            x: (batch, x_dim)
            u: (batch, y_dim)

        Returns:
            y: (batch, y_dim)
        """
        self.eval()

        x = self.to_device(x)
        u = self.to_device(u)

        if self.potential_type == "u":
            y_scaled = self.gradient_inverse(x=x, point=u)
        else:
            y_scaled = self.c_transform_inverse(x=x, point=u)

        return self.unscale_y(y_scaled).detach()

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.pullback(x=x, y=y)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.model_dump(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> Self:
        data = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        config = NeuralOptimalTransportPredictorConfig.model_validate(data["config"])
        model = cls(config)
        model.load_state_dict(data["state_dict"])

        return model