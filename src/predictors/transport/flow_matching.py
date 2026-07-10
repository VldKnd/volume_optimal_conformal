# src/predictors/transport/flow_matching.py

from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn

from configs.predictors.transport.flow_matching import FlowMatchingPredictorConfig
from predictors.transport.base import BaseTransportPredictor


class MLPVectorField(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
    ):
        super().__init__()

        input_dim = x_dim + y_dim + 1

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]

        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )

        layers.append(nn.Linear(hidden_dim, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([state, x, t], dim=-1))


class FlowMatchingPredictor(nn.Module, BaseTransportPredictor):
    def __init__(self, config: FlowMatchingPredictorConfig):
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        torch.manual_seed(config.seed)

        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self.vector_field = MLPVectorField(
            x_dim=config.x_dim,
            y_dim=config.y_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
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

    def predict_vector_field(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        state = self.to_device(state)
        x = self.to_device(x)
        t = self.to_device(t)

        return self.vector_field(
            state=state,
            x=x,
            t=t,
        )

    @torch.no_grad()
    def pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        ode_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Push latent u to observation space.

            y = T_x(u)

        Args:
            x: (batch, x_dim)
            u: (batch, y_dim)

        Returns:
            y: (batch, y_dim)
        """
        self.eval()

        x = self.to_device(x)
        state = self.to_device(u)

        steps = self.config.ode_steps if ode_steps is None else ode_steps

        dt = torch.full(
            (state.shape[0], 1),
            1.0 / steps,
            device=self.device,
            dtype=self.dtype,
        )

        for k in range(steps):
            t = (k + 0.5) * dt
            velocity = self.predict_vector_field(state=state, x=x, t=t)
            state = state + dt * velocity

        return self.unscale_y(state).detach()

    @torch.no_grad()
    def pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ode_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Pull observation y back to latent space.

            u = T_x^{-1}(y)

        Args:
            x: (batch, x_dim)
            y: (batch, y_dim)

        Returns:
            u: (batch, y_dim)
        """
        self.eval()

        x = self.to_device(x)
        state = self.scale_y(self.to_device(y))

        steps = self.config.ode_steps if ode_steps is None else ode_steps

        dt = torch.full(
            (state.shape[0], 1),
            1.0 / steps,
            device=self.device,
            dtype=self.dtype,
        )

        for k in range(steps):
            t = 1.0 - (k + 0.5) * dt
            velocity = self.predict_vector_field(state=state, x=x, t=t)
            state = state - dt * velocity

        return state.detach()

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default transport score:

            S_multi(x, y) = T_x^{-1}(y)
        """
        return self.pullback(x=x, y=y)

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        x = self.to_device(x)

        batch_size = x.shape[0]

        u = torch.randn(
            batch_size,
            n_samples,
            self.y_dim,
            device=self.device,
            dtype=self.dtype,
        )

        x_rep = (
            x[:, None, :]
            .expand(batch_size, n_samples, self.x_dim)
            .reshape(batch_size * n_samples, self.x_dim)
        )

        u_flat = u.reshape(batch_size * n_samples, self.y_dim)

        y_flat = self.pushforward(x=x_rep, u=u_flat)

        return y_flat.reshape(batch_size, n_samples, self.y_dim)

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
        data = torch.load(path, map_location=map_location, weights_only=False)

        config = FlowMatchingPredictorConfig.model_validate(data["config"])
        model = cls(config)
        model.load_state_dict(data["state_dict"])
        model.to(device=model.device, dtype=model.dtype)

        return model