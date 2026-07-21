from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

ActivationName = Literal[
    "elu",
    "gelu",
    "leaky_relu",
    "prelu",
    "relu",
    "silu",
    "softplus",
    "tanh",
]


class PReLU(nn.Module):
    """Power ReLU: ``(1 / p) * ReLU(x) ** p``."""

    def __init__(self, p: float = 2.0):
        super().__init__()

        if p <= 0.0:
            raise ValueError(f"p must be positive, got {p}.")

        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).pow(self.p) / self.p


def make_activation(
    activation: ActivationName,
    activation_power: float = 2.0,
) -> nn.Module:
    if activation == "elu":
        return nn.ELU()

    if activation == "gelu":
        return nn.GELU()

    if activation == "leaky_relu":
        return nn.LeakyReLU()

    if activation == "prelu":
        return PReLU(p=activation_power)

    if activation == "relu":
        return nn.ReLU()

    if activation == "silu":
        return nn.SiLU()

    if activation == "softplus":
        return nn.Softplus()

    if activation == "tanh":
        return nn.Tanh()

    raise ValueError(
        f"Unknown activation={activation!r}. "
        "Expected one of 'elu', 'gelu', 'leaky_relu', 'prelu', 'relu', "
        "'silu', 'softplus', or 'tanh'."
    )


class MeasurePreservingMLP(nn.Module):
    """MLP used inside Gaussian measure-preserving skew vector fields."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        state_dim: int | None = None,
        output_dim: int | None = None,
        time_dim: int = 1,
        activation: ActivationName = "softplus",
        activation_power: float = 2.0,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = y_dim if state_dim is None else state_dim
        self.output_dim = y_dim if output_dim is None else output_dim
        self.time_dim = time_dim
        self.activation = activation
        self.activation_power = float(activation_power)

        input_dim = x_dim + self.state_dim + time_dim

        layers: list[nn.Module] = [
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            make_activation(activation, activation_power=activation_power),
        ]

        for _ in range(num_hidden_layers):
            layers.extend([
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                make_activation(activation, activation_power=activation_power),
            ])

        output_layer = spectral_norm(nn.Linear(hidden_dim, self.output_dim))
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        layers.append(make_activation("tanh"))
        self.net = nn.Sequential(*layers)

    def _time_feature(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.time_dim == 0:
            return state.new_empty(*state.shape[:-1], 0)

        t = torch.as_tensor(t, device=state.device, dtype=state.dtype)

        if t.ndim == 0:
            return t.reshape(1).expand(*state.shape[:-1], self.time_dim)

        if t.shape == state.shape[:-1]:
            t = t.unsqueeze(-1)

        return t.expand(*state.shape[:-1], self.time_dim)

    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([state, x, self._time_feature(state, t)], dim=-1)
        return self.net(features)
