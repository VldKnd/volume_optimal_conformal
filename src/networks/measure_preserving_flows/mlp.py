from __future__ import annotations

import math
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

TRANSFORMER_POSITIONAL_ENCODING_BASE = 10_000.0
TRANSFORMER_TIME_POSITION_SCALE = 1_024.0


class TransformerTimeEncoding(nn.Module):
    """Sinusoidal encoding of continuous time used by Transformers.

    For channel pair ``2i, 2i + 1``, this computes

    ``sin(1024 t / 10000 ** (2i / d))`` and
    ``cos(1024 t / 10000 ** (2i / d))``.

    Scaling maps the ODE interval ``[0, 1]`` to Transformer-style positions
    ``[0, 1024]`` before encoding.

    The final unpaired channel is a sine channel when ``d`` is odd.
    """

    def __init__(self, embedding_dimension: int):
        super().__init__()

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive, "
                f"got {embedding_dimension}."
            )

        self.embedding_dimension = embedding_dimension
        channel_indexes = torch.arange(0, embedding_dimension, 2)
        angular_frequencies = torch.exp(
            channel_indexes *
            (-math.log(TRANSFORMER_POSITIONAL_ENCODING_BASE) / embedding_dimension)
        )
        self.register_buffer(
            "angular_frequencies",
            angular_frequencies,
            persistent=False,
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.shape[-1] != 1:
            raise ValueError(f"Expected time.shape[-1] = 1, got {time.shape[-1]}.")

        positions = TRANSFORMER_TIME_POSITION_SCALE * time
        angles = positions * self.angular_frequencies
        paired_encoding = torch.stack(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1,
        ).flatten(start_dim=-2)
        return paired_encoding[..., :self.embedding_dimension]


class PReLU(nn.Module):
    """Power ReLU: ``(1 / p) * ReLU(x) ** p``."""

    def __init__(self, p: float = 2.0):
        super().__init__()

        if p <= 0.0:
            raise ValueError(f"p must be positive, got {p}.")

        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).pow(self.p) / self.p


class ScaledTanh(nn.Module):
    """Tanh with a positive trainable scalar output amplitude."""

    def __init__(self, initial_scale: float = 1.0):
        super().__init__()

        if not math.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ValueError(
                f"initial_scale must be finite and positive, got {initial_scale}."
            )

        self.alpha = nn.Parameter(torch.tensor(math.log(initial_scale)))

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(x)


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
        time_encoding_dimension: int,
        activation: ActivationName = "softplus",
        activation_power: float = 2.0,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = y_dim if state_dim is None else state_dim
        self.output_dim = y_dim if output_dim is None else output_dim
        self.time_encoding_dimension = time_encoding_dimension
        self.activation = activation
        self.activation_power = float(activation_power)

        if time_encoding_dimension < 0:
            raise ValueError(
                "time_encoding_dimension must be non-negative, "
                f"got {time_encoding_dimension}."
            )

        self.time_encoding = (
            TransformerTimeEncoding(time_encoding_dimension)
            if time_encoding_dimension > 0 else None
        )

        input_dim = x_dim + self.state_dim + time_encoding_dimension

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]

        for _ in range(num_hidden_layers):
            layers.append(
                make_activation(
                    activation,
                    activation_power=activation_power,
                )
            )
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(make_activation("tanh"))

        output_layer = nn.Linear(hidden_dim, self.output_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

    def _time_feature(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.time_encoding is None:
            return state.new_empty(*state.shape[:-1], 0)

        t = torch.as_tensor(t, device=state.device, dtype=state.dtype)

        if t.ndim == 0:
            t = t.reshape(1).expand(*state.shape[:-1], 1)

        elif t.shape == state.shape[:-1]:
            t = t.unsqueeze(-1)

        try:
            t = torch.broadcast_to(t, (*state.shape[:-1], 1))
        except RuntimeError as error:
            raise ValueError(
                "t must be scalar or broadcastable to "
                f"state.shape[:-1] + (1,), got t.shape={tuple(t.shape)} "
                f"and state.shape={tuple(state.shape)}."
            ) from error

        return self.time_encoding(t)

    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.cat([x, self._time_feature(state, t)], dim=-1)
        features = torch.cat([state, context], dim=-1)
        return self.net(features)
