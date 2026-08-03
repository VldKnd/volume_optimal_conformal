from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.measure_preserving_flows.mlp import (
    ActivationName,
    MeasurePreservingMLP,
    PReLU,
)


def _activation_derivative(
    activation: nn.Module,
    value: torch.Tensor,
    activated_value: torch.Tensor,
) -> torch.Tensor:
    """Evaluate an activation's elementwise derivative explicitly."""
    if isinstance(activation, nn.ELU):
        away_from_zero = torch.where(
            value > 0.0,
            torch.ones_like(value),
            activation.alpha * torch.exp(value),
        )
        return torch.where(
            value == 0.0,
            torch.full_like(value, activation.alpha),
            away_from_zero,
        )

    if isinstance(activation, nn.GELU):
        if activation.approximate == "none":
            cdf = 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))
            density = torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)
            return cdf + value * density

        if activation.approximate == "tanh":
            coefficient = math.sqrt(2.0 / math.pi)
            inner = coefficient * (value + 0.044715 * value.pow(3))
            tanh_inner = torch.tanh(inner)
            inner_derivative = coefficient * (1.0 + 3.0 * 0.044715 * value.square())
            return (
                0.5 * (1.0 + tanh_inner) + 0.5 * value *
                (1.0 - tanh_inner.square()) * inner_derivative
            )

        raise ValueError(f"Unsupported GELU approximation {activation.approximate!r}.")

    if isinstance(activation, nn.LeakyReLU):
        return torch.where(
            value > 0.0,
            torch.ones_like(value),
            torch.full_like(value, activation.negative_slope),
        )

    if isinstance(activation, PReLU):
        # Selecting one before the power keeps the non-positive branch finite
        # when 0 < p < 1. The derivative there follows ReLU's zero convention.
        positive_value = torch.where(value > 0.0, value, torch.ones_like(value))
        return torch.where(
            value > 0.0,
            positive_value.pow(activation.p - 1.0),
            torch.zeros_like(value),
        )

    if isinstance(activation, nn.ReLU):
        return (value > 0.0).to(dtype=value.dtype)

    if isinstance(activation, nn.SiLU):
        sigmoid = torch.sigmoid(value)
        return sigmoid + value * sigmoid * (1.0 - sigmoid)

    if isinstance(activation, nn.Softplus):
        scaled_value = activation.beta * value
        away_from_threshold = torch.where(
            scaled_value > activation.threshold,
            torch.ones_like(value),
            torch.sigmoid(scaled_value),
        )
        threshold_derivative = 1.0 / (1.0 + math.exp(-activation.threshold))
        return away_from_threshold.masked_fill(
            scaled_value == activation.threshold,
            threshold_derivative,
        )

    if isinstance(activation, nn.Tanh):
        return 1.0 - activated_value.square()

    raise TypeError(
        "Explicit state derivatives are not implemented for "
        f"{type(activation).__name__}."
    )


class ExplicitDerivativeMLP(MeasurePreservingMLP):
    """Measure-preserving MLP with explicit sparse-edge derivatives.

    Hidden-layer Jacobians with respect to ``state`` are propagated in full.
    At the output layer, only ``d output[k] / d state[k]`` and
    ``d output[k] / d state[k + 1]`` are contracted. In particular, this class
    never constructs the full tensor of shape
    ``(..., output_dim, state_dim)``.
    """

    def forward_with_edge_derivatives(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return output and its left/right state derivative per sparse edge."""
        if self.output_dim != self.state_dim - 1:
            raise ValueError(
                "Sparse edge derivatives require output_dim = state_dim - 1; "
                f"got output_dim={self.output_dim}, state_dim={self.state_dim}."
            )

        value = torch.cat(
            [state, x, self._time_feature(state=state, t=t)],
            dim=-1,
        )
        hidden_state_jacobian: torch.Tensor | None = None
        output_layer = self.net[-2]
        output_activation = self.net[-1]

        if not isinstance(output_layer, nn.Linear):
            raise TypeError(
                "The penultimate MeasurePreservingMLP layer must be linear."
            )

        for layer_index, layer in enumerate(self.net):
            if layer_index == len(self.net) - 2:
                break

            if isinstance(layer, nn.Linear):
                next_value = layer(value)

                if hidden_state_jacobian is None:
                    state_weight = layer.weight[:, :self.state_dim]
                    hidden_state_jacobian = state_weight.expand(
                        *value.shape[:-1],
                        layer.out_features,
                        self.state_dim,
                    )
                else:
                    hidden_state_jacobian = F.linear(
                        hidden_state_jacobian.transpose(-2, -1),
                        layer.weight,
                    ).transpose(-2, -1)

                value = next_value
                continue

            activated_value = layer(value)
            if hidden_state_jacobian is None:
                raise RuntimeError(
                    "An activation appeared before the first linear layer."
                )

            hidden_state_jacobian = hidden_state_jacobian * _activation_derivative(
                activation=layer,
                value=value,
                activated_value=activated_value,
            ).unsqueeze(-1)
            value = activated_value

        if hidden_state_jacobian is None:
            raise RuntimeError("The MLP does not contain a hidden linear layer.")

        output_pre_activation = output_layer(value)
        output = output_activation(output_pre_activation)
        output_activation_derivative = _activation_derivative(
            activation=output_activation,
            value=output_pre_activation,
            activated_value=output,
        )
        output_weight = output_layer.weight.transpose(0, 1)
        left_derivatives = torch.einsum(
            "...he,he->...e",
            hidden_state_jacobian[..., :-1],
            output_weight,
        ) * output_activation_derivative
        right_derivatives = torch.einsum(
            "...he,he->...e",
            hidden_state_jacobian[..., 1:],
            output_weight,
        ) * output_activation_derivative
        return output, left_derivatives, right_derivatives


class ExplicitSparseGaussianSkewVectorField(nn.Module):
    """Sparse Gaussian-preserving field with an explicit MLP Jacobian.

    For edge potential ``b_k``, only ``d b_k / d u_k`` and
    ``d b_k / d u_(k+1)`` are needed. The final MLP Jacobian therefore contains
    exactly ``2 * (dimension - 1)`` selected entries per point rather than the
    full ``(dimension - 1) * dimension`` entries.
    """

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int = 64,
        number_of_hidden_layers: int = 2,
        context_dimension: int = 0,
        time_dependent: bool = True,
        activation: ActivationName = "softplus",
        activation_power: float = 2.0,
    ):
        super().__init__()

        if dimension < 2:
            raise ValueError(
                "ExplicitSparseGaussianSkewVectorField requires dimension >= 2."
            )

        self.dimension = dimension
        self.context_dimension = context_dimension
        self.hidden_dimension = hidden_dimension
        self.number_of_hidden_layers = number_of_hidden_layers
        self.time_dependent = time_dependent
        self.activation = activation
        self.activation_power = float(activation_power)
        self.number_of_edges = dimension - 1

        edge_indexes = torch.arange(self.number_of_edges)
        self.register_buffer("edge_indexes", edge_indexes)

        self.network = ExplicitDerivativeMLP(
            x_dim=context_dimension,
            y_dim=dimension,
            state_dim=dimension,
            output_dim=self.number_of_edges,
            time_dim=int(time_dependent),
            hidden_dim=hidden_dimension,
            num_hidden_layers=number_of_hidden_layers,
            activation=activation,
            activation_power=activation_power,
        )

    def _check_integrated_variable(self, u: torch.Tensor) -> None:
        if u.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected u.shape[-1] = {self.dimension}, got {u.shape[-1]}."
            )

    def _prepare_context(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.context_dimension == 0:
            if x is not None and x.shape[-1] != 0:
                raise ValueError(
                    "This vector field was created with context_dimension=0, "
                    "but a non-empty context was provided."
                )

            return u.new_empty(*u.shape[:-1], 0)

        if x is None:
            raise ValueError("x context must be provided when context_dimension > 0.")

        x = x.to(device=u.device, dtype=u.dtype)

        if x.shape[-1] != self.context_dimension:
            raise ValueError(
                f"Expected x.shape[-1] = {self.context_dimension}, "
                f"got {x.shape[-1]}."
            )

        return torch.broadcast_to(x, (*u.shape[:-1], self.context_dimension))

    def _prepare_time(
        self,
        u: torch.Tensor,
        t: torch.Tensor | float | None,
    ) -> torch.Tensor:
        if not self.time_dependent:
            return u.new_empty(*u.shape[:-1], 0)

        if t is None:
            return torch.zeros_like(u[..., :1])

        t_tensor = torch.as_tensor(t, dtype=u.dtype, device=u.device)

        if t_tensor.ndim == 0:
            return torch.zeros_like(u[..., :1]) + t_tensor

        if t_tensor.shape == u.shape[:-1]:
            t_tensor = t_tensor.unsqueeze(-1)

        return torch.broadcast_to(t_tensor, (*u.shape[:-1], 1))

    def calculate_potential(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        self._check_integrated_variable(u)
        context = self._prepare_context(u=u, x=x)
        time = self._prepare_time(u=u, t=t)
        return self.network(state=u, x=context, t=time)

    def calculate_skew_entries(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        return self.calculate_potential(u=u, x=x, t=t)

    def calculate_skew_symmetric_matrix(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        potential = self.calculate_potential(u=u, x=x, t=t)
        matrix = potential.new_zeros(
            *potential.shape[:-1],
            self.dimension,
            self.dimension,
        )
        edge_indexes = self.edge_indexes
        matrix[..., edge_indexes, edge_indexes + 1] = potential
        matrix[..., edge_indexes + 1, edge_indexes] = -potential
        return matrix

    def _potential_and_edge_derivatives(
        self,
        u: torch.Tensor,
        context: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        potential, left_derivatives, right_derivatives = (
            self.network.forward_with_edge_derivatives(
                state=u,
                x=context,
                t=time,
            )
        )
        return potential, left_derivatives, right_derivatives

    def _divergence_from_edge_derivatives(
        self,
        left_derivatives: torch.Tensor,
        right_derivatives: torch.Tensor,
    ) -> torch.Tensor:
        middle = right_derivatives[..., 1:] - left_derivatives[..., :-1]
        return torch.cat(
            [
                right_derivatives[..., :1],
                middle,
                -left_derivatives[..., -1:],
            ],
            dim=-1,
        )

    def calculate_divergence(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        self._check_integrated_variable(u)
        batch_shape = u.shape[:-1]
        u_flat = u.reshape(-1, self.dimension)
        point_count = u_flat.shape[0]
        context = self._prepare_context(u=u, x=x).reshape(
            point_count,
            self.context_dimension,
        )
        time = self._prepare_time(u=u, t=t).reshape(
            point_count,
            int(self.time_dependent),
        )
        _, left_derivatives, right_derivatives = (
            self._potential_and_edge_derivatives(
                u=u_flat,
                context=context,
                time=time,
            )
        )
        divergence = self._divergence_from_edge_derivatives(
            left_derivatives=left_derivatives,
            right_derivatives=right_derivatives,
        )
        return divergence.reshape(*batch_shape, self.dimension)

    def _matrix_vector_product(
        self,
        potential: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        product = u.new_zeros(u.shape)
        product[:, :-1] = product[:, :-1] + potential * u[:, 1:]
        product[:, 1:] = product[:, 1:] - potential * u[:, :-1]
        return product

    def calculate_exact_velocity_field(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        self._check_integrated_variable(u)
        batch_shape = u.shape[:-1]
        u_flat = u.reshape(-1, self.dimension)
        point_count = u_flat.shape[0]
        context = self._prepare_context(u=u, x=x).reshape(
            point_count,
            self.context_dimension,
        )
        time = self._prepare_time(u=u, t=t).reshape(
            point_count,
            int(self.time_dependent),
        )
        potential, left_derivatives, right_derivatives = (
            self._potential_and_edge_derivatives(
                u=u_flat,
                context=context,
                time=time,
            )
        )
        divergence = self._divergence_from_edge_derivatives(
            left_derivatives=left_derivatives,
            right_derivatives=right_derivatives,
        )
        velocity = divergence - self._matrix_vector_product(
            potential=potential,
            u=u_flat,
        )
        return velocity.reshape(*batch_shape, self.dimension)

    def forward(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        return self.calculate_exact_velocity_field(u=u, x=x, t=t)
