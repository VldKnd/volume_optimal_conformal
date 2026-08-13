from __future__ import annotations

import torch
import torch.nn as nn

from networks.measure_preserving_flows.mlp import ActivationName, MeasurePreservingMLP


class DenseGaussianSkewVectorField(nn.Module):
    """Gaussian-preserving vector field v = div_u(A) - A u.

    The matrix field A(u, x, t) is skew-symmetric. The divergence is taken
    only with respect to the integrated variable u; x is optional context.
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

        self.dimension = dimension
        self.context_dimension = context_dimension
        self.hidden_dimension = hidden_dimension
        self.number_of_hidden_layers = number_of_hidden_layers
        self.time_dependent = time_dependent
        self.activation = activation
        self.activation_power = float(activation_power)

        skew_matrix_indexes_i, skew_matrix_indexes_j = torch.triu_indices(
            dimension,
            dimension,
            offset=1,
        )
        number_of_skew_entries = dimension * (dimension - 1) // 2
        skew_matrix_flat_indexes = (
            skew_matrix_indexes_i * dimension + skew_matrix_indexes_j
        )

        self.register_buffer("skew_matrix_indexes_i", skew_matrix_indexes_i)
        self.register_buffer("skew_matrix_indexes_j", skew_matrix_indexes_j)
        self.register_buffer(
            "skew_matrix_flat_indexes",
            skew_matrix_flat_indexes,
        )

        self.network = MeasurePreservingMLP(
            x_dim=context_dimension,
            y_dim=dimension,
            state_dim=dimension,
            output_dim=number_of_skew_entries,
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

        return torch.broadcast_to(
            x,
            (*u.shape[:-1], self.context_dimension),
        )

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

    def calculate_skew_symmetric_matrix(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        self._check_integrated_variable(u)
        context = self._prepare_context(u=u, x=x)
        time = self._prepare_time(u=u, t=t)

        raw_entries = self.network(
            state=u,
            x=context,
            t=time,
        )

        upper_flat = raw_entries.new_zeros(
            (*raw_entries.shape[:-1], self.dimension * self.dimension)
        ).scatter(
            dim=-1,
            index=self.skew_matrix_flat_indexes.expand(
                *raw_entries.shape[:-1],
                -1,
            ),
            src=raw_entries,
        )
        upper_matrix = upper_flat.reshape(
            *raw_entries.shape[:-1],
            self.dimension,
            self.dimension,
        )

        return upper_matrix - upper_matrix.transpose(-2, -1)

    def calculate_single_input_skew_symmetric_matrix(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skew_matrix = self.calculate_skew_symmetric_matrix(
            u=u,
            x=x,
            t=t,
        )
        return skew_matrix, skew_matrix

    def calculate_single_input_exact_velocity_field(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor | float | None,
    ) -> torch.Tensor:
        jacobian_matrix, skew_matrix = torch.func.jacfwd(
            func=self.calculate_single_input_skew_symmetric_matrix,
            has_aux=True,
            argnums=0,
        )(u, x, t)

        divergence = jacobian_matrix.diagonal(dim1=1, dim2=2).sum(-1)
        return divergence - skew_matrix @ u

    def calculate_exact_velocity_field(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        self._check_integrated_variable(u)

        batch_shape = u.shape[:-1]
        u_flat = u.reshape(-1, self.dimension)
        number_of_points = u_flat.shape[0]
        context = self._prepare_context(u=u, x=x).reshape(
            number_of_points,
            self.context_dimension,
        )
        time = self._prepare_time(u=u, t=t).reshape(
            number_of_points,
            int(self.time_dependent),
        )

        velocity = torch.func.vmap(self.calculate_single_input_exact_velocity_field,
                                   )(u_flat, context, time)

        return velocity.reshape(*batch_shape, self.dimension)

    def forward(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        with torch.enable_grad():
            return self.calculate_exact_velocity_field(u=u, x=x, t=t)
