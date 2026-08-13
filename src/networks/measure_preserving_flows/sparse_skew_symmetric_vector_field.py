from __future__ import annotations

import torch
import torch.nn as nn

from networks.measure_preserving_flows.mlp import ActivationName, MeasurePreservingMLP


class SparseGaussianSkewVectorField(nn.Module):
    """Gaussian-preserving sparse tridiagonal skew vector field.

    The network outputs a potential b(u, x, t) in R^{d - 1}. It defines the
    skew-symmetric matrix A by

        A[k, k + 1] = b[k],    A[k + 1, k] = -b[k],

    for k = 0, ..., d - 2, with all other entries equal to zero. The velocity is

        v(u, x, t) = div_u A(u, x, t) - A(u, x, t) u.

    The divergence is computed from the Jacobian of b with one batched
    vector-Jacobian product call, then the tridiagonal structure is applied by
    slicing.
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
            raise ValueError("SparseGaussianSkewVectorField requires dimension >= 2.")

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

        self.network = MeasurePreservingMLP(
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

    def calculate_potential(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        self._check_integrated_variable(u)
        context = self._prepare_context(u=u, x=x)
        time = self._prepare_time(u=u, t=t)

        return self.network(
            state=u,
            x=context,
            t=time,
        )

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

    def calculate_divergence(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        self._check_integrated_variable(u)

        batch_shape = u.shape[:-1]
        u_flat = u.reshape(-1, self.dimension)
        if not u_flat.requires_grad:
            u_flat = u_flat.detach().clone().requires_grad_(True)

        number_of_points = u_flat.shape[0]
        context = self._prepare_context(u=u, x=x).reshape(
            number_of_points,
            self.context_dimension,
        )
        time = self._prepare_time(u=u, t=t).reshape(
            number_of_points,
            int(self.time_dependent),
        )
        potential = self.network(state=u_flat, x=context, t=time)

        divergence = self._divergence_from_potential(
            potential=potential,
            u=u_flat,
        )

        return divergence.reshape(*batch_shape, self.dimension)

    def _divergence_from_potential(
        self,
        potential: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        jacobian_rows = self._batched_potential_jacobian_rows(
            potential=potential,
            u=u,
        )
        edge_indexes = self.edge_indexes

        left_derivatives = jacobian_rows[
            edge_indexes,
            :,
            edge_indexes,
        ]
        right_derivatives = jacobian_rows[
            edge_indexes,
            :,
            edge_indexes + 1,
        ]

        divergence = u.new_zeros(u.shape)
        divergence[:, :-1] = divergence[:, :-1] + right_derivatives.transpose(0, 1)
        divergence[:, 1:] = divergence[:, 1:] - left_derivatives.transpose(0, 1)

        return divergence

    def _batched_potential_jacobian_rows(
        self,
        potential: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        basis = torch.eye(
            self.number_of_edges,
            device=potential.device,
            dtype=potential.dtype,
        )
        grad_outputs = basis[:, None, :].expand(
            self.number_of_edges,
            potential.shape[0],
            self.number_of_edges,
        )

        return torch.autograd.grad(
            outputs=potential,
            inputs=u,
            grad_outputs=grad_outputs,
            create_graph=True,
            is_grads_batched=True,
        )[0]

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
        if not u_flat.requires_grad:
            u_flat = u_flat.detach().clone().requires_grad_(True)

        number_of_points = u_flat.shape[0]
        context = self._prepare_context(u=u, x=x).reshape(
            number_of_points,
            self.context_dimension,
        )
        time = self._prepare_time(u=u, t=t).reshape(
            number_of_points,
            int(self.time_dependent),
        )

        potential = self.network(state=u_flat, x=context, t=time)
        divergence = self._divergence_from_potential(
            potential=potential,
            u=u_flat,
        )
        matrix_vector_product = self._matrix_vector_product(
            potential=potential,
            u=u_flat,
        )

        velocity = divergence - matrix_vector_product
        return velocity.reshape(*batch_shape, self.dimension)

    def forward(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        with torch.enable_grad():
            return self.calculate_exact_velocity_field(u=u, x=x, t=t)
