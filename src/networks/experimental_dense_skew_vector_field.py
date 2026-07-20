from __future__ import annotations

import torch
import torch.nn as nn

from networks.mlp_vector_field import MLPVectorField


class DenseGaussianSkewVectorField(nn.Module):
    """Gaussian-preserving dense skew vector field with coordinate-JVP divergence.

    The matrix field A(u, x, t) is skew-symmetric and represented by its upper
    triangular entries a_ij(u, x, t), i < j. The velocity is

        v(u, x, t) = div_u A(u, x, t) - A(u, x, t) u.

    This implementation avoids constructing the full Jacobian of A. It computes
    the divergence with one JVP per integrated coordinate and accumulates the
    sparse skew-entry contributions directly.
    """

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int = 64,
        number_of_hidden_layers: int = 2,
        context_dimension: int = 0,
        time_dependent: bool = True,
    ):
        super().__init__()

        self.dimension = dimension
        self.context_dimension = context_dimension
        self.hidden_dimension = hidden_dimension
        self.number_of_hidden_layers = number_of_hidden_layers
        self.time_dependent = time_dependent

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

        self.network = MLPVectorField(
            x_dim=context_dimension,
            y_dim=dimension,
            state_dim=dimension,
            output_dim=number_of_skew_entries,
            time_dim=int(time_dependent),
            hidden_dim=hidden_dimension,
            num_hidden_layers=number_of_hidden_layers,
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

    def calculate_skew_entries(
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

    def calculate_skew_symmetric_matrix(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        raw_entries = self.calculate_skew_entries(u=u, x=x, t=t)

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

    def _matrix_vector_product(
        self,
        skew_entries: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        i = self.skew_matrix_indexes_i
        j = self.skew_matrix_indexes_j

        product = u.new_zeros(u.shape)
        product = product.index_add(
            dim=1,
            index=i,
            source=skew_entries * u[:, j],
        )
        product = product.index_add(
            dim=1,
            index=j,
            source=-skew_entries * u[:, i],
        )

        return product

    def _divergence_from_coordinate_jvps(
        self,
        u: torch.Tensor,
        context: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        i = self.skew_matrix_indexes_i
        j = self.skew_matrix_indexes_j
        divergence = u.new_zeros(u.shape)

        def skew_entries(u_argument: torch.Tensor) -> torch.Tensor:
            return self.network(
                state=u_argument,
                x=context,
                t=time,
            )

        for coordinate in range(self.dimension):
            tangent = torch.zeros_like(u)
            tangent[:, coordinate] = 1.0

            _, derivatives = torch.func.jvp(
                skew_entries,
                (u,),
                (tangent,),
            )

            right_is_coordinate = j == coordinate
            if right_is_coordinate.any():
                divergence = divergence.index_add(
                    dim=1,
                    index=i[right_is_coordinate],
                    source=derivatives[:, right_is_coordinate],
                )

            left_is_coordinate = i == coordinate
            if left_is_coordinate.any():
                divergence = divergence.index_add(
                    dim=1,
                    index=j[left_is_coordinate],
                    source=-derivatives[:, left_is_coordinate],
                )

        return divergence

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

        skew_entries = self.network(
            state=u_flat,
            x=context,
            t=time,
        )
        matrix_vector_product = self._matrix_vector_product(
            skew_entries=skew_entries,
            u=u_flat,
        )
        divergence = self._divergence_from_coordinate_jvps(
            u=u_flat,
            context=context,
            time=time,
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
