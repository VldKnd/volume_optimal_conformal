"""Unconditional three-petal transformation of a Gaussian."""

import math

import torch

from configs.datasets.synthetic.star_shaped_gaussian import (
    StarShapedGaussianDatasetConfig,
)
from data.datasets.base import DatasetSplits, XYData
from data.datasets.synthetic.base import BaseSyntheticDataset


class StarShapedGaussianDataset(BaseSyntheticDataset):
    """Push a standard 2D Gaussian through an area-preserving star map.

    For ``U ~ N(0, I_2)``, write its polar coordinates as ``(r, theta)``.
    Let ``phi`` be the target angle, ``a`` denote ``petal_amplitude``, and

    ``c = sqrt(1 + a^2 / 2)``,

    ``R(phi) = (1 + a cos(3 phi)) / c``.

    The target angle is the unique solution of

    ``theta = phi + 2a sin(3phi)/(3c^2) + a^2 sin(6phi)/(12c^2)``,

    and ``r' = r R(phi)``. The angular relation has derivative ``R(phi)^2``,
    so the radial and angular Cartesian Jacobian factors cancel exactly:

    ``(r' / r) * (d r' / d r) * (d phi / d theta) = 1``.

    The image of every centered circle has the classic polar-star boundary
    ``R(phi)``. Its three valleys are strictly concave for ``a > 0.1``. A
    one-dimensional zero condition is retained only for compatibility with
    the conditional pipeline.
    """

    number_of_petals = 3

    def __init__(self, config: StarShapedGaussianDatasetConfig):
        self.config = config
        self._splits: DatasetSplits | None = None

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(config.seed)

    @property
    def x_dim(self) -> int:
        return self.config.x_dim

    @property
    def y_dim(self) -> int:
        return self.config.y_dim

    @property
    def n_total(self) -> int:
        return (self.config.n_train + self.config.n_calibration + self.config.n_test)

    @property
    def supports_density(self) -> bool:
        return True

    def sample_x(self, n: int) -> torch.Tensor:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer.")

        return torch.zeros(
            n,
            self.x_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def sample_source(self, n: int) -> torch.Tensor:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer.")

        u = torch.randn(
            n,
            self.y_dim,
            generator=self._generator,
            dtype=self.dtype,
        )
        return u.to(self.device)

    def sample_target(self, n: int) -> torch.Tensor:
        x = self.sample_x(n)
        return self.sample_conditional(x=x, n_samples=1).squeeze(1)

    def sample_conditional(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        if (
            isinstance(n_samples, bool) or not isinstance(n_samples, int)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer.")

        x = self._fixed_condition(x, require_batch_matrix=True)

        u = torch.randn(
            x.shape[0],
            n_samples,
            self.y_dim,
            generator=self._generator,
            dtype=self.dtype,
        ).to(self.device)
        expanded_x = x[:, None, :].expand(
            x.shape[0],
            n_samples,
            self.x_dim,
        )
        return self.push_u_given_x(u=u, x=expanded_x)

    def push_u_given_x(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the area-preserving three-petal deformation."""
        u = u.to(device=self.device, dtype=self.dtype)
        x = self._fixed_condition(x)
        self._validate_matching_shapes(point=u, x=x, point_name="u")

        radius = torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        source_angle = torch.atan2(u[..., 1:2], u[..., 0:1])
        target_angle = self._target_angle(source_angle)
        target_radius = radius * self._radial_factor(target_angle)

        return self._from_polar(radius=target_radius, angle=target_angle)

    def push_y_given_x(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Invert the area-preserving three-petal deformation."""
        y = y.to(device=self.device, dtype=self.dtype)
        x = self._fixed_condition(x)
        self._validate_matching_shapes(point=y, x=x, point_name="y")

        target_radius = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        target_angle = torch.atan2(y[..., 1:2], y[..., 0:1])
        source_angle = self._source_angle(target_angle)
        source_radius = target_radius / self._radial_factor(target_angle)

        return self._from_polar(radius=source_radius, angle=source_angle)

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Return the identically zero ``log |det D_u T(u)|``."""
        x = self._fixed_condition(x)
        u = u.to(device=self.device, dtype=self.dtype)
        self._validate_matching_shapes(point=u, x=x, point_name="u")
        return torch.zeros(u.shape[:-1], device=u.device, dtype=u.dtype)

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the exact log-density by change of variables."""
        u = self.push_y_given_x(y=y, x=x)
        log_base = -0.5 * (
            u.square().sum(dim=-1) + self.y_dim * math.log(2.0 * math.pi)
        )
        return log_base - self.log_det(x=x, u=u)

    def prepare(self) -> None:
        x, y = self.sample_joint(self.n_total)
        n_train = self.config.n_train
        n_calibration = self.config.n_calibration

        self._splits = DatasetSplits(
            train=XYData(
                x=x[:n_train],
                y=y[:n_train],
            ),
            calibration=XYData(
                x=x[n_train:n_train + n_calibration],
                y=y[n_train:n_train + n_calibration],
            ),
            test=XYData(
                x=x[n_train + n_calibration:],
                y=y[n_train + n_calibration:],
            ),
        )

    def get_splits(self) -> DatasetSplits:
        if self._splits is None:
            self.prepare()

        assert self._splits is not None
        return self._splits

    def _radial_factor(self, target_angle: torch.Tensor) -> torch.Tensor:
        amplitude = self.config.petal_amplitude
        normalization = math.sqrt(1.0 + 0.5 * amplitude**2)
        return (
            1.0 + amplitude * torch.cos(self.number_of_petals * target_angle)
        ) / normalization

    def _source_angle(self, target_angle: torch.Tensor) -> torch.Tensor:
        """Map the target polar angle to its area coordinate."""
        amplitude = self.config.petal_amplitude
        petals = self.number_of_petals
        normalization_squared = 1.0 + 0.5 * amplitude**2
        return (
            target_angle + 2.0 * amplitude /
            (petals * normalization_squared) * torch.sin(petals * target_angle) +
            amplitude**2 / (4.0 * petals * normalization_squared) *
            torch.sin(2.0 * petals * target_angle)
        )

    def _target_angle(self, source_angle: torch.Tensor) -> torch.Tensor:
        """Invert the monotone area coordinate with differentiable refinement."""
        lower = torch.full_like(source_angle, -math.pi)
        upper = torch.full_like(source_angle, math.pi)

        # Float64 reaches machine precision; float32 stabilizes much earlier.
        for _ in range(64):
            midpoint = 0.5 * (lower + upper)
            move_lower = self._source_angle(midpoint) < source_angle
            lower = torch.where(move_lower, midpoint, lower)
            upper = torch.where(move_lower, upper, midpoint)

        # Bisection supplies a globally robust value. Newton refinement restores
        # the implicit derivative d phi / d theta = 1 / R(phi)^2, which is
        # needed when differentiating the synthetic transport.
        target_angle = (0.5 * (lower + upper)).detach()
        for _ in range(2):
            target_angle = target_angle - (
                self._source_angle(target_angle) - source_angle
            ) / self._radial_factor(target_angle).square()

        return target_angle

    @staticmethod
    def _from_polar(radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        return radius * torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)

    def _validate_matching_shapes(
        self,
        point: torch.Tensor,
        x: torch.Tensor,
        point_name: str,
    ) -> None:
        if point.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Expected {point_name}.shape[:-1] == x.shape[:-1], got "
                f"{point.shape[:-1]} and {x.shape[:-1]}."
            )
        if point.shape[-1] != self.y_dim:
            raise ValueError(
                f"Expected {point_name}.shape[-1] = {self.y_dim}, "
                f"got {point.shape[-1]}."
            )
        if x.shape[-1] != self.x_dim:
            raise ValueError(f"Expected x.shape[-1] = {self.x_dim}, got {x.shape[-1]}.")

    def _fixed_condition(
        self,
        x: torch.Tensor,
        require_batch_matrix: bool = False,
    ) -> torch.Tensor:
        """Validate the dummy condition shape and replace its values by zero."""
        x = x.to(device=self.device, dtype=self.dtype)
        if x.ndim < 1 or x.shape[-1] != self.x_dim:
            raise ValueError(
                f"Expected x with trailing dimension {self.x_dim}, "
                f"got shape {tuple(x.shape)}."
            )
        if require_batch_matrix and x.ndim != 2:
            raise ValueError(
                f"Expected x with shape (batch, {self.x_dim}), "
                f"got {tuple(x.shape)}."
            )
        return torch.zeros_like(x)
