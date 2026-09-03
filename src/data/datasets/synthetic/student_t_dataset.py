"""Unconditional multivariate Student-t synthetic dataset."""

import math
from numbers import Real
from typing import NamedTuple

import torch
from scipy.stats import f as f_distribution

from configs.datasets.synthetic.student_t_dataset import StudentTDatasetConfig
from data.datasets.base import DatasetSplits, XYData
from data.datasets.synthetic.base import BaseSyntheticDataset


class StudentTHDR(NamedTuple):
    """Analytic geometry of a centered multivariate Student-t HDR."""

    radius: float
    semi_axis_lengths: torch.Tensor
    volume: float

    @property
    def log_volume(self) -> float:
        """Return the HDR volume in the numerically stable reporting domain."""
        return math.log(self.volume)


class StudentTDataset(BaseSyntheticDataset):
    """Centered elliptical Student-t distribution with a dummy condition.

    Every condition is the one-dimensional zero vector. For target dimension
    ``d``, the Student-t scale matrix is diagonal with entries

    ``(k ** (1 - 1/d), k ** (-1/d), ..., k ** (-1/d))``.

    The exponents sum to zero, so the scale matrix has determinant one for
    every positive ``k`` and every target dimension ``d``.

    Samples use the canonical multivariate Student-t construction

    ``Y = scale_matrix ** (1/2) @ Z / sqrt(G / nu)``,

    where ``Z ~ N(0, I_d)`` and the scalar ``G ~ ChiSquare(nu)`` is shared by
    all coordinates of an observation.
    """

    def __init__(self, config: StudentTDatasetConfig):
        self.config = config
        self._splits: DatasetSplits | None = None

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(config.seed)

        repeated_entry = config.k**(-1.0 / config.y_dim)
        first_entry = config.k**(1.0 - 1.0 / config.y_dim)
        scale_diagonal = torch.full(
            (config.y_dim, ),
            repeated_entry,
            dtype=self.dtype,
        )
        scale_diagonal[0] = first_entry
        self._scale_diagonal = scale_diagonal.to(self.device)
        self._coordinate_scales = torch.sqrt(self._scale_diagonal)

    @property
    def x_dim(self) -> int:
        return self.config.x_dim

    @property
    def y_dim(self) -> int:
        return self.config.y_dim

    @property
    def n_total(self) -> int:
        return self.config.n_train + self.config.n_calibration + self.config.n_test

    @property
    def scale_matrix(self) -> torch.Tensor:
        """Return the diagonal scale matrix parameterizing the Student-t law."""
        return torch.diag(self._scale_diagonal)

    @property
    def correlation_matrix(self) -> torch.Tensor:
        """Return the requested diagonal matrix.

        This name mirrors the dataset specification. Mathematically, a matrix
        with non-unit diagonal is a scale matrix rather than a correlation
        matrix.
        """
        return self.scale_matrix

    @property
    def covariance(self) -> torch.Tensor:
        """Return the finite covariance, which exists only when ``nu > 2``."""
        if self.config.nu <= 2.0:
            raise RuntimeError("The Student-t covariance is undefined when nu <= 2.")
        return self.config.nu / (self.config.nu - 2.0) * self.scale_matrix

    @property
    def supports_density(self) -> bool:
        return True

    def hdr(self, alpha: float) -> StudentTHDR:
        """Return the analytic ``(1 - alpha)`` highest-density region.

        The region is the axis-aligned ellipsoid

        ``sum_i y_i**2 / sigma_i <= radius**2``,

        where ``sigma_i`` are the diagonal entries of ``scale_matrix`` and
        ``radius**2 = d * F^{-1}_{d, nu}(1 - alpha)``.
        """
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a real number.")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")

        dimension = self.y_dim
        radius_squared = dimension * float(
            f_distribution.ppf(
                1.0 - alpha,
                dimension,
                self.config.nu,
            )
        )
        if not math.isfinite(radius_squared) or radius_squared <= 0.0:
            raise RuntimeError("The Student-t HDR F quantile is invalid.")

        radius = math.sqrt(radius_squared)
        semi_axis_lengths = radius * self._coordinate_scales
        log_unit_ball_volume = (
            0.5 * dimension * math.log(math.pi) - math.lgamma(0.5 * dimension + 1.0)
        )
        log_scale_volume = 0.5 * float(
            torch.log(self._scale_diagonal).sum().detach().cpu()
        )
        log_volume = (
            log_unit_ball_volume + log_scale_volume + dimension * math.log(radius)
        )

        return StudentTHDR(
            radius=radius,
            semi_axis_lengths=semi_axis_lengths,
            volume=math.exp(log_volume),
        )

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """Return the zero center; the distribution is independent of ``x``."""
        x = x.to(device=self.device, dtype=self.dtype)
        self._validate_x(x)
        return torch.zeros(
            x.shape[0],
            self.y_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def sample_x(self, n: int) -> torch.Tensor:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer.")
        return torch.zeros(
            n,
            self.x_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def sample_target(self, n: int) -> torch.Tensor:
        x = self.sample_x(n)
        return self.sample_conditional(x=x, n_samples=1).squeeze(1)

    def sample_conditional(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from the same multivariate Student-t law for every ``x``."""
        if (
            isinstance(n_samples, bool) or not isinstance(n_samples, int)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer.")

        x = x.to(device=self.device, dtype=self.dtype)
        self._validate_x(x)
        batch_size = x.shape[0]

        gaussian = torch.randn(
            batch_size,
            n_samples,
            self.y_dim,
            generator=self._generator,
            dtype=self.dtype,
        )
        gamma_concentration = torch.full(
            (batch_size, n_samples, 1),
            self.config.nu / 2.0,
            dtype=self.dtype,
        )
        chi_square = 2.0 * torch._standard_gamma(
            gamma_concentration,
            generator=self._generator,
        )
        radial_scale = torch.rsqrt(chi_square / self.config.nu)
        samples = gaussian * radial_scale * self._coordinate_scales.cpu()
        return samples.to(self.device)

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact multivariate Student-t log-density."""
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
        self._validate_xy(x=x, y=y)

        dimension = self.y_dim
        nu = self.config.nu
        quadratic = (y.square() / self._scale_diagonal).sum(dim=-1)
        log_scale_determinant = torch.log(self._scale_diagonal).sum()
        log_normalizer = (
            math.lgamma((nu + dimension) / 2.0) - math.lgamma(nu / 2.0) -
            0.5 * dimension * math.log(nu * math.pi) - 0.5 * log_scale_determinant
        )
        return log_normalizer - 0.5 * (nu + dimension) * torch.log1p(quadratic / nu)

    def prepare(self) -> None:
        x, y = self.sample_joint(self.n_total)
        n_train = self.config.n_train
        n_calibration = self.config.n_calibration

        self._splits = DatasetSplits(
            train=XYData(x=x[:n_train], y=y[:n_train]),
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

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.x_dim:
            raise ValueError(
                f"Expected x with shape (batch, {self.x_dim}), "
                f"got {tuple(x.shape)}."
            )

    def _validate_xy(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._validate_x(x)
        if y.ndim != 2 or y.shape[1] != self.y_dim:
            raise ValueError(
                f"Expected y with shape (batch, {self.y_dim}), "
                f"got {tuple(y.shape)}."
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "x and y must have matching batch sizes, got "
                f"{x.shape[0]} and {y.shape[0]}."
            )
