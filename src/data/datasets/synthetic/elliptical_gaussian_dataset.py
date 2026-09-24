"""Unconditional Gaussian target with determinant-one covariance."""

import math
from numbers import Real
from typing import NamedTuple

import torch
from scipy.stats import chi2

from configs.datasets.synthetic.elliptical_gaussian_dataset import (
    EllipticalGaussianDatasetConfig,
)
from data.datasets.base import DatasetSplits, XYData
from data.datasets.synthetic.base import BaseSyntheticDataset


class GaussianHDR(NamedTuple):
    """Analytic geometry of a centered Gaussian highest-density region."""

    radius: float
    semi_axis_lengths: torch.Tensor
    volume: float

    @property
    def log_volume(self) -> float:
        return math.log(self.volume)


class EllipticalGaussianDataset(BaseSyntheticDataset):
    """Centered Gaussian target independent of its dummy condition.

    For target dimension ``d``, the covariance is

    ``diag(k ** (1 - 1/d), k ** (-1/d), ..., k ** (-1/d))``.

    Its determinant is one and its spectral condition number is ``k``.
    """

    def __init__(self, config: EllipticalGaussianDatasetConfig):
        self.config = config
        self._splits: DatasetSplits | None = None

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(config.seed)

        repeated_entry = config.k**(-1.0 / config.y_dim)
        first_entry = config.k**(1.0 - 1.0 / config.y_dim)
        covariance_diagonal = torch.full(
            (config.y_dim, ),
            repeated_entry,
            dtype=self.dtype,
        )
        covariance_diagonal[0] = first_entry
        self._covariance_diagonal = covariance_diagonal.to(self.device)
        self._coordinate_scales = torch.sqrt(self._covariance_diagonal)

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
    def covariance(self) -> torch.Tensor:
        return torch.diag(self._covariance_diagonal)

    @property
    def supports_density(self) -> bool:
        return True

    def hdr(self, alpha: float) -> GaussianHDR:
        """Return the analytic ``(1 - alpha)`` Gaussian HDR ellipsoid."""
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a real number.")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")

        radius_squared = float(chi2.ppf(1.0 - alpha, df=self.y_dim))
        if not math.isfinite(radius_squared) or radius_squared <= 0.0:
            raise RuntimeError("The Gaussian HDR Chi-square quantile is invalid.")
        radius = math.sqrt(radius_squared)
        semi_axis_lengths = radius * self._coordinate_scales
        log_volume = (
            0.5 * self.y_dim * math.log(math.pi) - math.lgamma(0.5 * self.y_dim + 1.0) +
            self.y_dim * math.log(radius) +
            0.5 * float(torch.log(self._covariance_diagonal).sum().detach().cpu())
        )
        return GaussianHDR(
            radius=radius,
            semi_axis_lengths=semi_axis_lengths,
            volume=math.exp(log_volume),
        )

    def mean(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.sample_conditional(self.sample_x(n), n_samples=1).squeeze(1)

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

        x = x.to(device=self.device, dtype=self.dtype)
        self._validate_x(x)
        samples = torch.randn(
            x.shape[0],
            n_samples,
            self.y_dim,
            generator=self._generator,
            dtype=self.dtype,
        )
        samples = samples * self._coordinate_scales.cpu()
        return samples.to(self.device)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
        self._validate_xy(x=x, y=y)

        quadratic = (y.square() / self._covariance_diagonal).sum(dim=-1)
        log_determinant = torch.log(self._covariance_diagonal).sum()
        return -0.5 * (
            self.y_dim * math.log(2.0 * math.pi) + log_determinant + quadratic
        )

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
        if y.ndim != 2 or y.shape != (x.shape[0], self.y_dim):
            raise ValueError(
                f"Expected y with shape ({x.shape[0]}, {self.y_dim}), "
                f"got {tuple(y.shape)}."
            )
