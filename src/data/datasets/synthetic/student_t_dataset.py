# src/datasets/synthetic/student_t_dataset.py

from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch.distributions import StudentT

from data.datasets.base import XYData, DatasetSplits
from data.datasets.synthetic.base import BaseSyntheticDataset
from configs.datasets.synthetic.student_t_dataset import StudentTDatasetConfig

class StudentTDataset(BaseSyntheticDataset):
    """
    Conditional non-Gaussian student t dataset.

        X ~ N(0, I)
        Y | X=x = f(x) + Sigma^{1/2} Z

    where Z has independent Student-t coordinates and Sigma is diagonal
    positive definite with different diagonal entries.

    Strictly speaking, independent-coordinate Student-t is not the canonical
    multivariate Student-t law. If you want a genuinely elliptic
    Student-t distribution, use sample_radial_conditional() below instead.
    """

    def __init__(self, config: StudentTDatasetConfig):
        self.config = config
        self._splits: DatasetSplits | None = None

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        if config.max_scale <= config.min_scale:
            raise ValueError("max_scale must be larger than min_scale.")

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(config.seed)

        self.weight = torch.randn(
            config.x_dim,
            config.y_dim,
            generator=self._generator,
            dtype=self.dtype,
        ) / config.x_dim**0.5

        self.bias = 0.2 * torch.randn(
            config.y_dim,
            generator=self._generator,
            dtype=self.dtype,
        )

        # Positive, all-different diagonal scales.
        self.scales = torch.linspace(
            config.min_scale,
            config.max_scale,
            config.y_dim,
            dtype=self.dtype,
        )

        self.weight = self.weight.to(self.device)
        self.bias = self.bias.to(self.device)
        self.scales = self.scales.to(self.device)

    @property
    def x_dim(self) -> int:
        return self.config.x_dim

    @property
    def y_dim(self) -> int:
        return self.config.y_dim

    @property
    def n_total(self) -> int:
        return (
            self.config.n_train
            + self.config.n_calibration
            + self.config.n_test
        )

    @property
    def covariance(self) -> torch.Tensor:
        """
        Diagonal PSD covariance-like matrix.

        Shape:
            (y_dim, y_dim)
        """
        return torch.diag(self.scales.square())

    @property
    def supports_density(self) -> bool:
        return True

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nonlinear conditional center f(x).

        Args:
            x: (batch, x_dim)

        Returns:
            mean: (batch, y_dim)
        """
        linear = x @ self.weight + self.bias
        radial = x.square().mean(dim=-1, keepdim=True)

        nonlinear = (
            0.5 * torch.sin(linear)
            + 0.25 * torch.cos(radial)
        )

        return linear + nonlinear

    def sample_x(self, n: int) -> torch.Tensor:
        x = torch.randn(
            n,
            self.config.x_dim,
            generator=self._generator,
            dtype=self.dtype,
        )
        return x.to(self.device)

    def sample_conditional(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from Y | X=x.

        Args:
            x: (batch, x_dim)
            n_samples: number of samples per x

        Returns:
            y: (batch, n_samples, y_dim)
        """
        x = x.to(device=self.device, dtype=self.dtype)

        batch_size = x.shape[0]
        mean = self.mean(x)

        dist = StudentT(df=self.config.df)

        eps = dist.sample(
            sample_shape=(batch_size, n_samples, self.config.y_dim)
        ).to(device=self.device, dtype=self.dtype)

        return mean[:, None, :] + self.scales[None, None, :] * eps

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Coordinate-wise Student-t log-density.

        Args:
            x: (batch, x_dim)
            y: (batch, y_dim)

        Returns:
            log_prob: (batch,)
        """
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        mean = self.mean(x)
        z = (y - mean) / self.scales

        dist = StudentT(df=self.config.df)

        log_base = dist.log_prob(z).sum(dim=-1)
        log_det = torch.log(self.scales).sum()

        return log_base - log_det

    def prepare(self) -> None:
        x, y = self.sample_joint(self.n_total)

        n_train = self.config.n_train
        n_cal = self.config.n_calibration
        n_test = self.config.n_test

        self._splits = DatasetSplits(
            train=XYData(
                x=x[:n_train],
                y=y[:n_train],
            ),
            calibration=XYData(
                x=x[n_train:n_train + n_cal],
                y=y[n_train:n_train + n_cal],
            ),
            test=XYData(
                x=x[n_train + n_cal:n_train + n_cal + n_test],
                y=y[n_train + n_cal:n_train + n_cal + n_test],
            ),
        )

    def get_splits(self) -> DatasetSplits:
        if self._splits is None:
            self.prepare()

        assert self._splits is not None
        return self._splits