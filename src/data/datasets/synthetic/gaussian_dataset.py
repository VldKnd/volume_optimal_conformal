# src/datasets/synthetic/gaussian_target.py

from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch.distributions import Normal

from data.datasets.base import XYData, DatasetSplits
from data.datasets.synthetic.base import BaseSyntheticDataset

from configs.datasets.synthetic.gaussian_dataset import GaussianDatasetConfig

class GaussianDatasetTarget(BaseSyntheticDataset):
    """
    Synthetic conditional Gaussian dataset.

        X ~ N(0, I)
        Y | X=x ~ N(f(x), sigma^2 I)

    with a fixed nonlinear map f: R^{x_dim} -> R^{y_dim}.
    """

    def __init__(self, config: GaussianDatasetConfig):
        self.config = config
        self._splits: DatasetSplits | None = None

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

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

        self.weight = self.weight.to(self.device)
        self.bias = self.bias.to(self.device)

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
    def supports_density(self) -> bool:
        return True

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nonlinear mean map f(x).

        Args:
            x: (batch, x_dim)

        Returns:
            mean: (batch, y_dim)
        """
        linear = x @ self.weight + self.bias

        radial = x.square().mean(dim=-1, keepdim=True)
        nonlinear = 0.5 * torch.sin(linear) + 0.25 * torch.cos(radial)

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

        eps = torch.randn(
            batch_size,
            n_samples,
            self.config.y_dim,
            device=self.device,
            dtype=self.dtype,
        )

        return mean[:, None, :] + self.config.noise_scale * eps

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log p(y | x).

        Args:
            x: (batch, x_dim)
            y: (batch, y_dim)

        Returns:
            log_prob: (batch,)
        """
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        mean = self.mean(x)
        dist = Normal(mean, self.config.noise_scale)

        return dist.log_prob(y).sum(dim=-1)

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