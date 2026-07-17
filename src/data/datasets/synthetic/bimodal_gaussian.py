# src/datasets/synthetic/bimodal_gaussian.py

import torch

from configs.datasets.synthetic.bimodal_gaussian import (
    BimodalGaussianDatasetConfig,
)
from data.datasets.base import DatasetSplits, XYData
from data.datasets.synthetic.base import BaseSyntheticDataset


class BimodalGaussianDataset(BaseSyntheticDataset):
    """
    Unconditional bimodal Gaussian target used in the figure-1 experiment.

        U ~ N(0, I_2)

        Y ~ 0.5 N((-1, 0), I_2) + 0.5 N((1, 0), I_2)

    The dataset still returns an x tensor so it can be used through the
    conditional dataset interface. By default x is a single dummy zero feature.
    """

    def __init__(self, config: BimodalGaussianDatasetConfig):
        self.config = config
        self._splits: DatasetSplits | None = None

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        if config.y_dim != 2:
            raise ValueError("BimodalGaussianDataset requires y_dim = 2.")

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(config.seed)

        self._modes = torch.tensor(
            [
                [-config.mode_offset, 0.0],
                [config.mode_offset, 0.0],
            ],
            device=self.device,
            dtype=self.dtype,
        )

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
    def modes(self) -> torch.Tensor:
        return self._modes

    @property
    def covariance(self) -> torch.Tensor:
        return torch.eye(
            self.y_dim,
            device=self.device,
            dtype=self.dtype,
        ) * self.config.noise_scale**2

    @property
    def supports_density(self) -> bool:
        return True

    def sample_x(self, n: int) -> torch.Tensor:
        return torch.zeros(
            n,
            self.config.x_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def sample_source(self, n: int) -> torch.Tensor:
        """
        Sample the unimodal source law U ~ N(0, I_2).

        This is useful for experiments that explicitly construct an OT map from
        the standard Gaussian source to the bimodal Gaussian target.
        """
        u = torch.randn(
            n,
            self.y_dim,
            generator=self._generator,
            dtype=self.dtype,
        )
        return u.to(self.device)

    def sample_target(self, n: int) -> torch.Tensor:
        x = self.sample_x(n)
        return self.sample_conditional(x, n_samples=1).squeeze(1)

    def sample_conditional(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample Y | X=x. The conditional law does not depend on x.

        Args:
            x: (batch, x_dim)
            n_samples: number of samples per x

        Returns:
            y: (batch, n_samples, 2)
        """
        x = x.to(device=self.device, dtype=self.dtype)

        if x.shape[-1] != self.config.x_dim:
            raise ValueError(
                f"Expected x.shape[-1] = {self.config.x_dim}, "
                f"got {x.shape[-1]}."
            )

        batch_size = x.shape[0]

        component = torch.randint(
            low=0,
            high=2,
            size=(batch_size, n_samples),
            generator=self._generator,
        ).to(self.device)

        eps = torch.randn(
            batch_size,
            n_samples,
            self.y_dim,
            generator=self._generator,
            dtype=self.dtype,
        ).to(self.device)

        return self.modes[component] + self.config.noise_scale * eps

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log p(y | x) for the equal-weight bimodal Gaussian mixture.

        Args:
            x: (batch, x_dim)
            y: (batch, 2)

        Returns:
            log_prob: (batch,)
        """
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        if x.shape[-1] != self.config.x_dim:
            raise ValueError(
                f"Expected x.shape[-1] = {self.config.x_dim}, "
                f"got {x.shape[-1]}."
            )

        if y.shape[-1] != self.y_dim:
            raise ValueError(
                f"Expected y.shape[-1] = {self.y_dim}, got {y.shape[-1]}."
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Expected x.shape[0] == y.shape[0], got "
                f"{x.shape[0]} and {y.shape[0]}."
            )

        z = (y[:, None, :] - self.modes[None, :, :]) / self.config.noise_scale

        log_normalizer = self.y_dim * torch.log(
            torch.tensor(
                self.config.noise_scale,
                device=self.device,
                dtype=self.dtype,
            )
        ) + 0.5 * self.y_dim * torch.log(
            torch.tensor(
                2.0 * torch.pi,
                device=self.device,
                dtype=self.dtype,
            )
        )

        log_components = -0.5 * z.square().sum(dim=-1) - log_normalizer
        log_weights = torch.log(
            torch.full(
                (2,),
                0.5,
                device=self.device,
                dtype=self.dtype,
            )
        )

        return torch.logsumexp(log_components + log_weights, dim=-1)

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
