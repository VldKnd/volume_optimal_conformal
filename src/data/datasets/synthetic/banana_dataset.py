# src/datasets/synthetic/banana_dataset.py

import torch

from data.datasets.base import XYData, DatasetSplits
from data.datasets.synthetic.base import BaseSyntheticDataset
from configs.datasets.synthetic.banana_dataset import BananaDatasetConfig


class BananaDataset(BaseSyntheticDataset):
    """
    Synthetic banana-shaped conditional dataset.

        X ~ Uniform([x_low, x_high])

        U ~ N(0, I_2)

        Y_1 = U_1 X
        Y_2 = U_2 / X + U_1^2 + X^3

    Hence Y | X=x is non-Gaussian and banana-shaped.
    """

    def __init__(self, config: BananaDatasetConfig):
        self.config = config
        self._splits: DatasetSplits | None = None

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(config.seed)

        if config.x_dim != 1:
            raise ValueError("BananaDataset requires x_dim = 1.")

        if config.y_dim != 2:
            raise ValueError("BananaDataset requires y_dim = 2.")

        if config.x_high <= config.x_low:
            raise ValueError("x_high must be larger than x_low.")

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

    def sample_x(self, n: int) -> torch.Tensor:
        x = (
            self.config.x_low
            + (self.config.x_high - self.config.x_low)
            * torch.rand(
                n,
                1,
                generator=self._generator,
                dtype=self.dtype,
            )
        )

        return x.to(self.device)

    def sample_conditional(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample Y | X=x.

        Args:
            x: (batch, 1)
            n_samples: number of conditional samples per x

        Returns:
            y: (batch, n_samples, 2)
        """
        x = x.to(device=self.device, dtype=self.dtype)

        if x.shape[-1] != 1:
            raise ValueError(f"Expected x.shape[-1] = 1, got {x.shape[-1]}.")

        batch_size = x.shape[0]

        u = torch.randn(
            batch_size,
            n_samples,
            2,
            device=self.device,
            dtype=self.dtype,
        )

        x_expanded = x[:, None, :]  # (batch, 1, 1)

        y1 = u[..., 0:1] * x_expanded
        y2 = (
            u[..., 1:2] / x_expanded
            + u[..., 0:1].square()
            + x_expanded.pow(3)
        )

        return torch.cat([y1, y2], dim=-1)

    def sample_joint(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.sample_x(n)
        y = self.sample_conditional(x, n_samples=1).squeeze(1)
        return x, y

    def push_y_given_x(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pull Y back to latent U given X.

        Args:
            y: (batch, 2)
            x: (batch, 1)

        Returns:
            u: (batch, 2)
        """
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        if y.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Expected y.shape[:-1] == x.shape[:-1], got "
                f"{y.shape[:-1]} and {x.shape[:-1]}."
            )

        if y.shape[-1] != 2:
            raise ValueError(f"Expected y.shape[-1] = 2, got {y.shape[-1]}.")

        if x.shape[-1] != 1:
            raise ValueError(f"Expected x.shape[-1] = 1, got {x.shape[-1]}.")

        y_flat = y.reshape(-1, 2)
        x_flat = x.reshape(-1, 1)

        u1 = y_flat[:, 0:1] / x_flat
        u2 = (
            y_flat[:, 1:2]
            - u1.square()
            - x_flat.pow(3)
        ) * x_flat

        u = torch.cat([u1, u2], dim=-1)

        return u.reshape(y.shape[:-1] + (2,))

    def push_u_given_x(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Push latent U forward to Y given X.

        Args:
            u: (..., 2)
            x: (..., 1)

        Returns:
            y: (..., 2)
        """
        x = x.to(device=self.device, dtype=self.dtype)
        u = u.to(device=self.device, dtype=self.dtype)

        if u.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Expected u.shape[:-1] == x.shape[:-1], got "
                f"{u.shape[:-1]} and {x.shape[:-1]}."
            )

        if u.shape[-1] != 2:
            raise ValueError(f"Expected u.shape[-1] = 2, got {u.shape[-1]}.")

        if x.shape[-1] != 1:
            raise ValueError(f"Expected x.shape[-1] = 1, got {x.shape[-1]}.")

        y1 = u[..., 0:1] * x
        y2 = u[..., 1:2] / x + u[..., 0:1].square() + x.pow(3)

        return torch.cat([y1, y2], dim=-1)

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log p(y | x) by change of variables.

        The map u -> y has Jacobian determinant 1 / x^2? Actually:

            y1 = x u1
            y2 = u2 / x + u1^2 + x^3

        Jacobian wrt u:

            [[x, 0],
             [2u1, 1/x]]

        det = 1.

        Therefore log p(y | x) = log phi(u).
        """
        u = self.push_y_given_x(y=y, x=x)
        return -0.5 * (
            u.square().sum(dim=-1)
            + self.y_dim * torch.log(
                torch.tensor(
                    2.0 * torch.pi,
                    device=u.device,
                    dtype=u.dtype,
                )
            )
        )

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