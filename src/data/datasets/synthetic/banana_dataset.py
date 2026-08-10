# src/datasets/synthetic/banana_dataset.py

import torch

from data.datasets.base import XYData, DatasetSplits
from data.datasets.synthetic.base import BaseSyntheticDataset
from configs.datasets.synthetic.banana_dataset import BananaDatasetConfig


class BananaDataset(BaseSyntheticDataset):
    """Synthetic banana-shaped distribution with a fixed dummy condition.

        X = 2

        U ~ N(0, I_2)

        Y_1 = 2 U_1
        Y_2 = U_2 / 2 + U_1^2 + 8

    The one-dimensional ``x`` tensor is retained for compatibility with the
    conditional modeling pipeline, but the target distribution is independent
    of the supplied condition values.
    """

    CONDITION_VALUE = 2.0

    def __init__(self, config: BananaDatasetConfig):
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
        return torch.full(
            (n, self.x_dim),
            self.CONDITION_VALUE,
            device=self.device,
            dtype=self.dtype,
        )

    def sample_conditional(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from the same banana distribution for every ``x``.

        Args:
            x: (batch, 1)
            n_samples: number of conditional samples per x

        Returns:
            y: (batch, n_samples, 2)
        """
        if (
            isinstance(n_samples, bool) or not isinstance(n_samples, int)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer.")

        x = self._fixed_condition(x, require_batch_matrix=True)
        batch_size = x.shape[0]

        u = torch.randn(
            batch_size,
            n_samples,
            2,
            generator=self._generator,
            dtype=self.dtype,
        ).to(self.device)

        x_expanded = x[:, None, :]  # (batch, 1, 1)

        y1 = u[..., 0:1] * x_expanded
        y2 = (u[..., 1:2] / x_expanded + u[..., 0:1].square() + x_expanded.pow(3))

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
        x = self._fixed_condition(x)
        y = y.to(device=self.device, dtype=self.dtype)

        if y.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Expected y.shape[:-1] == x.shape[:-1], got "
                f"{y.shape[:-1]} and {x.shape[:-1]}."
            )

        if y.shape[-1] != 2:
            raise ValueError(f"Expected y.shape[-1] = 2, got {y.shape[-1]}.")

        y_flat = y.reshape(-1, 2)
        x_flat = x.reshape(-1, 1)

        u1 = y_flat[:, 0:1] / x_flat
        u2 = (y_flat[:, 1:2] - u1.square() - x_flat.pow(3)) * x_flat

        u = torch.cat([u1, u2], dim=-1)

        return u.reshape(y.shape[:-1] + (2, ))

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
        x = self._fixed_condition(x)
        u = u.to(device=self.device, dtype=self.dtype)

        if u.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Expected u.shape[:-1] == x.shape[:-1], got "
                f"{u.shape[:-1]} and {x.shape[:-1]}."
            )

        if u.shape[-1] != 2:
            raise ValueError(f"Expected u.shape[-1] = 2, got {u.shape[-1]}.")

        y1 = u[..., 0:1] * x
        y2 = u[..., 1:2] / x + u[..., 0:1].square() + x.pow(3)

        return torch.cat([y1, y2], dim=-1)

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Return the identically zero ``log |det D_u T(u)|``."""
        x = self._fixed_condition(x)
        u = u.to(device=self.device, dtype=self.dtype)

        if u.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Expected u.shape[:-1] == x.shape[:-1], got "
                f"{u.shape[:-1]} and {x.shape[:-1]}."
            )
        if u.shape[-1] != self.y_dim:
            raise ValueError(f"Expected u.shape[-1] = {self.y_dim}, got {u.shape[-1]}.")

        return torch.zeros(u.shape[:-1], device=u.device, dtype=u.dtype)

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

        Therefore ``log_det(x, u) = 0`` and ``log p(y | x) = log phi(u)``.
        """
        u = self.push_y_given_x(y=y, x=x)
        return -0.5 * (
            u.square().sum(dim=-1) + self.y_dim *
            torch.log(torch.tensor(
                2.0 * torch.pi,
                device=u.device,
                dtype=u.dtype,
            ))
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

    def _fixed_condition(
        self,
        x: torch.Tensor,
        require_batch_matrix: bool = False,
    ) -> torch.Tensor:
        """Validate the dummy condition shape and replace its values by two."""
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
        return torch.full_like(x, self.CONDITION_VALUE)
