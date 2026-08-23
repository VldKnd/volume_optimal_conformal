# src/data/datasets/synthetic/sinusoidal_transport.py

import math

import torch

from configs.datasets.synthetic.sinusoidal_transport import (
    SinusoidalTransportDatasetConfig,
)
from data.datasets.base import DatasetSplits, XYData
from data.datasets.synthetic.base import BaseSyntheticDataset


class SinusoidalTransportDataset(BaseSyntheticDataset):
    """Unconditional 2m-D target from pairwise sinusoidal transports.

        U ~ N(0, I_{2m})
        X = 0

    The same triangular map is applied independently to every consecutive
    coordinate pair ``(U_{2j-1}, U_{2j})``:

        Y_{2j-1} = U_{2j-1} / vertical_scale
        Y_{2j} = vertical_scale * U_{2j}
                 + amplitude * sin(frequency * U_{2j-1} + phase)

    The Jacobian is triangular:

        [[1 / vertical_scale, 0],
         [amplitude * frequency * cos(frequency * u1 + phase), vertical_scale]]

    Therefore det DY/DU = 1 for every positive ``vertical_scale``. Gaussian
    level sets become sinusoidal, wavy contours in target space. The
    one-dimensional zero condition is retained only for compatibility with the
    conditional pipeline.
    """

    def __init__(self, config: SinusoidalTransportDatasetConfig):
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
        return self.sample_conditional(x, n_samples=1).squeeze(1)

    def sample_conditional(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from the same sinusoidal distribution for every ``x``.

        Args:
            x: (batch, x_dim)
            n_samples: number of samples per x

        Returns:
            y: (batch, n_samples, y_dim)
        """
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

        x_expanded = x[:, None, :].expand(
            x.shape[0],
            n_samples,
            self.x_dim,
        )

        return self.push_u_given_x(u=u, x=x_expanded)

    def push_u_given_x(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Push latent U forward through the fixed triangular map.

        Args:
            u: (..., y_dim)
            x: (..., x_dim)

        Returns:
            y: (..., y_dim)
        """
        u = u.to(device=self.device, dtype=self.dtype)
        x = self._fixed_condition(x)
        self._validate_matching_shapes(point=u, x=x, point_name="u")

        u_pairs = u.reshape(u.shape[:-1] + (self.y_dim // 2, 2))
        u1 = u_pairs[..., 0:1]
        u2 = u_pairs[..., 1:2]

        amplitude, vertical_scale, _ = self._transport_parameters(x)
        amplitude = amplitude.unsqueeze(-2)
        vertical_scale = vertical_scale.unsqueeze(-2)
        wave = amplitude * torch.sin(self.config.frequency * u1 + self.config.phase)
        y1 = u1 / vertical_scale
        y2 = vertical_scale * u2 + wave

        return torch.cat([y1, y2], dim=-1).reshape(u.shape)

    def push_y_given_x(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pull Y back to latent U using the exact triangular inverse.

        Args:
            y: (..., y_dim)
            x: (..., x_dim)

        Returns:
            u: (..., y_dim)
        """
        y = y.to(device=self.device, dtype=self.dtype)
        x = self._fixed_condition(x)
        self._validate_matching_shapes(point=y, x=x, point_name="y")

        y_pairs = y.reshape(y.shape[:-1] + (self.y_dim // 2, 2))
        y1 = y_pairs[..., 0:1]
        y2 = y_pairs[..., 1:2]

        amplitude, vertical_scale, _ = self._transport_parameters(x)
        amplitude = amplitude.unsqueeze(-2)
        vertical_scale = vertical_scale.unsqueeze(-2)
        u1 = vertical_scale * y1
        wave = amplitude * torch.sin(self.config.frequency * u1 + self.config.phase)
        u2 = (y2 - wave) / vertical_scale

        return torch.cat([u1, u2], dim=-1).reshape(y.shape)

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return log |det D_u T_x(u)|.
        """
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
            u.square().sum(dim=-1) + self.y_dim *
            torch.log(torch.tensor(
                2.0 * torch.pi,
                device=u.device,
                dtype=u.dtype,
            ))
        )
        return log_base - self.log_det(x=x, u=u)

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

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.ndim < 1 or x.shape[-1] != self.x_dim:
            raise ValueError(
                f"Expected x with trailing dimension {self.x_dim}, "
                f"got shape {tuple(x.shape)}."
            )

    def _transport_parameters(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_x(x)

        template = x[..., 0:1]
        amplitude = torch.full_like(template, self.config.amplitude)
        log_vertical_scale = torch.full_like(
            template,
            math.log(self.config.vertical_scale),
        )
        vertical_scale = torch.exp(log_vertical_scale)

        return amplitude, vertical_scale, log_vertical_scale

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

        self._validate_x(x)

    def _fixed_condition(
        self,
        x: torch.Tensor,
        require_batch_matrix: bool = False,
    ) -> torch.Tensor:
        """Validate the dummy condition shape and replace its values by zero."""
        x = x.to(device=self.device, dtype=self.dtype)
        self._validate_x(x)
        if require_batch_matrix and x.ndim != 2:
            raise ValueError(
                f"Expected x with shape (batch, {self.x_dim}), "
                f"got {tuple(x.shape)}."
            )
        return torch.zeros_like(x)
