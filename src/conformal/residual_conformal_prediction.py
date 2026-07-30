from __future__ import annotations

from typing import Any, Self

import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from configs.conformal import ResidualConformalPredictionConfig
from conformal.calibrators.elliptic_calibrator import EllipticCalibrator
from conformal.calibrators.factory import make_calibrator


class ResidualConformalPrediction:
    """Conformal prediction regions built from regression residuals."""

    def __init__(
        self,
        predictor: Any,
        config: ResidualConformalPredictionConfig,
    ):
        self.predictor = predictor
        self.base_predictor = predictor
        self.config = config

        self.x_dim = predictor.x_dim
        self.y_dim = predictor.y_dim
        self.device = torch.device(predictor.device)
        self.dtype = predictor.dtype

        self.calibrator = make_calibrator(config.calibrator)
        self.calibration_x: torch.Tensor | None = None
        self.calibration_residuals: torch.Tensor | None = None
        self.volume_neighbors: NearestNeighbors | None = None

    @property
    def coverage_mass(self) -> float:
        return self.config.coverage_mass

    @property
    def threshold(self) -> torch.Tensor | None:
        return self.calibrator.threshold

    @property
    def is_calibrated(self) -> bool:
        return self.threshold is not None

    def calibrate(self, dataloader: DataLoader) -> Self:
        """Fit the calibrator from batches of calibration observations."""
        calibration_x = []
        calibration_residuals = []
        cpu_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32

        for x_batch, y_batch in dataloader:
            with torch.no_grad():
                residuals = self.multivariate_score(x_batch, y_batch)

            calibration_x.append(x_batch.detach().to(device="cpu", dtype=cpu_dtype))
            calibration_residuals.append(
                residuals.detach().to(device="cpu", dtype=cpu_dtype)
            )

        if not calibration_residuals:
            raise ValueError(
                "Calibration dataloader must contain at least one observation."
            )

        x = torch.cat(calibration_x, dim=0)
        residuals = torch.cat(calibration_residuals, dim=0)
        if not torch.isfinite(residuals).all():
            raise ValueError("Calibration residuals must be finite.")

        self.calibrator.fit(
            x=x,
            scores=residuals,
            coverage_mass=self.coverage_mass,
        )
        self.calibration_x = x
        self.calibration_residuals = residuals
        self.volume_neighbors = NearestNeighbors(
            n_neighbors=self.config.volume_n_neighbors,
        )
        self.volume_neighbors.fit(x.numpy())
        return self

    def fit(self, dataloader: DataLoader) -> Self:
        return self.calibrate(dataloader)

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return regression residuals ``y - f(x)``."""
        self._validate_inputs(x, y)
        self._set_predictor_eval()
        residuals = self.predictor.multivariate_score(x=x, y=y)

        expected_shape = (x.shape[0], self.y_dim)
        if not isinstance(residuals, torch.Tensor):
            raise TypeError("predictor.multivariate_score must return a tensor.")
        if tuple(residuals.shape) != expected_shape:
            raise ValueError(
                "predictor.multivariate_score must return shape "
                f"{expected_shape}, got {tuple(residuals.shape)}."
            )
        return residuals

    def scalar_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        residuals = self.multivariate_score(x, y)
        return self.calibrator.scalar_score(
            x=x.to(device=residuals.device, dtype=residuals.dtype),
            scores=residuals,
        )

    def contains(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether each observation belongs to its prediction region."""
        self._require_calibrated()
        with torch.no_grad():
            residuals = self.multivariate_score(x, y)
            return self.calibrator.contains(
                x=x.to(device=residuals.device, dtype=residuals.dtype),
                scores=residuals,
            )

    def estimate_log_volume(
        self,
        x: torch.Tensor,
        number_of_samples: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Return one prediction-region log-volume per covariate.

        Elliptic regions use their exact volume. For every other calibrator,
        Monte Carlo estimates the volume of the conformal residual region
        intersected with the residual bounding box formed from the nearest
        calibration covariates.
        """
        self._require_calibrated()
        x = self._prepare_x(x)
        if x.shape[0] == 0:
            return x.new_empty(0)

        if isinstance(self.calibrator, EllipticCalibrator):
            return self.calibrator.estimate_log_volume(x)

        if self.volume_neighbors is None or self.calibration_residuals is None:
            raise RuntimeError("Calibration nearest neighbors are unavailable.")
        if self.config.volume_n_neighbors > self.calibration_residuals.shape[0]:
            raise ValueError(
                "volume_n_neighbors cannot exceed the number of calibration "
                f"observations ({self.calibration_residuals.shape[0]})."
            )

        number_of_samples = (
            self.config.volume_mc_samples
            if number_of_samples is None else number_of_samples
        )
        seed = self.config.volume_seed if seed is None else seed
        if number_of_samples < 1:
            raise ValueError("number_of_samples must be positive.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        uniform_samples = torch.rand(
            number_of_samples,
            self.y_dim,
            dtype=self.calibration_residuals.dtype,
            generator=generator,
        )
        query_x = x.detach().to(
            device="cpu",
            dtype=self.calibration_residuals.dtype,
        )
        neighbor_indices = self.volume_neighbors.kneighbors(
            query_x.numpy(),
            return_distance=False,
        )

        log_volumes = []
        with torch.no_grad():
            for x_value, indices in zip(
                query_x,
                neighbor_indices,
                strict=True,
            ):
                index = torch.as_tensor(indices, dtype=torch.long)
                local_residuals = self.calibration_residuals.index_select(
                    0,
                    index,
                )
                residual_min = local_residuals.amin(dim=0)
                residual_max = local_residuals.amax(dim=0)
                widths = residual_max - residual_min

                if torch.any(widths == 0):
                    log_volumes.append(widths.new_tensor(-torch.inf))
                    continue

                residual_samples = residual_min + widths * uniform_samples
                repeated_x = x_value.unsqueeze(0).expand(
                    number_of_samples,
                    self.x_dim,
                )
                inside = self.calibrator.contains(
                    x=repeated_x,
                    scores=residual_samples,
                )
                inside_probability = inside.to(residual_samples.dtype).mean()
                log_volumes.append(
                    torch.log(widths).sum() + torch.log(inside_probability)
                )

        return torch.stack(log_volumes).to(
            device=x.device,
            dtype=x.dtype,
        )

    def estimate_volume(
        self,
        x: torch.Tensor,
        number_of_samples: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        return torch.exp(
            self.estimate_log_volume(
                x=x,
                number_of_samples=number_of_samples,
                seed=seed,
            )
        )

    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch tensor.")
        if x.ndim != 2 or x.shape[1] != self.x_dim:
            raise ValueError(
                f"Expected x with shape (n, {self.x_dim}), got {tuple(x.shape)}."
            )
        return x.to(device=self.device, dtype=self.dtype)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        if x.ndim != 2 or x.shape[1] != self.x_dim:
            raise ValueError(
                f"Expected x with shape (n, {self.x_dim}), got {tuple(x.shape)}."
            )
        if y.ndim != 2 or y.shape[1] != self.y_dim:
            raise ValueError(
                f"Expected y with shape (n, {self.y_dim}), got {tuple(y.shape)}."
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same batch size.")

    def _set_predictor_eval(self) -> None:
        eval_method = getattr(self.predictor, "eval", None)
        if callable(eval_method):
            eval_method()

    def _require_calibrated(self) -> None:
        if not self.is_calibrated:
            raise RuntimeError("ResidualConformalPrediction must be calibrated first.")
