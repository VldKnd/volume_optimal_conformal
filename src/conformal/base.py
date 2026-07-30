from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import torch
from torch.utils.data import DataLoader

from conformal.calibrators.base import BaseCalibrator


class ConformalPredictor(ABC):
    """Shared interface for conformal prediction regions."""

    config: Any
    calibrator: BaseCalibrator
    predictor: Any
    base_predictor: Any
    x_dim: int
    y_dim: int
    device: torch.device
    dtype: torch.dtype

    @property
    def coverage_mass(self) -> float:
        return float(self.config.coverage_mass)

    @property
    def threshold(self) -> torch.Tensor | None:
        return getattr(self.calibrator, "threshold", None)

    @property
    def is_calibrated(self) -> bool:
        return self.threshold is not None

    @abstractmethod
    def calibrate(self, dataloader: DataLoader) -> Self:
        """Fit the conformal calibrator."""

    def fit(self, dataloader: DataLoader) -> Self:
        """Alias for :meth:`calibrate`."""
        return self.calibrate(dataloader)

    @abstractmethod
    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return one multivariate nonconformity score per observation."""

    def scalar_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.multivariate_score(x, y)
        return self.calibrator.scalar_score(
            x=x.to(device=scores.device, dtype=scores.dtype),
            scores=scores,
        )

    def contains(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether each observation belongs to its conformal region."""
        self._require_calibrated()
        with torch.no_grad():
            scores = self.multivariate_score(x, y)
            return self.calibrator.contains(
                x=x.to(device=scores.device, dtype=scores.dtype),
                scores=scores,
            )

    @abstractmethod
    def estimate_log_volume(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return one conformal-region log-volume per covariate."""

    def estimate_volume(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return torch.exp(self.estimate_log_volume(x=x, **kwargs))

    def log_volume(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.estimate_log_volume(x=x, **kwargs)

    def volume(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.estimate_volume(x=x, **kwargs)

    def _set_predictor_eval(self) -> None:
        eval_method = getattr(self.predictor, "eval", None)
        if callable(eval_method):
            eval_method()

    def _require_calibrated(self) -> None:
        if not self.is_calibrated:
            raise RuntimeError(
                f"{type(self).__name__} must be calibrated before this operation."
            )
