from abc import ABC, abstractmethod

import torch


class BaseCalibrator(ABC):
    """Convert multivariate scores to scalar scores and fit a threshold."""

    threshold: torch.Tensor | None

    @abstractmethod
    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        coverage_mass: float,
    ) -> None:
        """Fit a threshold at the requested probability coverage."""

    @abstractmethod
    def scalar_score(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return one scalar nonconformity score per input point."""

    def contains(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether each multivariate score is inside the fitted region."""
        if not hasattr(self, "threshold") or self.threshold is None:
            raise RuntimeError("Calibrator must be fitted before contains().")

        scalar_scores = self.scalar_score(x, scores)
        threshold = self.threshold.to(
            device=scalar_scores.device,
            dtype=scalar_scores.dtype,
        )
        return scalar_scores <= threshold
