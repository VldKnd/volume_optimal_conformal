# src/calibrators/base.py

from abc import ABC, abstractmethod
import torch


class BaseCalibrator(ABC):
    @abstractmethod
    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        alpha: float,
    ) -> None:
        ...

    @abstractmethod
    def scalar_score(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def contains(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "threshold") or self.threshold is None:
            raise RuntimeError("Calibrator must be fitted before contains().")

        return self.scalar_score(x, scores) <= self.threshold