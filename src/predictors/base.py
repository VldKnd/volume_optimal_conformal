# src/predictors/base.py

from abc import ABC, abstractmethod
import torch


class BasePredictor(ABC):
    @abstractmethod
    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            score: (batch, y_dim)
        """