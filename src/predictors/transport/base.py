# src/predictors/transport/base.py

from abc import abstractmethod
import torch

from predictors.base import BasePredictor


class BaseTransportPredictor(BasePredictor):
    @abstractmethod
    def pushforward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def pullback(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    def multivariate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.pullback(x, y)