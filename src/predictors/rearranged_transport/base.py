# src/predictors/rearranged_transport/base.py

from abc import abstractmethod
import torch

from predictors.base import BasePredictor


class BaseRearrangedTransportPredictor(BasePredictor):

    @abstractmethod
    def rearrangement_pushforward(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def rearrangement_pullback(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def transport_pushforward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def transport_pullback(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def pushforward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def pullback(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    def multivariate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.pullback(x, y)

    def transport_log_det(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError("Log det is not implemented for this module.")

    def rearrangement_log_det(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError("Log det is not implemented for this module.")

    def log_det(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError("Log det is not implemented for this module.")
