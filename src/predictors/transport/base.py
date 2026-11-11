# src/predictors/transport/base.py

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from predictors.base import BasePredictor


def checkpoint_config_on_device(
    config_data: Mapping[str, Any],
    map_location: Any,
) -> dict[str, Any]:
    """Return a copy of a saved predictor config whose device follows map_location.

    Predictor checkpoints embed the device they were trained on, typically
    ``cuda``. Loading them on a machine without that device must still work,
    so when ``map_location`` names a device it replaces the saved one. Other
    ``map_location`` forms accepted by ``torch.load`` (``None``, mappings,
    callables) leave the saved device unchanged.
    """
    config_data = dict(config_data)
    if isinstance(map_location, (str, torch.device)):
        config_data["device"] = str(torch.device(map_location))
    return config_data


class BaseTransportPredictor(BasePredictor):

    @abstractmethod
    def pushforward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def pullback(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    def multivariate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.pullback(x, y)

    def log_det(self, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError("Log det is not implemented for this module.")
