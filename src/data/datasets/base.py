# src/datasets/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class XYData:
    x: torch.Tensor  # (n, x_dim)
    y: torch.Tensor  # (n, y_dim)


@dataclass(frozen=True)
class DatasetSplits:
    train: XYData
    calibration: XYData
    test: XYData


class BaseDataset(ABC):
    @abstractmethod
    def prepare(self) -> None:
        ...

    @abstractmethod
    def get_splits(self) -> DatasetSplits:
        ...

    @property
    @abstractmethod
    def x_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def y_dim(self) -> int:
        ...