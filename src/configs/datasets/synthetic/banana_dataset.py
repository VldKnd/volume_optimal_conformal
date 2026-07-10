# src/configs/datasets/synthetic/banana_dataset.py

from typing import Literal

from pydantic import BaseModel, Field


class BananaDatasetConfig(BaseModel):
    type: Literal["banana"] = "banana"

    n_train: int = 10_000
    n_calibration: int = 2_000
    n_test: int = 2_000

    x_dim: int = 1
    y_dim: int = 2

    x_low: float = 0.5
    x_high: float = 2.5

    seed: int = 31337
    device: str = "cpu"
    dtype: str = "float32"