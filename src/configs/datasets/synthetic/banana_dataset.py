# src/configs/datasets/synthetic/banana_dataset.py

from typing import Literal

from pydantic import BaseModel, Field


class BananaDatasetConfig(BaseModel):
    """Configuration for the unconditional banana-shaped dataset."""

    type: Literal["banana"] = "banana"

    n_train: int = Field(default=10_000, gt=0)
    n_calibration: int = Field(default=2_000, ge=0)
    n_test: int = Field(default=2_000, ge=0)

    # A fixed one-dimensional dummy condition preserves the conditional APIs.
    x_dim: Literal[1] = 1
    y_dim: int = Field(default=2, ge=2)

    seed: int = 31337
    device: str = "cpu"
    dtype: str = "float32"
