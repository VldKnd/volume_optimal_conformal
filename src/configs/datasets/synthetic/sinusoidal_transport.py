# src/configs/datasets/synthetic/sinusoidal_transport.py

from typing import Literal

from pydantic import BaseModel, Field


class SinusoidalTransportDatasetConfig(BaseModel):
    type: Literal["sinusoidal_transport"] = "sinusoidal_transport"

    n_train: int = Field(default=10_000, gt=0)
    n_calibration: int = Field(default=2_000, ge=0)
    n_test: int = Field(default=2_000, ge=0)

    x_dim: int = Field(default=1, gt=0)
    y_dim: int = 2

    x_low: float = -1.0
    x_high: float = 1.0

    amplitude: float = 1.0
    amplitude_x_scale: float = 0.5
    frequency: float = Field(default=2.0, gt=0.0)
    phase: float = 0.0
    vertical_scale: float = Field(default=1.0, gt=0.0)
    vertical_scale_x_scale: float = 0.35

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
