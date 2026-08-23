# src/configs/datasets/synthetic/sinusoidal_transport.py

from typing import Literal

from pydantic import BaseModel, Field


class SinusoidalTransportDatasetConfig(BaseModel):
    """Configuration for an unconditional sinusoidal transport."""

    type: Literal["sinusoidal_transport"] = "sinusoidal_transport"

    n_train: int = Field(default=10_000, gt=0)
    n_calibration: int = Field(default=2_000, ge=0)
    n_test: int = Field(default=2_000, ge=0)

    x_dim: Literal[1] = 1
    y_dim: int = Field(default=2, gt=0, multiple_of=2)
    # Compatibility fields are fixed at zero and cannot restore conditioning.
    x_low: Literal[0.0] = 0.0
    x_high: Literal[0.0] = 0.0

    amplitude: float = 1.0
    amplitude_x_scale: Literal[0.0] = 0.0
    frequency: float = Field(default=2.0, gt=0.0)
    phase: float = 0.0
    vertical_scale: float = Field(default=1.0, gt=0.0)
    vertical_scale_x_scale: Literal[0.0] = 0.0

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
