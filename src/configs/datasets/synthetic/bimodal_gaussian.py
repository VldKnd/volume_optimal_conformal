from typing import Literal

from pydantic import BaseModel, Field


class BimodalGaussianDatasetConfig(BaseModel):
    type: Literal["bimodal_gaussian"] = "bimodal_gaussian"

    n_train: int = 10_000
    n_calibration: int = 2_000
    n_test: int = 2_000

    # The law is unconditional; x is a dummy zero covariate for compatibility
    # with the existing conditional dataset interface.
    x_dim: int = Field(default=1, ge=0)
    y_dim: int = 2

    mode_offset: float = Field(default=1.0, gt=0.0)
    noise_scale: float = Field(default=1.0, gt=0.0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
