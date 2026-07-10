
from typing import Literal
from pydantic import BaseModel, Field

class GaussianDatasetConfig(BaseModel):
    type: Literal["gaussian_dataset"] = "gaussian_dataset"

    n_train: int = 10_000
    n_calibration: int = 2_000
    n_test: int = 2_000

    x_dim: int = 10
    y_dim: int = 2

    noise_scale: float = Field(default=0.3, gt=0.0)
    seed: int = 0

    device: str = "cpu"
    dtype: str = "float32"
