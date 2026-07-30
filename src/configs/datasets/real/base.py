import math
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator


class BaseRealDatasetConfig(BaseModel):
    file_path: Path
    x_dim: int
    y_dim: int

    train_fraction: float = Field(default=0.6, gt=0.0, lt=1.0)
    calibration_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    @model_validator(mode="after")
    def validate_split_fractions(self) -> Self:
        total = (self.train_fraction + self.calibration_fraction + self.test_fraction)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Dataset split fractions must sum to 1.")
        return self
