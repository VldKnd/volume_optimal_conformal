from typing import Literal
from pydantic import BaseModel, Field

class StudentTDatasetConfig(BaseModel):
    type: Literal["student_t_dataset"] = "student_t_dataset"
    n_train: int = 10_000
    n_calibration: int = 2_000
    n_test: int = 2_000
    x_dim: int = 10
    y_dim: int = 2
    df: float = Field(default=5.0, gt=2.0)
    min_scale: float = Field(default=0.2, gt=0.0)
    max_scale: float = Field(default=1.0, gt=0.0)
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"