from typing import Literal

from pydantic import BaseModel, Field


class EllipticalGaussianDatasetConfig(BaseModel):
    """Configuration for an unconditional determinant-one Gaussian target."""

    type: Literal["elliptical_gaussian_dataset"] = "elliptical_gaussian_dataset"

    n_train: int = Field(default=10_000, gt=0)
    n_calibration: int = Field(default=2_000, ge=0)
    n_test: int = Field(default=2_000, ge=0)

    # A zero-valued dummy condition preserves the conditional pipeline API.
    x_dim: Literal[1] = 1
    y_dim: int = Field(default=2, gt=0)
    k: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Condition number of the determinant-one diagonal covariance: "
            "the first entry is k ** (1 - 1/d), and all remaining entries "
            "are k ** (-1/d)."
        ),
    )

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
