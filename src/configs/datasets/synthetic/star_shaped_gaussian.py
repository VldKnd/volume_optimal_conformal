from typing import Literal

from pydantic import BaseModel, Field


class StarShapedGaussianDatasetConfig(BaseModel):
    """Configuration for an unconditional three-petal Gaussian transport."""

    type: Literal["star_shaped_gaussian", "star_shaped"] = ("star_shaped_gaussian")

    n_train: int = Field(default=10_000, gt=0)
    n_calibration: int = Field(default=2_000, ge=0)
    n_test: int = Field(default=2_000, ge=0)

    x_dim: Literal[1] = 1
    y_dim: int = Field(default=2, gt=0, multiple_of=2)
    # Compatibility fields are fixed at zero and cannot restore conditioning.
    x_low: Literal[0.0] = 0.0
    x_high: Literal[0.0] = 0.0

    petal_amplitude: float = Field(
        default=0.45,
        gt=0.1,
        lt=1.0,
        description=(
            "Strength of the area-preserving three-petal deformation. "
            "Values above 0.1 make the valleys strictly concave, while "
            "values below one keep the transport invertible."
        ),
    )
    rotation_fraction: Literal[0.0] = 0.0

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
