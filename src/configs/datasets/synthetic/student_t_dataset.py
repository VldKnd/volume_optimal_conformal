from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class StudentTDatasetConfig(BaseModel):
    """Configuration for an unconditional multivariate Student-t dataset."""

    type: Literal["student_t_dataset"] = "student_t_dataset"

    n_train: int = Field(default=10_000, gt=0)
    n_calibration: int = Field(default=2_000, ge=0)
    n_test: int = Field(default=2_000, ge=0)

    # A single zero-valued dummy condition preserves the conditional pipeline API.
    x_dim: Literal[1] = 1
    y_dim: int = Field(default=2, gt=0)

    nu: float = Field(
        default=3.0,
        gt=0.0,
        validation_alias=AliasChoices("nu", "df"),
        description="Student-t degrees of freedom controlling tail weight.",
    )
    k: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Positive parameter defining the determinant-one diagonal scale "
            "matrix with first entry k ** (1 - 1/d) and all remaining "
            "entries k ** (-1/d)."
        ),
    )

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    @property
    def df(self) -> float:
        """Backward-compatible name for the degrees of freedom."""
        return self.nu
