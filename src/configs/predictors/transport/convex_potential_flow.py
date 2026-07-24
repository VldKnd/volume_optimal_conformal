from typing import Literal

from pydantic import BaseModel, Field


class ConvexPotentialFlowPredictorConfig(BaseModel):
    type: Literal["convex_potential_flow"] = "convex_potential_flow"

    x_dim: int = Field(ge=0)
    y_dim: int = Field(gt=0)

    hidden_dim: int = Field(default=64, gt=0)
    num_hidden_layers: int = Field(default=2, gt=0)
    num_blocks: int = Field(default=4, gt=0)

    inverse_max_iter: int = Field(default=1000, gt=0)
    inverse_lr: float = Field(default=1.0, gt=0.0)
    inverse_tolerance: float = Field(default=1e-8, gt=0.0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
