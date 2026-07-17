# src/configs/predictors/transport/neural_spline_flow.py

from typing import Literal

from pydantic import BaseModel, Field


class NeuralSplineFlowPredictorConfig(BaseModel):
    type: Literal["neural_spline_flow"] = "neural_spline_flow"

    x_dim: int
    y_dim: int

    hidden_dim: int = Field(default=128, gt=0)
    num_hidden_layers: int = Field(default=3, ge=0)

    num_flow_layers: int = Field(default=6, gt=0)
    num_bins: int = Field(default=16, ge=2)
    tail_bound: float = Field(default=4.0, gt=0.0)

    min_bin_width: float = Field(default=1e-3, gt=0.0)
    min_bin_height: float = Field(default=1e-3, gt=0.0)
    min_derivative: float = Field(default=1e-3, gt=0.0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
