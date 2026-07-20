# src/configs/predictors/transport/normalizing_flow.py

from typing import Literal

from pydantic import BaseModel, Field


class NormalizingFlowPredictorConfig(BaseModel):
    type: Literal["normalizing_flow"] = "normalizing_flow"

    x_dim: int = Field(ge=0)
    y_dim: int = Field(gt=0)

    hidden_dim: int = Field(default=128, gt=0)
    num_hidden_layers: int = Field(default=3, ge=0)
    num_flow_layers: int = Field(default=6, gt=0)
    log_scale_bound: float = Field(default=3.0, gt=0.0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
