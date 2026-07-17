# src/configs/predictors/transport/flow_matching.py

from typing import Literal

from pydantic import BaseModel, Field


class FlowMatchingPredictorConfig(BaseModel):
    type: Literal["flow_matching"] = "flow_matching"

    x_dim: int
    y_dim: int

    hidden_dim: int = 128
    num_hidden_layers: int = 3

    ode_steps: int = Field(default=100, gt=0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    use_hutchinson_trace_estimator: bool = False
    hutchinson_num_samples: int = Field(default=1, gt=0)
    hutchinson_noise: Literal["rademacher", "gaussian"] = "rademacher"
