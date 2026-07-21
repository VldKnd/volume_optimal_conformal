# src/configs/predictors/rearranged_transport/dense.py

from typing import Literal

from pydantic import BaseModel, Field


class RearrangedTransportPredictorConfig(BaseModel):
    type: Literal[
        "rearranged_transport",
        "dense_rearranged_transport",
    ] = "rearranged_transport"

    x_dim: int
    y_dim: int

    hidden_dimension: int = Field(default=64, gt=0)
    number_of_hidden_layers: int = Field(default=2, ge=0)
    time_dependent: bool = True
    vector_field_implementation: Literal[
        "standard",
        "experimental",
        "sparse",
    ] = "standard"
    activation: Literal[
        "elu",
        "gelu",
        "leaky_relu",
        "prelu",
        "relu",
        "silu",
        "softplus",
        "tanh",
    ] = "softplus"
    activation_power: float = Field(default=2.0, gt=0.0)

    use_adjoint: bool = False
    method: str = "rk4"
    rtol: float = Field(default=1e-5, gt=0.0)
    atol: float = Field(default=1e-6, gt=0.0)
    number_of_steps: int | None = Field(default=16, gt=0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"


DenseRearrangedTransportPredictorConfig = RearrangedTransportPredictorConfig
