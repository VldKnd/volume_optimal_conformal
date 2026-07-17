# src/configs/predictors/transport/neural_optimal_transport.py

from typing import Literal

from pydantic import BaseModel


class NeuralOptimalTransportPredictorConfig(BaseModel):
    type: Literal["neural_optimal_transport"] = "neural_optimal_transport"

    x_dim: int
    y_dim: int

    hidden_dim: int = 128
    num_hidden_layers: int = 3

    potential_type: Literal["u", "y"] = "u"

    c_transform_lr: float = 0.25
    c_transform_max_iter: int = 100

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"