from typing import Literal

from pydantic import BaseModel, Field


class RandomForestPredictorConfig(BaseModel):
    type: Literal["random_forest"] = "random_forest"

    x_dim: int = Field(gt=0)
    y_dim: int = Field(gt=0)

    n_estimators: int = Field(default=100, gt=0)
    max_depth: int | None = Field(default=None, gt=0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"


class MLPPredictorConfig(BaseModel):
    type: Literal["mlp"] = "mlp"

    x_dim: int = Field(gt=0)
    y_dim: int = Field(gt=0)

    hidden_dim: int = Field(default=128, gt=0)
    num_hidden_layers: int = Field(default=2, gt=0)

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"


class NearestNeighborsPredictorConfig(BaseModel):
    type: Literal["nearest_neighbors"] = "nearest_neighbors"

    x_dim: int = Field(gt=0)
    y_dim: int = Field(gt=0)

    n_neighbors: int = Field(default=5, gt=0)

    device: str = "cpu"
    dtype: str = "float32"
