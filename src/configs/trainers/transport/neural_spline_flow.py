# src/configs/trainers/transport/neural_spline_flow.py

from pydantic import BaseModel, Field


class NeuralSplineFlowTrainerConfig(BaseModel):
    epochs: int = Field(default=100, gt=0)

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)

    grad_clip_norm: float = Field(default=1.0, gt=0.0)

    use_cosine_scheduler: bool = True
    verbose: bool = True
