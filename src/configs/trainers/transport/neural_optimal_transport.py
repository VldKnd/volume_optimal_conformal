# src/configs/trainers/neural_optimal_transport.py

from pydantic import BaseModel, Field


class NeuralOptimalTransportTrainerConfig(BaseModel):
    epochs: int = 100
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)

    warmup_iterations: int = 1
    grad_clip_norm: float = 1.0

    use_cosine_scheduler: bool = True
    verbose: bool = True