from pydantic import BaseModel, Field


class ConvexPotentialFlowTrainerConfig(BaseModel):
    epochs: int = Field(default=100, gt=0)

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)

    grad_clip_norm: float = Field(default=10.0, gt=0.0)

    use_cosine_scheduler: bool = True
    verbose: bool = True
