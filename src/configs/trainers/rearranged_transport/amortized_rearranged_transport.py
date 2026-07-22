from pydantic import BaseModel, Field


class AmortizedRearrangedTransportTrainerConfig(BaseModel):
    """Training configuration for all-coverage rearrangement training."""

    epochs: int = Field(default=100, gt=0)

    mc_samples_per_x: int = Field(default=1, gt=0)
    train_transport_map: bool = False

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    grad_clip_norm: float = Field(default=1.0, gt=0.0)

    use_cosine_scheduler: bool = True
    verbose: bool = True
