# src/configs/trainers/flow_matching.py

from pydantic import BaseModel, Field

class FlowMatchingTrainerConfig(BaseModel):
    epochs: int = 100

    batch_size: int = 256

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)

    grad_clip_norm: float = 1.0

    use_cosine_scheduler: bool = True

    num_workers: int = 0
    pin_memory: bool = False

    verbose: bool = True