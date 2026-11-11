# src/configs/trainers/neural_optimal_transport.py

from pydantic import BaseModel, Field


class NeuralOptimalTransportTrainerConfig(BaseModel):
    epochs: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)

    warmup_iterations: int = 1
    grad_clip_norm: float = 1.0

    # Training-time Gaussian noise regularisation; 0 disables it. Covariate
    # noise is in the units of the covariates (the real datasets standardise
    # them), target noise in units of each target's standard deviation over
    # the training loader. Nothing is added at calibration or test time.
    covariate_noise_std: float = Field(default=0.0, ge=0.0)
    target_noise_std: float = Field(default=0.0, ge=0.0)

    use_cosine_scheduler: bool = True
    verbose: bool = True
