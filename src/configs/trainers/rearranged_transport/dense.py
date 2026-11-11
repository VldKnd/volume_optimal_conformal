# src/configs/trainers/rearranged_transport/dense.py

from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class RearrangedTransportTrainerConfig(BaseModel):
    epochs: int = Field(default=100, gt=0)

    mc_samples_per_x: int = Field(default=1, gt=0)
    train_transport_map: bool = False
    # Keyword arguments forwarded to the wrapped transport predictor's
    # ``log_det`` during rearrangement training only, for example
    # ``{"ode_steps": 20, "use_hutchinson_trace_estimator": true}`` for a
    # flow matching base; evaluation keeps the predictor's configured solve.
    transport_log_det_options: dict[str, Any] = Field(default_factory=dict)

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    grad_clip_norm: float = Field(default=1.0, gt=0.0)

    use_cosine_scheduler: bool = True
    verbose: bool = True


class SupervisedRearrangedTransportTrainerConfig(RearrangedTransportTrainerConfig):
    coverage_mass: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        validation_alias=AliasChoices("coverage_mass", "alpha"),
    )
