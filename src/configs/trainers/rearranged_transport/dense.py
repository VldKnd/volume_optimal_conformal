# src/configs/trainers/rearranged_transport/dense.py

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator


class RearrangedTransportTrainerConfig(BaseModel):
    epochs: int = Field(default=100, gt=0)

    mc_samples_per_x: int = Field(default=1, gt=0)
    train_transport_map: bool = False
    # Every covariate of every batch gets a coverage mass drawn uniformly
    # from this interval, and its latents are the standard Gaussian
    # restricted to the ball of that mass. The default trains the
    # rearrangement across all levels; a degenerate interval such as
    # ``(0.9, 0.9)`` trains it at that single level only, and a narrow one
    # such as ``(0.85, 0.95)`` at a band around it.
    coverage_mass_range: tuple[float, float] = (0.0, 1.0)
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

    @field_validator("coverage_mass_range")
    @classmethod
    def _validate_coverage_mass_range(
        cls,
        value: tuple[float, float],
    ) -> tuple[float, float]:
        low, high = value
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                "coverage_mass_range must satisfy 0 <= low <= high <= 1, "
                f"got {value!r}."
            )
        return (float(low), float(high))


class SupervisedRearrangedTransportTrainerConfig(RearrangedTransportTrainerConfig):
    coverage_mass: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        validation_alias=AliasChoices("coverage_mass", "alpha"),
    )
