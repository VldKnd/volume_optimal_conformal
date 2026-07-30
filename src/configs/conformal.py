from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, Field

from configs.calibrators import (
    CalibratorConfig,
    EllipticCalibratorConfig,
    GlobalOTCPCalibratorConfig,
    LocalOTCPCalibratorConfig,
    NormCalibratorConfig,
)

ResidualCalibratorConfig = Annotated[
    EllipticCalibratorConfig
    | GlobalOTCPCalibratorConfig
    | LocalOTCPCalibratorConfig
    | NormCalibratorConfig,
    Field(discriminator="type"),
]


class TransportBasedConformalPredictorConfig(BaseModel):
    """Configuration for a calibrated wrapper around a transport predictor.

    ``coverage_mass`` is the desired probability mass of the prediction
    region and is used consistently by the predictor and calibrator.
    """

    type: Literal["transport_based"] = "transport_based"

    coverage_mass: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        validation_alias=AliasChoices("coverage_mass", "coverage"),
    )
    calibrator: CalibratorConfig = Field(
        default_factory=NormCalibratorConfig,
        validation_alias=AliasChoices("calibrator", "calibrator_config"),
    )

    volume_mc_samples: int = Field(default=10_000, gt=0)
    volume_batch_size: int = Field(
        default=1_024,
        gt=0,
        description=(
            "Maximum number of flattened covariate/latent pairs passed to "
            "predictor.log_det in one call."
        ),
    )
    volume_seed: int = 0


class ResidualConformalPredictorConfig(BaseModel):
    """Configuration for conformalizing regression residuals."""

    type: Literal["residual"] = "residual"

    coverage_mass: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        validation_alias=AliasChoices("coverage_mass", "coverage"),
    )
    calibrator: ResidualCalibratorConfig = Field(
        default_factory=NormCalibratorConfig,
        validation_alias=AliasChoices("calibrator", "calibrator_config"),
    )
    volume_mc_samples: int = Field(default=10_000, gt=0)
    volume_n_neighbors: int = Field(default=100, gt=1)
    volume_seed: int = 0
