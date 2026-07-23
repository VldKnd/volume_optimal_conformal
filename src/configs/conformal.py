from pydantic import AliasChoices, BaseModel, Field

from configs.calibrators import CalibratorConfig, NormCalibratorConfig


class ConformalPredictorConfig(BaseModel):
    """Configuration for a calibrated wrapper around a transport predictor.

    ``coverage_mass`` is the desired probability mass of the prediction
    region and is used consistently by the predictor and calibrator.
    """

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
