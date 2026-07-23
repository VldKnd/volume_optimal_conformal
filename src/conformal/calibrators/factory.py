from configs.calibrators import (
    CalibratorConfig,
    EllipticCalibratorConfig,
    NoCalibratorConfig,
    NormCalibratorConfig,
)
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.elliptic_calibrator import EllipticCalibrator
from conformal.calibrators.no_calibrator import NoCalibrator
from conformal.calibrators.norm_calibrator import NormCalibrator


def make_calibrator(config: CalibratorConfig) -> BaseCalibrator:
    """Construct the calibrator selected by a validated config."""
    if isinstance(config, NormCalibratorConfig):
        return NormCalibrator(config)

    if isinstance(config, EllipticCalibratorConfig):
        return EllipticCalibrator(config)

    if isinstance(config, NoCalibratorConfig):
        return NoCalibrator(config)

    raise TypeError(
        "Unsupported calibrator config type "
        f"{type(config).__name__}. Expected NormCalibratorConfig, "
        "EllipticCalibratorConfig, or NoCalibratorConfig."
    )


create_calibrator = make_calibrator
