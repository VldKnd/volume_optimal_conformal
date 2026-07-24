from configs.calibrators import (
    CalibratorConfig,
    EllipticCalibratorConfig,
    GlobalOTCPCalibratorConfig,
    LocalOTCPCalibratorConfig,
    NoCalibratorConfig,
    NormCalibratorConfig,
)
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.elliptic_calibrator import EllipticCalibrator
from conformal.calibrators.no_calibrator import NoCalibrator
from conformal.calibrators.norm_calibrator import NormCalibrator
from conformal.calibrators.optimal_transport_calibrator import (
    GlobalOTCPCalibrator,
    LocalOTCPCalibrator,
)


def make_calibrator(config: CalibratorConfig) -> BaseCalibrator:
    """Construct the calibrator selected by a validated config."""
    if isinstance(config, NormCalibratorConfig):
        return NormCalibrator(config)

    if isinstance(config, EllipticCalibratorConfig):
        return EllipticCalibrator(config)

    if isinstance(config, GlobalOTCPCalibratorConfig):
        return GlobalOTCPCalibrator(config)

    if isinstance(config, LocalOTCPCalibratorConfig):
        return LocalOTCPCalibrator(config)

    if isinstance(config, NoCalibratorConfig):
        return NoCalibrator(config)

    raise TypeError(
        "Unsupported calibrator config type "
        f"{type(config).__name__}. Expected NormCalibratorConfig, "
        "EllipticCalibratorConfig, GlobalOTCPCalibratorConfig, "
        "LocalOTCPCalibratorConfig, or NoCalibratorConfig."
    )


create_calibrator = make_calibrator
