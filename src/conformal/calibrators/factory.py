from configs.calibrators import (
    CDFCalibratorConfig,
    CalibratorConfig,
    EllipticCalibratorConfig,
    GlobalOTCPCalibratorConfig,
    LogProbabilityCalibratorConfig,
    LocalOTCPCalibratorConfig,
    NoCalibratorConfig,
    NormCalibratorConfig,
)
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.cdf_calibrator import CDFCalibrator
from conformal.calibrators.elliptic_calibrator import EllipticCalibrator
from conformal.calibrators.log_probability_calibrator import (
    LogProbabilityCalibrator,
)
from conformal.calibrators.no_calibrator import NoCalibrator
from conformal.calibrators.norm_calibrator import NormCalibrator
from conformal.calibrators.optimal_transport_calibrator import (
    GlobalOTCPCalibrator,
    LocalOTCPCalibrator,
)


def make_calibrator(config: CalibratorConfig) -> BaseCalibrator:
    """Construct the calibrator selected by a validated config."""
    if isinstance(config, CDFCalibratorConfig):
        return CDFCalibrator(config)

    if isinstance(config, NormCalibratorConfig):
        return NormCalibrator(config)

    if isinstance(config, EllipticCalibratorConfig):
        return EllipticCalibrator(config)

    if isinstance(config, GlobalOTCPCalibratorConfig):
        return GlobalOTCPCalibrator(config)

    if isinstance(config, LogProbabilityCalibratorConfig):
        return LogProbabilityCalibrator(config)

    if isinstance(config, LocalOTCPCalibratorConfig):
        return LocalOTCPCalibrator(config)

    if isinstance(config, NoCalibratorConfig):
        return NoCalibrator(config)

    raise TypeError(
        "Unsupported calibrator config type "
        f"{type(config).__name__}. Expected CDFCalibratorConfig, "
        "NormCalibratorConfig, EllipticCalibratorConfig, "
        "GlobalOTCPCalibratorConfig, "
        "LogProbabilityCalibratorConfig, LocalOTCPCalibratorConfig, or "
        "NoCalibratorConfig."
    )


create_calibrator = make_calibrator
