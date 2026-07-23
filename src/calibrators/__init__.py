"""Backward-compatible aliases for :mod:`conformal.calibrators`."""

from conformal.calibrators import (
    BaseCalibrator,
    CalibratorConfig,
    EllipticCalibrator,
    EllipticCalibratorConfig,
    NoCalibrator,
    NoCalibratorConfig,
    NormCalibrator,
    NormCalibratorConfig,
    conformal_quantile,
    create_calibrator,
    make_calibrator,
)

__all__ = [
    "BaseCalibrator",
    "CalibratorConfig",
    "EllipticCalibrator",
    "EllipticCalibratorConfig",
    "NoCalibrator",
    "NoCalibratorConfig",
    "NormCalibrator",
    "NormCalibratorConfig",
    "conformal_quantile",
    "create_calibrator",
    "make_calibrator",
]
