"""Backward-compatible imports for the canonical analytic calibrator."""

from configs.calibrators.no_calibrator import NoCalibratorConfig
from conformal.calibrators.no_calibrator import NoCalibrator

__all__ = [
    "NoCalibrator",
    "NoCalibratorConfig",
]
