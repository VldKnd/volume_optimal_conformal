"""Backward-compatible imports for the canonical norm calibrator."""

from configs.calibrators.norm_calibrator import NormCalibratorConfig
from conformal.calibrators.norm_calibrator import NormCalibrator

__all__ = [
    "NormCalibrator",
    "NormCalibratorConfig",
]
