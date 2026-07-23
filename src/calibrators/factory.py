"""Backward-compatible imports for the canonical calibrator factory."""

from conformal.calibrators.factory import create_calibrator, make_calibrator

__all__ = [
    "create_calibrator",
    "make_calibrator",
]
