from typing import Annotated

from pydantic import Field

from configs.calibrators.elliptic_calibrator import EllipticCalibratorConfig
from configs.calibrators.no_calibrator import NoCalibratorConfig
from configs.calibrators.norm_calibrator import NormCalibratorConfig

CalibratorConfig = Annotated[
    EllipticCalibratorConfig | NormCalibratorConfig | NoCalibratorConfig,
    Field(discriminator="type"),
]

__all__ = [
    "CalibratorConfig",
    "EllipticCalibratorConfig",
    "NoCalibratorConfig",
    "NormCalibratorConfig",
]
