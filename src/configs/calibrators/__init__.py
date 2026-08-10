from typing import Annotated

from pydantic import Field

from configs.calibrators.cdf_calibrator import CDFCalibratorConfig
from configs.calibrators.elliptic_calibrator import EllipticCalibratorConfig
from configs.calibrators.log_probability_calibrator import (
    LogPCalibratorConfig,
    LogProbCalibratorConfig,
    LogProbabilityCalibratorConfig,
)
from configs.calibrators.no_calibrator import NoCalibratorConfig
from configs.calibrators.norm_calibrator import NormCalibratorConfig
from configs.calibrators.optimal_transport_calibrator import (
    GlobalOTCPCalibratorConfig,
    LocalOTCPCalibratorConfig,
)

CalibratorConfig = Annotated[
    CDFCalibratorConfig
    | EllipticCalibratorConfig
    | GlobalOTCPCalibratorConfig
    | LogProbabilityCalibratorConfig
    | LocalOTCPCalibratorConfig
    | NormCalibratorConfig
    | NoCalibratorConfig,
    Field(discriminator="type"),
]

__all__ = [
    "CalibratorConfig",
    "CDFCalibratorConfig",
    "EllipticCalibratorConfig",
    "GlobalOTCPCalibratorConfig",
    "LogPCalibratorConfig",
    "LogProbCalibratorConfig",
    "LogProbabilityCalibratorConfig",
    "LocalOTCPCalibratorConfig",
    "NoCalibratorConfig",
    "NormCalibratorConfig",
]
