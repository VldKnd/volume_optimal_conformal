from typing import Literal

from pydantic import BaseModel


class LogProbabilityCalibratorConfig(BaseModel):
    """Configuration for conditional log-density conformal calibration."""

    type: Literal["log_probability", "log_prob", "log_p"] = "log_probability"


# Concise aliases for callers that use the model method name in class names.
LogProbCalibratorConfig = LogProbabilityCalibratorConfig
LogPCalibratorConfig = LogProbabilityCalibratorConfig
