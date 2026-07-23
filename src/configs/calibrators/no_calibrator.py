from typing import Literal

from pydantic import BaseModel


class NoCalibratorConfig(BaseModel):
    """Configuration for the analytic standard-Gaussian baseline."""

    type: Literal["no_calibrator", "none"] = "no_calibrator"
