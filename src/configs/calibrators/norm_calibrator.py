# src/configs/calibrators/norm_calibrator.py

from typing import Literal

from pydantic import BaseModel, Field

class NormCalibratorConfig(BaseModel):
    type: Literal["norm"] = "norm"
    norm: float = 2.0
