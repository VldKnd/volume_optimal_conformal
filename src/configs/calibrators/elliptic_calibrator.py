# src/configs/calibrators/elliptic_calibrator.py

from typing import Literal

from pydantic import BaseModel, Field


class EllipticCalibratorConfig(BaseModel):
    type: Literal["elliptic"] = "elliptic"

    n_neighbors: int = Field(default=100, gt=1)
    regularization: float = Field(default=1e-4, ge=0.0)
    local_weight: float = Field(default=0.8, ge=0.0, le=1.0)
