# src/calibrators/norm.py

from typing import Literal

import torch
from pydantic import BaseModel, Field

from calibrators.base import BaseCalibrator
from calibrators.quantile import conformal_quantile


class NormCalibratorConfig(BaseModel):
    type: Literal["norm"] = "norm"
    p: float = Field(default=2.0, gt=0.0)


class NormCalibrator(BaseCalibrator):
    def __init__(self, config: NormCalibratorConfig):
        self.config = config
        self.threshold: torch.Tensor | None = None

    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        alpha: float,
    ) -> None:
        scalar_scores = self.scalar_score(x, scores)
        self.threshold = conformal_quantile(scalar_scores, alpha)

    def scalar_score(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        return scores.norm(p=self.config.p, dim=-1)