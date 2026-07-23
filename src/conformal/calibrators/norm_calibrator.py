import torch

from configs.calibrators.norm_calibrator import NormCalibratorConfig
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.quantile import conformal_quantile


class NormCalibrator(BaseCalibrator):
    """Calibrate the L-p norm of a multivariate score."""

    def __init__(self, config: NormCalibratorConfig):
        self.config = config
        self.threshold: torch.Tensor | None = None

    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        coverage_mass: float,
    ) -> None:
        scalar_scores = self.scalar_score(x, scores)
        self.threshold = conformal_quantile(
            scalar_scores,
            coverage_mass,
        ).detach()

    def scalar_score(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        del x

        if scores.ndim < 1:
            raise ValueError("scores must have at least one dimension.")

        return scores.norm(p=self.config.p, dim=-1)
