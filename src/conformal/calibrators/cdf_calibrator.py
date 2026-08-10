"""Conformal calibration of Monte Carlo conditional-density ranks."""

import torch

from configs.calibrators.cdf_calibrator import CDFCalibratorConfig
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.quantile import conformal_quantile


class CDFCalibrator(BaseCalibrator):
    """Calibrate scalar empirical density-CDF scores.

    The transport conformal predictor computes

    ``mean(log p(y | x) <= log p(Y_i | x))``

    over draws ``Y_i`` from the fitted conditional model. Smaller values are
    more conforming, so the default ``BaseCalibrator.contains`` comparison is
    the desired membership rule.
    """

    def __init__(self, config: CDFCalibratorConfig):
        self.config = config
        self.threshold: torch.Tensor | None = None

    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        coverage_mass: float,
    ) -> None:
        scalar_scores = self.scalar_score(x=x, scores=scores)
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

        if scores.ndim == 1:
            return scores

        if scores.ndim == 2 and scores.shape[1] == 1:
            return scores[:, 0]

        raise ValueError(
            "CDF scores must have shape (n,) or (n, 1), "
            f"got {tuple(scores.shape)}."
        )
