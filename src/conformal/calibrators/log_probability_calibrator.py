"""Conformal calibration based on a model's conditional log density."""

import torch

from configs.calibrators.log_probability_calibrator import (
    LogProbabilityCalibratorConfig,
)
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.quantile import conformal_quantile


class LogProbabilityCalibrator(BaseCalibrator):
    """Accept observations whose log density exceeds a calibrated cutoff.

    The transport conformal predictor supplies values of ``log p(y | x)`` as
    one-column scores. Calibration applies the usual split-conformal quantile
    to their negatives, then stores the equivalent lower log-probability
    cutoff so that membership is expressed directly as ``log_prob >= cutoff``.
    """

    def __init__(self, config: LogProbabilityCalibratorConfig):
        self.config = config
        self.threshold: torch.Tensor | None = None

    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        coverage_mass: float,
    ) -> None:
        log_probabilities = self.scalar_score(x=x, scores=scores)
        self.threshold = (-conformal_quantile(
            -log_probabilities,
            coverage_mass,
        )).detach()

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
            "Log-probability scores must have shape (n,) or (n, 1), "
            f"got {tuple(scores.shape)}."
        )

    def contains(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError(
                "LogProbabilityCalibrator must be fitted before contains()."
            )

        log_probabilities = self.scalar_score(x=x, scores=scores)
        threshold = self.threshold.to(
            device=log_probabilities.device,
            dtype=log_probabilities.dtype,
        )
        return log_probabilities >= threshold


LogProbCalibrator = LogProbabilityCalibrator
LogPCalibrator = LogProbabilityCalibrator
