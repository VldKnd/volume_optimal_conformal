import math

import torch
from scipy.stats import chi

from configs.calibrators.no_calibrator import NoCalibratorConfig
from conformal.calibrators.base import BaseCalibrator


class NoCalibrator(BaseCalibrator):
    """Analytic standard-Gaussian baseline without empirical calibration.

    The multivariate score is reduced to its Euclidean norm and the threshold
    is the corresponding Chi-distribution quantile at ``coverage_mass``.
    """

    def __init__(self, config: NoCalibratorConfig):
        self.config = config
        self.threshold: torch.Tensor | None = None

    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        coverage_mass: float,
    ) -> None:
        del x

        if scores.ndim != 2:
            raise ValueError(
                "Expected scores with shape (n, dimension), "
                f"got {tuple(scores.shape)}."
            )

        dimension = scores.shape[-1]
        if dimension < 1:
            raise ValueError("Score dimension must be positive.")

        if not 0.0 < coverage_mass < 1.0:
            raise ValueError(
                "coverage_mass must be in (0, 1), "
                f"got {coverage_mass}."
            )

        threshold = float(chi.ppf(coverage_mass, df=dimension))
        if not math.isfinite(threshold):
            raise RuntimeError("The analytic Chi threshold is not finite.")

        dtype = scores.dtype if scores.is_floating_point() else torch.get_default_dtype(
        )
        self.threshold = torch.tensor(
            threshold,
            device=scores.device,
            dtype=dtype,
        )

    def scalar_score(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        del x

        if scores.ndim < 1:
            raise ValueError("scores must have at least one dimension.")

        return scores.norm(p=2, dim=-1)
