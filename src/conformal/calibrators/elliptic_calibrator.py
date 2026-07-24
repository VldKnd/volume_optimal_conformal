import math

import torch
from sklearn.neighbors import NearestNeighbors

from configs.calibrators.elliptic_calibrator import EllipticCalibratorConfig
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.quantile import conformal_quantile


class EllipticCalibrator(BaseCalibrator):
    """Conditional local Mahalanobis-score calibrator.

    The first half of the calibration sample fits a nearest-neighbor model and
    covariance estimates. The second half calibrates the scalar Mahalanobis
    scores, keeping covariance estimation separate from threshold calibration.
    """

    def __init__(self, config: EllipticCalibratorConfig):
        self.config = config
        self.threshold: torch.Tensor | None = None
        self.x_train: torch.Tensor | None = None
        self.score_train: torch.Tensor | None = None
        self.global_covariance: torch.Tensor | None = None
        self.global_inverse_covariance: torch.Tensor | None = None
        self.knn: NearestNeighbors | None = None

    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        coverage_mass: float,
    ) -> None:
        self._validate_xy(x=x, scores=scores)
        if x.shape[1] == 0:
            raise ValueError(
                "EllipticCalibrator requires at least one covariate dimension."
            )

        split = scores.shape[0] // 2
        if split <= self.config.n_neighbors:
            raise ValueError(
                "Need more covariance-fitting points than n_neighbors. "
                f"Got split={split}, n_neighbors={self.config.n_neighbors}."
            )

        x_cov = x[:split]
        score_cov = scores[:split]
        x_conformal = x[split:]
        score_conformal = scores[split:]

        self.x_train = x_cov.detach()
        self.score_train = score_cov.detach()
        self.global_covariance = self._regularized_covariance(score_cov)
        self.global_inverse_covariance = self._inverse_covariance(
            self.global_covariance
        )

        self.knn = NearestNeighbors(
            n_neighbors=self.config.n_neighbors,
            n_jobs=-1,
        )
        self.knn.fit(x_cov.detach().cpu().numpy())

        scalar_scores = self.scalar_score(x_conformal, score_conformal)
        self.threshold = conformal_quantile(
            scalar_scores,
            coverage_mass,
        ).detach()

    def scalar_score(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_xy(x=x, scores=scores)
        inverse_covariances = self._local_inverse_covariances(x).to(
            device=scores.device,
            dtype=scores.dtype,
        )

        return torch.sqrt(
            torch.einsum(
                "bi,bij,bj->b",
                scores,
                inverse_covariances,
                scores,
            ).clamp_min(0.0)
        )

    def estimate_log_volume(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log-volume of the calibrated local ellipsoid at each x."""
        if self.threshold is None:
            raise RuntimeError("EllipticCalibrator must be fitted first.")

        inverse_covariances = self._local_inverse_covariances(x).to(
            device=x.device,
            dtype=x.dtype,
        )
        sign, log_abs_determinant = torch.linalg.slogdet(inverse_covariances)
        if torch.any(sign <= 0):
            raise RuntimeError(
                "Local inverse covariance must have positive determinant."
            )

        dimension = inverse_covariances.shape[-1]
        log_unit_ball_volume = (
            0.5 * dimension * math.log(math.pi)
            - math.lgamma(0.5 * dimension + 1.0)
        )
        threshold = self.threshold.to(device=x.device, dtype=x.dtype)

        return (
            log_unit_ball_volume
            + dimension * torch.log(threshold)
            - 0.5 * log_abs_determinant
        )

    def _local_inverse_covariances(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.knn is None or self.score_train is None
            or self.global_covariance is None
        ):
            raise RuntimeError("EllipticCalibrator must be fitted first.")

        neighbor_indices = self.knn.kneighbors(
            x.detach().cpu().numpy(),
            return_distance=False,
        )
        inverse_covariances = []

        for indices in neighbor_indices:
            index = torch.as_tensor(
                indices,
                device=self.score_train.device,
                dtype=torch.long,
            )
            local_scores = self.score_train.index_select(0, index)
            local_covariance = self._regularized_covariance(local_scores)
            covariance = (
                self.config.local_weight * local_covariance +
                (1.0 - self.config.local_weight) * self.global_covariance
            )
            inverse_covariances.append(self._inverse_covariance(covariance))

        return torch.stack(inverse_covariances, dim=0)

    def _regularized_covariance(
        self,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        scores = scores.detach()

        if scores.ndim != 2:
            raise ValueError(
                "Expected scores with shape (n, dimension), "
                f"got {tuple(scores.shape)}."
            )

        if scores.shape[0] < 2:
            raise ValueError("At least two scores are required for covariance.")

        centered = scores - scores.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (scores.shape[0] - 1)
        identity = torch.eye(
            scores.shape[1],
            device=scores.device,
            dtype=scores.dtype,
        )

        return covariance + self.config.regularization * identity

    def _inverse_covariance(
        self,
        covariance: torch.Tensor,
    ) -> torch.Tensor:
        try:
            return torch.linalg.inv(covariance)
        except RuntimeError as error:
            raise RuntimeError(
                "Covariance is singular. Increase calibrator regularization."
            ) from error

    @staticmethod
    def _validate_xy(
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (n, x_dim), got {tuple(x.shape)}.")

        if scores.ndim != 2:
            raise ValueError(
                "Expected scores with shape (n, dimension), "
                f"got {tuple(scores.shape)}."
            )

        if x.shape[0] != scores.shape[0]:
            raise ValueError(
                "x and scores must have matching batch dimensions, got "
                f"{x.shape[0]} and {scores.shape[0]}."
            )

        if scores.shape[1] < 1:
            raise ValueError("Score dimension must be positive.")
