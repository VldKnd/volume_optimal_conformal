# src/calibrators/elliptic.py

from typing import Literal

import torch
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors

from calibrators.base import BaseCalibrator
from calibrators.quantile import conformal_quantile
from configs.calibrators.elliptic_calibrator import EllipticCalibratorConfig


class EllipticCalibrator(BaseCalibrator):
    """
    Conditional/local elliptic calibrator.

    First half of calibration data:
        fit kNN over x and estimate global covariance of multivariate scores.

    Second half:
        compute local Mahalanobis scalar scores and conformal threshold.

    At test time:
        use kNN in x-space to estimate local covariance for each x.
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
        alpha: float,
    ) -> None:
        n = scores.shape[0]
        split = n // 2

        x_cov = x[:split]
        z_cov = scores[:split]

        x_conf = x[split:]
        z_conf = scores[split:]

        if split <= self.config.n_neighbors:
            raise ValueError(
                f"Need more covariance-fitting points than n_neighbors. "
                f"Got split={split}, n_neighbors={self.config.n_neighbors}."
            )

        self.x_train = x_cov.detach()
        self.score_train = z_cov.detach()

        self.global_covariance = self._regularized_covariance(z_cov)
        self.global_inverse_covariance = torch.linalg.inv(self.global_covariance)

        self.knn = NearestNeighbors(
            n_neighbors=self.config.n_neighbors,
            n_jobs=-1,
        )
        self.knn.fit(x_cov.detach().cpu().numpy())

        scalar_scores = self.scalar_score(x_conf, z_conf)
        self.threshold = conformal_quantile(scalar_scores, alpha)

    def scalar_score(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        inverse_covariances = self._local_inverse_covariances(x)

        return torch.sqrt(
            torch.einsum(
                "bi,bij,bj->b",
                scores,
                inverse_covariances,
                scores,
            ).clamp_min(0.0)
        )

    def _local_inverse_covariances(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.knn is None or self.score_train is None or self.global_covariance is None:
            raise RuntimeError("EllipticCalibrator must be fitted first.")

        device = self.score_train.device
        dtype = self.score_train.dtype

        x_np = x.detach().cpu().numpy()
        neighbor_idx = self.knn.kneighbors(x_np, return_distance=False)

        inv_covs = []

        for idx in neighbor_idx:
            local_scores = self.score_train[idx]
            local_cov = self._regularized_covariance(local_scores)

            cov = (
                self.config.local_weight * local_cov
                + (1.0 - self.config.local_weight) * self.global_covariance
            )

            inv_covs.append(torch.linalg.inv(cov))

        return torch.stack(inv_covs, dim=0).to(device=device, dtype=dtype)

    def _regularized_covariance(
        self,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        scores = scores.detach()

        cov = torch.cov(scores.T)

        eye = torch.eye(
            cov.shape[0],
            device=scores.device,
            dtype=scores.dtype,
        )

        return cov + self.config.regularization * eye