"""Optimal-transport conformal calibrators.

The scalar score is the norm of the empirical center-outward rank from
Thurin, Nadjahi, and Boyer's Optimal Transport-based Conformal Prediction.
"""

import numpy as np
import ot
import torch
from scipy.stats import norm, qmc
from sklearn.neighbors import NearestNeighbors

from configs.calibrators.optimal_transport_calibrator import (
    GlobalOTCPCalibratorConfig,
    LocalOTCPCalibratorConfig,
)
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.quantile import conformal_quantile


def _reference_grid(
    size: int,
    dimension: int,
    seed: int,
) -> np.ndarray:
    sampler = qmc.Halton(
        d=dimension,
        rng=np.random.default_rng(seed),
    )
    uniform = sampler.random(n=size + 1)[1:]
    directions = norm.ppf(uniform)
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)

    zero_directions = direction_norms[:, 0] == 0.0
    directions[zero_directions, 0] = 1.0
    direction_norms[zero_directions] = 1.0

    radii = np.linspace(0.0, 1.0, size)
    return radii[:, None] * directions / direction_norms


def _rank_potential(
    reference: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    cost = ot.dist(scores, reference) / 2.0
    _, reference_potential = ot.solve(cost).potentials
    return 0.5 * np.square(reference).sum(axis=1) - reference_potential


def _rank(
    scores: np.ndarray,
    reference: np.ndarray,
    potential: np.ndarray,
) -> np.ndarray:
    cells = scores @ reference.T - potential[None, :]
    return reference[np.argmax(cells, axis=1)]


def _validate_inputs(
    x: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    if x.ndim != 2 or scores.ndim != 2:
        raise ValueError("x and scores must both have two dimensions.")
    if x.shape[0] != scores.shape[0]:
        raise ValueError("x and scores must have the same number of rows.")
    if scores.shape[1] == 0:
        raise ValueError("Score dimension must be positive.")


class GlobalOTCPCalibrator(BaseCalibrator):
    """OTCP with one unconditional empirical rank map."""

    def __init__(self, config: GlobalOTCPCalibratorConfig):
        self.config = config
        self.threshold: torch.Tensor | None = None
        self.reference: torch.Tensor | None = None
        self.potential: torch.Tensor | None = None

    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        coverage_mass: float,
    ) -> None:
        _validate_inputs(x, scores)
        split = scores.shape[0] // 2
        if split == 0 or split == scores.shape[0]:
            raise ValueError("Global OTCP requires at least two calibration scores.")

        ot_scores = scores[:split].detach().to(
            device="cpu",
            dtype=torch.float64,
        ).numpy()
        reference = _reference_grid(
            size=split,
            dimension=scores.shape[1],
            seed=self.config.seed,
        )
        potential = _rank_potential(reference, ot_scores)

        self.reference = torch.from_numpy(reference)
        self.potential = torch.from_numpy(potential)
        scalar_scores = self.scalar_score(
            x=x[split:],
            scores=scores[split:],
        )
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
        if self.reference is None or self.potential is None:
            raise RuntimeError("GlobalOTCPCalibrator must be fitted first.")

        reference = self.reference.to(
            device=scores.device,
            dtype=scores.dtype,
        )
        potential = self.potential.to(
            device=scores.device,
            dtype=scores.dtype,
        )
        cells = scores @ reference.T - potential
        rank_indices = cells.argmax(dim=1)
        return reference.index_select(0, rank_indices).norm(dim=1)


class LocalOTCPCalibrator(BaseCalibrator):
    """OTCP using a fresh rank map over neighboring calibration scores."""

    def __init__(self, config: LocalOTCPCalibratorConfig):
        self.config = config
        self.threshold: torch.Tensor | None = None
        self.neighbors: NearestNeighbors | None = None
        self.training_scores: np.ndarray | None = None
        self.reference: np.ndarray | None = None

    def fit(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        coverage_mass: float,
    ) -> None:
        _validate_inputs(x, scores)
        if x.shape[1] == 0:
            raise ValueError("Local OTCP requires at least one covariate.")

        split = scores.shape[0] // 2
        if split < self.config.n_neighbors:
            raise ValueError(
                "The first calibration half must contain at least "
                f"{self.config.n_neighbors} observations."
            )

        x_train = x[:split].detach().to(
            device="cpu",
            dtype=torch.float64,
        ).numpy()
        self.training_scores = scores[:split].detach().to(
            device="cpu",
            dtype=torch.float64,
        ).numpy()
        self.reference = _reference_grid(
            size=self.config.n_neighbors,
            dimension=scores.shape[1],
            seed=self.config.seed,
        )
        self.neighbors = NearestNeighbors(
            n_neighbors=self.config.n_neighbors,
        )
        self.neighbors.fit(x_train)

        scalar_scores = self.scalar_score(
            x=x[split:],
            scores=scores[split:],
        )
        self.threshold = conformal_quantile(
            scalar_scores,
            coverage_mass,
        ).detach()

    def scalar_score(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(x, scores)
        if (
            self.neighbors is None or self.training_scores is None
            or self.reference is None
        ):
            raise RuntimeError("LocalOTCPCalibrator must be fitted first.")

        x_numpy = x.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).numpy()
        scores_numpy = scores.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).numpy()
        neighbor_indices = self.neighbors.kneighbors(
            x_numpy,
            return_distance=False,
        )

        scalar_scores = []
        for score, indices in zip(
            scores_numpy,
            neighbor_indices,
            strict=True,
        ):
            local_scores = self.training_scores[indices]
            potential = _rank_potential(self.reference, local_scores)
            rank = _rank(
                score[None, :],
                self.reference,
                potential,
            )[0]
            scalar_scores.append(np.linalg.norm(rank))

        return torch.tensor(
            scalar_scores,
            device=scores.device,
            dtype=scores.dtype,
        )
