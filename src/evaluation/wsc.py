"""Worst-slab coverage.

Adapted from the metric used by Cauchois, Gupta, and Duchi (2021) and
Romano, Sesia, and Candès (2020).
"""

import math

import numpy as np
from sklearn.model_selection import train_test_split


def _worst_slab_for_direction(
    representations: np.ndarray,
    coverages: np.ndarray,
    delta: float,
    direction: np.ndarray,
) -> tuple[float, float, float]:
    projections = representations @ direction
    order = np.argsort(projections)
    sorted_projections = projections[order]
    sorted_coverages = coverages[order]
    values, starts, counts = np.unique(
        sorted_projections,
        return_index=True,
        return_counts=True,
    )
    covered = np.add.reduceat(sorted_coverages, starts)
    cumulative_counts = np.concatenate(([0], np.cumsum(counts)))
    cumulative_covered = np.concatenate(([0.0], np.cumsum(covered)))

    minimum_count = math.ceil(delta * len(coverages))
    best_coverage = float(coverages.mean())
    best_start = 0
    best_end = len(values) - 1

    for start in range(len(values)):
        first_end = np.searchsorted(
            cumulative_counts,
            cumulative_counts[start] + minimum_count,
        )
        ends = np.arange(first_end, len(values) + 1)
        if ends.size == 0:
            continue

        interval_covered = cumulative_covered[ends] - cumulative_covered[start]
        interval_counts = cumulative_counts[ends] - cumulative_counts[start]
        interval_coverages = interval_covered / interval_counts
        local_end = int(np.argmin(interval_coverages))
        if interval_coverages[local_end] < best_coverage:
            best_coverage = float(interval_coverages[local_end])
            best_start = start
            best_end = int(ends[local_end] - 1)

    return (
        float(best_coverage),
        float(values[best_start]),
        float(values[best_end]),
    )


def wsc(
    representations: np.ndarray,
    coverages: np.ndarray,
    delta: float,
    M: int = 1000,
    random_state: int = 42,
) -> tuple[float, np.ndarray, float, float]:
    """Find the lowest-coverage slab containing at least ``delta`` mass."""
    representations = np.asarray(representations)
    coverages = np.asarray(coverages).reshape(-1)
    if representations.ndim == 1:
        representations = representations[:, None]

    if representations.ndim != 2:
        raise ValueError("representations must have shape (n, dimension).")
    if representations.shape[0] != coverages.shape[0]:
        raise ValueError("representations and coverages must have equal length.")
    if representations.shape[0] == 0:
        raise ValueError("Worst-slab coverage requires at least one observation.")
    if representations.shape[1] == 0:
        raise ValueError("Worst-slab coverage requires at least one covariate.")
    if not 0.0 < delta <= 1.0:
        raise ValueError("delta must be in (0, 1].")
    if M < 1:
        raise ValueError("M must be positive.")

    generator = np.random.default_rng(random_state)
    directions = generator.standard_normal((M, representations.shape[1]))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    zero_norm = norms[:, 0] == 0.0
    directions[zero_norm, 0] = 1.0
    norms[zero_norm] = 1.0
    directions /= norms

    best_coverage = math.inf
    best_direction = directions[0]
    best_lower = 0.0
    best_upper = 0.0

    for direction in directions:
        coverage, lower, upper = _worst_slab_for_direction(
            representations=representations,
            coverages=coverages,
            delta=delta,
            direction=direction,
        )
        if coverage < best_coverage:
            best_coverage = coverage
            best_direction = direction
            best_lower = lower
            best_upper = upper

    return best_coverage, best_direction, best_lower, best_upper


def wsc_unbiased(
    representations: np.ndarray,
    coverages: np.ndarray,
    delta: float,
    M: int = 1000,
    test_size: float = 0.75,
    random_state: int = 0,
) -> tuple[float, float]:
    """Return held-out mean and standard deviation in the selected slab."""
    (
        representations_search,
        representations_evaluation,
        coverages_search,
        coverages_evaluation,
    ) = train_test_split(
        representations,
        coverages,
        test_size=test_size,
        random_state=random_state,
    )

    _, direction, lower, upper = wsc(
        representations=representations_search,
        coverages=coverages_search,
        delta=delta,
        M=M,
        random_state=random_state,
    )
    projections = np.asarray(representations_evaluation) @ direction
    in_slab = (lower <= projections) & (projections <= upper)

    if not np.any(in_slab):
        raise ValueError(
            "The selected slab contains no observations in the evaluation split."
        )

    slab_coverages = np.asarray(coverages_evaluation)[in_slab]
    return (
        float(slab_coverages.mean()),
        float(slab_coverages.std()),
    )
