import math

import torch


def conformal_quantile(
    scores: torch.Tensor,
    coverage_mass: float,
) -> torch.Tensor:
    """Return the split-conformal order statistic for ``coverage_mass``.

    The selected one-indexed rank is
    ``ceil((n + 1) * coverage_mass)`` in the calibration scores augmented with
    ``+inf``. The infinite threshold is returned when the requested
    finite-sample coverage is unattainable with ``n`` calibration points.
    Selecting the order statistic directly avoids the interpolation performed
    by :func:`torch.quantile`.
    """
    if scores.ndim != 1:
        raise ValueError(f"Expected scores with shape (n,), got {scores.shape}.")

    if scores.numel() == 0:
        raise ValueError("At least one calibration score is required.")

    if not 0.0 < coverage_mass < 1.0:
        raise ValueError("coverage_mass must be in (0, 1), "
                         f"got {coverage_mass}.")

    if not torch.isfinite(scores).all():
        raise ValueError("Calibration scores must contain only finite values.")

    rank = math.ceil((scores.shape[0] + 1) * coverage_mass)
    if rank > scores.shape[0]:
        dtype = scores.dtype if scores.is_floating_point() else torch.get_default_dtype(
        )
        return torch.tensor(
            torch.inf,
            device=scores.device,
            dtype=dtype,
        )

    return scores.kthvalue(rank).values
