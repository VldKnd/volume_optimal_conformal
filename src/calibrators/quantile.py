# src/calibrators/quantile.py

import math
import torch


def conformal_quantile(
    scores: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected scores with shape (n,), got {scores.shape}.")

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    n = scores.shape[0]
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)

    return torch.quantile(scores, level)