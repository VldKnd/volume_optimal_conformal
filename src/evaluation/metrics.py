import torch
from torch.utils.data import DataLoader

from conformal.base import ConformalPredictor
from evaluation.wsc import wsc_unbiased


def log_volume(
    dataloader: DataLoader,
    conformal_predictor: ConformalPredictor,
) -> tuple[float, float]:
    """Return the mean and population standard deviation of log-volume."""
    log_volumes = []

    for x_batch, _ in dataloader:
        with torch.no_grad():
            batch_log_volumes = conformal_predictor.estimate_log_volume(x_batch)

        batch_log_volumes = batch_log_volumes.detach().reshape(-1).to(
            device="cpu",
            dtype=torch.float64,
        )
        if batch_log_volumes.numel() != x_batch.shape[0]:
            raise ValueError(
                "estimate_log_volume must return one value per observation."
            )
        log_volumes.append(batch_log_volumes)

    if not log_volumes:
        raise ValueError("Validation dataloader must not be empty.")

    log_volumes = torch.cat(log_volumes)
    return (
        float(log_volumes.mean()),
        float(log_volumes.std(unbiased=False)),
    )


def marginal_coverage(
    dataloader: DataLoader,
    conformal_predictor: ConformalPredictor,
) -> tuple[float, None]:
    """Return empirical marginal coverage and no dispersion estimate."""
    covered = 0
    total = 0

    for x_batch, y_batch in dataloader:
        with torch.no_grad():
            inside = conformal_predictor.contains(x_batch, y_batch)

        inside = inside.detach().reshape(-1)
        if inside.numel() != x_batch.shape[0]:
            raise ValueError("contains must return one value per observation.")
        covered += int(inside.sum().item())
        total += inside.numel()

    if total == 0:
        raise ValueError("Validation dataloader must not be empty.")

    return covered / total, None


def worst_slab_coverage(
    dataloader: DataLoader,
    conformal_predictor: ConformalPredictor,
    delta: float = 0.1,
    number_of_directions: int = 1_000,
    test_size: float = 0.75,
    seed: int = 0,
) -> tuple[float, float]:
    """Return held-out worst-slab coverage mean and standard deviation."""
    representations = []
    coverages = []

    for x_batch, y_batch in dataloader:
        with torch.no_grad():
            inside = conformal_predictor.contains(x_batch, y_batch)

        inside = inside.detach().reshape(-1)
        if inside.numel() != x_batch.shape[0]:
            raise ValueError("contains must return one value per observation.")
        representations.append(
            x_batch.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
        )
        coverages.append(inside.to(device="cpu", dtype=torch.float64))

    if not coverages:
        raise ValueError("Validation dataloader must not be empty.")

    return wsc_unbiased(
        representations=torch.cat(representations).numpy(),
        coverages=torch.cat(coverages).numpy(),
        delta=delta,
        M=number_of_directions,
        test_size=test_size,
        random_state=seed,
    )
