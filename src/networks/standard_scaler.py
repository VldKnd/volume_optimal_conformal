from __future__ import annotations

import torch
import torch.nn as nn


class FrozenStandardScaler(nn.Module):
    """Exact dataset standardizer with no trainable parameters.

    The buffer names intentionally match the previous normalizer's running
    statistics so older checkpoints that only rely on fitted mean/variance
    remain compatible. ``num_batches_tracked`` stores the number of samples
    accumulated by this module, not the number of mini-batches.
    """

    def __init__(
        self,
        feature_dimension: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        if feature_dimension <= 0:
            raise ValueError("feature_dimension must be positive.")

        self.feature_dimension = int(feature_dimension)
        self.eps = float(eps)

        self.register_buffer("running_mean", torch.zeros(feature_dimension))
        self.register_buffer("running_var", torch.ones(feature_dimension))
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long),
        )

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_batches_tracked.zero_()

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = self._flatten_values(values.detach())
        if values.shape[0] == 0:
            return

        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_m2 = (values - batch_mean).square().sum(dim=0)

        current_count = int(self.num_batches_tracked.item())
        if current_count == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_m2 / batch_count)
            self.num_batches_tracked.fill_(batch_count)
            return

        total_count = current_count + batch_count
        current_count_tensor = values.new_tensor(float(current_count))
        batch_count_tensor = values.new_tensor(float(batch_count))
        total_count_tensor = values.new_tensor(float(total_count))

        delta = batch_mean - self.running_mean
        current_m2 = self.running_var * current_count_tensor
        total_m2 = (
            current_m2
            + batch_m2
            + delta.square()
            * current_count_tensor
            * batch_count_tensor
            / total_count_tensor
        )

        self.running_mean.add_(delta * batch_count_tensor / total_count_tensor)
        self.running_var.copy_(total_m2 / total_count_tensor)
        self.num_batches_tracked.fill_(total_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.transform(values)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        self._check_values(values)
        return (values - self.running_mean) / torch.sqrt(self.running_var + self.eps)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        self._check_values(values)
        return values * torch.sqrt(self.running_var + self.eps) + self.running_mean

    def _flatten_values(self, values: torch.Tensor) -> torch.Tensor:
        self._check_values(values)
        return values.reshape(-1, self.feature_dimension)

    def _check_values(self, values: torch.Tensor) -> None:
        if values.shape[-1] != self.feature_dimension:
            raise ValueError(
                f"Expected values.shape[-1] = {self.feature_dimension}, "
                f"got {values.shape[-1]}."
            )
