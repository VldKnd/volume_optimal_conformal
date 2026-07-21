# src/predictors/transport/normalizing_flow.py

from __future__ import annotations

import math
from typing import Self

import torch
import torch.nn as nn

from configs.predictors.transport.normalizing_flow import (
    NormalizingFlowPredictorConfig,
)
from predictors.transport.base import BaseTransportPredictor
from networks.standard_scaler import FrozenStandardScaler


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
    ]

    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        ])

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class _AffineCouplingLayer(nn.Module):
    """
    Conditional RealNVP affine coupling layer.

    The mask dimensions are left unchanged and condition the shift/log-scale
    network for the remaining dimensions.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        y_dim: int,
        mask: torch.Tensor,
        hidden_dim: int,
        num_hidden_layers: int,
        log_scale_bound: float,
    ):
        super().__init__()

        self.y_dim = y_dim
        self.log_scale_bound = float(log_scale_bound)

        self.register_buffer("mask", mask.bool())
        self.register_buffer("transform_mask", ~mask.bool())

        self.num_transformed = int(self.transform_mask.sum().item())

        self.conditioner = _build_mlp(
            input_dim=x_dim + y_dim,
            output_dim=2 * self.num_transformed,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        self._initialize_as_identity()

    def _initialize_as_identity(self) -> None:
        final_layer = self.conditioner[-1]
        if not isinstance(final_layer, nn.Linear):
            return

        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.zero_()

    def _shift_and_log_scale(
        self,
        x: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioner_input = torch.cat(
            [
                x,
                inputs * self.mask.to(dtype=inputs.dtype),
            ],
            dim=-1,
        )
        raw_params = self.conditioner(conditioner_input).view(
            inputs.shape[0],
            self.num_transformed,
            2,
        )
        shift = raw_params[..., 0]
        log_scale = self.log_scale_bound * torch.tanh(raw_params[..., 1])

        return shift, log_scale

    def forward(
        self,
        x: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self._shift_and_log_scale(x=x, inputs=inputs)

        transformed_inputs = inputs[:, self.transform_mask]
        transformed_outputs = transformed_inputs * torch.exp(log_scale) + shift

        outputs = inputs.clone()
        outputs[:, self.transform_mask] = transformed_outputs

        return outputs, log_scale.sum(dim=-1)

    def inverse(
        self,
        x: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self._shift_and_log_scale(x=x, inputs=inputs)

        transformed_inputs = inputs[:, self.transform_mask]
        transformed_outputs = (transformed_inputs - shift) * torch.exp(-log_scale)

        outputs = inputs.clone()
        outputs[:, self.transform_mask] = transformed_outputs

        return outputs, -log_scale.sum(dim=-1)


class NormalizingFlowPredictor(nn.Module, BaseTransportPredictor):
    """
    Conditional RealNVP normalizing flow transport predictor.

    The public map is y = T_x(u), where u is standard Gaussian latent noise and
    y is returned in original target coordinates. Internally, the RealNVP flow
    maps u to scaled y coordinates, matching the transport predictor convention
    used elsewhere in the codebase.
    """

    def __init__(self, config: NormalizingFlowPredictorConfig):
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        torch.manual_seed(config.seed)

        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self.flow_layers = nn.ModuleList(
            [
                _AffineCouplingLayer(
                    x_dim=config.x_dim,
                    y_dim=config.y_dim,
                    mask=self._make_mask(layer_idx),
                    hidden_dim=config.hidden_dim,
                    num_hidden_layers=config.num_hidden_layers,
                    log_scale_bound=config.log_scale_bound,
                ) for layer_idx in range(config.num_flow_layers)
            ]
        ).to(device=self.device, dtype=self.dtype)

        self.y_scaler = FrozenStandardScaler(config.y_dim).to(
            device=self.device,
            dtype=self.dtype,
        )

    def _make_mask(self, layer_idx: int) -> torch.Tensor:
        if self.config.y_dim == 1:
            return torch.zeros(self.config.y_dim)

        return torch.tensor(
            [float((dim + layer_idx) % 2 == 0) for dim in range(self.config.y_dim)]
        )

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def warmup_y_scaler(self, dataloader) -> None:
        self.y_scaler.reset_running_stats()

        for _, y_batch in dataloader:
            y_batch = self.to_device(y_batch)
            self.y_scaler.update(y_batch)

        self.y_scaler.eval()

    def scale_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.y_scaler(y)

    def unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return (
            y_scaled * torch.sqrt(self.y_scaler.running_var + self.y_scaler.eps) +
            self.y_scaler.running_mean
        )

    def _unscale_y_log_det(self) -> torch.Tensor:
        return 0.5 * torch.log(self.y_scaler.running_var + self.y_scaler.eps).sum()

    def _scale_y_log_det(self) -> torch.Tensor:
        return -self._unscale_y_log_det()

    def _validate_condition_and_point_shapes(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got shape {tuple(x.shape)}.")

        if point.ndim != 2:
            raise ValueError(
                f"Expected point to be 2D, got shape {tuple(point.shape)}."
            )

        if x.shape[-1] != self.x_dim:
            raise ValueError(f"Expected x.shape[-1] = {self.x_dim}, got {x.shape[-1]}.")

        if point.shape[-1] != self.y_dim:
            raise ValueError(
                f"Expected point.shape[-1] = {self.y_dim}, "
                f"got {point.shape[-1]}."
            )

        if x.shape[0] != point.shape[0]:
            raise ValueError(
                f"Expected x.shape[0] == point.shape[0], got "
                f"{x.shape[0]} and {point.shape[0]}."
            )

    def _pushforward_scaled_with_log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.to_device(x)
        state = self.to_device(u)
        self._validate_condition_and_point_shapes(x=x, point=state)

        log_det = torch.zeros(
            state.shape[0],
            device=self.device,
            dtype=self.dtype,
        )

        for layer in self.flow_layers:
            state, layer_log_det = layer(x=x, inputs=state)
            log_det = log_det + layer_log_det

        return state, log_det

    def _pullback_scaled_with_log_det(
        self,
        x: torch.Tensor,
        y_scaled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.to_device(x)
        state = self.to_device(y_scaled)
        self._validate_condition_and_point_shapes(x=x, point=state)

        log_det = torch.zeros(
            state.shape[0],
            device=self.device,
            dtype=self.dtype,
        )

        for layer in reversed(self.flow_layers):
            state, layer_log_det = layer.inverse(x=x, inputs=state)
            log_det = log_det + layer_log_det

        return state, log_det

    @torch.no_grad()
    def pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()

        y_scaled, _ = self._pushforward_scaled_with_log_det(x=x, u=u)
        return self.unscale_y(y_scaled).detach()

    @torch.no_grad()
    def pushforward_with_log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        y_scaled, log_det = self._pushforward_scaled_with_log_det(x=x, u=u)
        y = self.unscale_y(y_scaled)

        return y.detach(), (log_det + self._unscale_y_log_det()).detach()

    @torch.no_grad()
    def pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()

        y_scaled = self.scale_y(self.to_device(y))
        u, _ = self._pullback_scaled_with_log_det(x=x, y_scaled=y_scaled)

        return u.detach()

    @torch.no_grad()
    def pullback_with_log_det(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        y_scaled = self.scale_y(self.to_device(y))
        u, log_det = self._pullback_scaled_with_log_det(
            x=x,
            y_scaled=y_scaled,
        )

        return u.detach(), (log_det + self._scale_y_log_det()).detach()

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return log |det D_u T_x(u)| for the public unscaled pushforward map.
        """
        self.eval()

        _, log_det = self._pushforward_scaled_with_log_det(x=x, u=u)
        return log_det + self._unscale_y_log_det()

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact conditional log density log p(y | x) under the flow.
        """
        x = self.to_device(x)
        y_scaled = self.scale_y(self.to_device(y))

        u, inverse_log_det = self._pullback_scaled_with_log_det(
            x=x,
            y_scaled=y_scaled,
        )
        inverse_log_det = inverse_log_det + self._scale_y_log_det()

        base_log_prob = (
            -0.5 * u.square() - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)

        return base_log_prob + inverse_log_det

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.pullback(x=x, y=y)

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        x = self.to_device(x)

        batch_size = x.shape[0]

        u = torch.randn(
            batch_size,
            n_samples,
            self.y_dim,
            device=self.device,
            dtype=self.dtype,
        )

        x_rep = x[:, None, :].expand(
            batch_size,
            n_samples,
            self.x_dim,
        ).reshape(batch_size * n_samples, self.x_dim)
        u_flat = u.reshape(batch_size * n_samples, self.y_dim)

        y_flat = self.pushforward(x=x_rep, u=u_flat)

        return y_flat.reshape(batch_size, n_samples, self.y_dim)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.model_dump(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> Self:
        data = torch.load(path, map_location=map_location, weights_only=False)

        config = NormalizingFlowPredictorConfig.model_validate(data["config"])
        model = cls(config)
        model.load_state_dict(data["state_dict"])
        model.to(device=model.device, dtype=model.dtype)

        return model
