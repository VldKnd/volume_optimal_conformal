# src/predictors/transport/neural_spline_flow.py

from __future__ import annotations

import math
from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.predictors.transport.neural_spline_flow import (
    NeuralSplineFlowPredictorConfig,
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


def _inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


def _rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    *,
    inverse: bool,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monotone rational-quadratic spline with linear tails.

    The spline maps [-tail_bound, tail_bound] to itself. Inputs outside the
    interval are left unchanged and contribute zero log-determinant.
    """
    num_bins = unnormalized_widths.shape[-1]
    eps = torch.finfo(inputs.dtype).eps

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    widths = 2.0 * tail_bound * widths

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    heights = 2.0 * tail_bound * heights

    internal_derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    boundary_derivatives = torch.ones(
        *internal_derivatives.shape[:-1],
        1,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    derivatives = torch.cat(
        [
            boundary_derivatives,
            internal_derivatives,
            boundary_derivatives,
        ],
        dim=-1,
    )

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = cumwidths - tail_bound
    cumwidths[..., 0] = -tail_bound
    cumwidths[..., -1] = tail_bound

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = cumheights - tail_bound
    cumheights[..., 0] = -tail_bound
    cumheights[..., -1] = tail_bound

    inside_interval = (inputs >= -tail_bound) & (inputs <= tail_bound)
    clipped_inputs = inputs.clamp(
        min=-tail_bound + eps,
        max=tail_bound - eps,
    )

    bin_boundaries = cumheights if inverse else cumwidths
    bin_idx = (
        torch.searchsorted(
            bin_boundaries.contiguous(),
            clipped_inputs.unsqueeze(-1).contiguous(),
            right=True,
        ) - 1
    )
    bin_idx = bin_idx.clamp(min=0, max=num_bins - 1)

    input_cumwidths = cumwidths.gather(-1, bin_idx).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx).squeeze(-1)
    input_cumheights = cumheights.gather(-1, bin_idx).squeeze(-1)
    input_bin_heights = heights.gather(-1, bin_idx).squeeze(-1)
    input_derivatives = derivatives.gather(-1, bin_idx).squeeze(-1)
    input_derivatives_plus_one = derivatives.gather(
        -1,
        bin_idx + 1,
    ).squeeze(-1)

    delta = input_bin_heights / input_bin_widths

    if inverse:
        shifted_inputs = clipped_inputs - input_cumheights
        derivative_sum = input_derivatives + input_derivatives_plus_one
        a = (
            shifted_inputs * (derivative_sum - 2.0 * delta) + input_bin_heights *
            (delta - input_derivatives)
        )
        b = (
            input_bin_heights * input_derivatives - shifted_inputs *
            (derivative_sum - 2.0 * delta)
        )
        c = -delta * shifted_inputs

        discriminant = (b.square() - 4.0 * a * c).clamp_min(eps)
        theta = 2.0 * c / (-b - torch.sqrt(discriminant))
        theta = theta.clamp(min=0.0, max=1.0)
        outputs_inside = input_cumwidths + theta * input_bin_widths

    else:
        theta = (clipped_inputs - input_cumwidths) / input_bin_widths
        theta = theta.clamp(min=0.0, max=1.0)
        theta_one_minus_theta = theta * (1.0 - theta)

        numerator = input_bin_heights * (
            delta * theta.square() + input_derivatives * theta_one_minus_theta
        )
        denominator = (
            delta + (input_derivatives + input_derivatives_plus_one - 2.0 * delta) *
            theta_one_minus_theta
        )
        outputs_inside = input_cumheights + numerator / denominator

    theta_one_minus_theta = theta * (1.0 - theta)
    denominator = (
        delta + (input_derivatives + input_derivatives_plus_one - 2.0 * delta) *
        theta_one_minus_theta
    )
    derivative_numerator = delta.square() * (
        input_derivatives_plus_one * theta.square() +
        2.0 * delta * theta_one_minus_theta + input_derivatives * (1.0 - theta).square()
    )
    derivative_denominator = denominator.square()

    logabsdet_inside = (
        torch.log(derivative_numerator.clamp_min(eps)) -
        torch.log(derivative_denominator.clamp_min(eps))
    )

    if inverse:
        logabsdet_inside = -logabsdet_inside

    outputs = torch.where(inside_interval, outputs_inside, inputs)
    logabsdet = torch.where(
        inside_interval,
        logabsdet_inside,
        torch.zeros_like(inputs),
    )

    return outputs, logabsdet


class _RationalQuadraticSplineCoupling(nn.Module):

    def __init__(
        self,
        *,
        x_dim: int,
        y_dim: int,
        mask: torch.Tensor,
        hidden_dim: int,
        num_hidden_layers: int,
        num_bins: int,
        tail_bound: float,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
    ):
        super().__init__()

        self.y_dim = y_dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.register_buffer("mask", mask)
        self.register_buffer("transform_mask", ~mask.bool())

        self.num_transformed = int(self.transform_mask.sum().item())
        self.params_per_dim = 3 * num_bins - 1

        self.conditioner = _build_mlp(
            input_dim=x_dim + y_dim,
            output_dim=self.num_transformed * self.params_per_dim,
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

            derivative_bias = _inverse_softplus(1.0 - self.min_derivative)
            bias = final_layer.bias.view(
                self.num_transformed,
                self.params_per_dim,
            )
            bias[:, 2 * self.num_bins:] = derivative_bias

    def _spline_params(
        self,
        x: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conditioner_input = torch.cat([x, inputs * self.mask], dim=-1)
        raw_params = self.conditioner(conditioner_input).view(
            inputs.shape[0],
            self.num_transformed,
            self.params_per_dim,
        )

        unnormalized_widths = raw_params[..., :self.num_bins]
        unnormalized_heights = raw_params[
            ...,
            self.num_bins:2 * self.num_bins,
        ]
        unnormalized_derivatives = raw_params[..., 2 * self.num_bins:]

        return (
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
        )

    def forward(
        self,
        x: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._call(x=x, inputs=inputs, inverse=False)

    def inverse(
        self,
        x: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._call(x=x, inputs=inputs, inverse=True)

    def _call(
        self,
        x: torch.Tensor,
        inputs: torch.Tensor,
        *,
        inverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
        ) = self._spline_params(x=x, inputs=inputs)

        transformed_inputs = inputs[:, self.transform_mask]
        transformed_outputs, logabsdet = _rational_quadratic_spline(
            inputs=transformed_inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        outputs = inputs.clone()
        outputs[:, self.transform_mask] = transformed_outputs

        return outputs, logabsdet.sum(dim=-1)


class NeuralSplineFlowPredictor(nn.Module, BaseTransportPredictor):
    """
    Conditional rational-quadratic neural spline flow.

    The public transport map is

        y = T_x(u),

    where u is standard Gaussian latent noise and y is returned in the original
    target coordinates. Internally, the invertible spline maps u to scaled y,
    matching the scaler convention used by the other transport predictors.
    """

    def __init__(self, config: NeuralSplineFlowPredictorConfig):
        super().__init__()

        self._validate_config(config)

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        torch.manual_seed(config.seed)

        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self.flow_layers = nn.ModuleList(
            [
                _RationalQuadraticSplineCoupling(
                    x_dim=config.x_dim,
                    y_dim=config.y_dim,
                    mask=self._make_mask(layer_idx),
                    hidden_dim=config.hidden_dim,
                    num_hidden_layers=config.num_hidden_layers,
                    num_bins=config.num_bins,
                    tail_bound=config.tail_bound,
                    min_bin_width=config.min_bin_width,
                    min_bin_height=config.min_bin_height,
                    min_derivative=config.min_derivative,
                ) for layer_idx in range(config.num_flow_layers)
            ]
        ).to(device=self.device, dtype=self.dtype)

        self.y_scaler = FrozenStandardScaler(config.y_dim).to(
            device=self.device,
            dtype=self.dtype,
        )

    def _validate_config(
        self,
        config: NeuralSplineFlowPredictorConfig,
    ) -> None:
        if config.num_bins * config.min_bin_width >= 1.0:
            raise ValueError("num_bins * min_bin_width must be smaller than 1.0.")

        if config.num_bins * config.min_bin_height >= 1.0:
            raise ValueError("num_bins * min_bin_height must be smaller than 1.0.")

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

        base_log_prob = (-0.5 * u.square() - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)

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

        x_rep = (
            x[:,
              None, :].expand(batch_size, n_samples,
                              self.x_dim).reshape(batch_size * n_samples, self.x_dim)
        )
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

        config = NeuralSplineFlowPredictorConfig.model_validate(data["config"])
        model = cls(config)
        model.load_state_dict(data["state_dict"])
        model.to(device=model.device, dtype=model.dtype)

        return model
