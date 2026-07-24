"""Conditional convex-potential flow predictor.

The implementation is adapted from Huang et al., "Convex Potential Flows:
Universal Probability Distributions with Optimal Transport and Convex
Optimization" (2021).
"""

from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn

from configs.predictors.transport.convex_potential_flow import (
    ConvexPotentialFlowPredictorConfig,
)
from predictors.transport.base import BaseTransportPredictor
from predictors.transport.convex_potential_flow.cpflows import DeepConvexFlow
from predictors.transport.convex_potential_flow.flows import ActNorm, SequentialFlow
from predictors.transport.convex_potential_flow.icnn import PICNN


class ConvexPotentialFlowPredictor(nn.Module, BaseTransportPredictor):
    """Conditional convex-potential normalizing flow.

    Internally, ``flow.forward_transform`` maps observations ``y`` to standard
    Gaussian latent variables ``u``. The public transport convention used by
    this repository is the inverse: ``pushforward(x, u)`` maps ``u`` to ``y``.
    """

    def __init__(self, config: ConvexPotentialFlowPredictorConfig):
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        torch.manual_seed(config.seed)

        convex_layers = [
            DeepConvexFlow(
                PICNN(
                    dim=config.y_dim,
                    dimh=config.hidden_dim,
                    dimc=config.x_dim,
                    num_hidden_layers=config.num_hidden_layers,
                    symm_act_first=True,
                    softplus_type="softplus",
                    zero_softplus=True,
                ),
                dim=config.y_dim,
                unbiased=False,
            ) for _ in range(config.num_blocks)
        ]

        layers: list[nn.Module] = []
        for convex_layer in convex_layers:
            layers.extend([ActNorm(config.y_dim), convex_layer])
        layers.append(ActNorm(config.y_dim))

        self.flow = SequentialFlow(layers).to(
            device=self.device,
            dtype=self.dtype,
        )

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        for layer in self.flow.flows:
            if isinstance(layer, DeepConvexFlow):
                layer.no_bruteforce = mode
        return self

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype)

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.to_device(x)
        point = self.to_device(point)

        if x.ndim != 2 or x.shape[1] != self.x_dim:
            raise ValueError(
                f"Expected x with shape (batch, {self.x_dim}), "
                f"got {tuple(x.shape)}."
            )
        if point.ndim != 2 or point.shape[1] != self.y_dim:
            raise ValueError(
                f"Expected points with shape (batch, {self.y_dim}), "
                f"got {tuple(point.shape)}."
            )
        if x.shape[0] != point.shape[0]:
            raise ValueError("x and points must have the same batch size.")

        return x, point

    def _pullback_without_log_det(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        state = y
        for layer in self.flow.flows:
            if isinstance(layer, DeepConvexFlow):
                state = layer(state, context=x)
            else:
                state, _ = layer.forward_transform(state)
        return state

    def _reverse(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow.reverse(
            u,
            context=x,
            max_iter=self.config.inverse_max_iter,
            lr=self.config.inverse_lr,
            tol=self.config.inverse_tolerance,
        )

    @torch.enable_grad()
    def pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        x, u = self._prepare_inputs(x=x, point=u)
        return self._reverse(x=x, u=u).detach()

    @torch.enable_grad()
    def pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        x, y = self._prepare_inputs(x=x, point=y)
        return self._pullback_without_log_det(x=x, y=y).detach()

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.pullback(x=x, y=y)

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        x, y = self._prepare_inputs(x=x, point=y)
        return self.flow.logp(y, context=x)

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log |det D_u T_x(u)|`` for the public pushforward.

        The numerical inverse used by the copied flow is detached. When ``u``
        requires gradients, the method therefore supplies the exact first-order
        implicit gradient

        ``J_G(y)^(-T) grad_y[-log det J_G(y)]``,

        where ``G`` is the internal observation-to-latent map and
        ``y = G^(-1)(u)``. This is the gradient required when the predictor is
        wrapped by a trainable measure-preserving rearrangement.
        """
        implicit_gradient = torch.is_grad_enabled() and u.requires_grad

        self.eval()
        x, u = self._prepare_inputs(x=x, point=u)

        with torch.enable_grad():
            y = self._reverse(x=x.detach(), u=u.detach()).detach()

            if not implicit_gradient:
                _, inverse_log_det = self.flow.forward_transform(
                    y,
                    context=x.detach(),
                    create_graph=False,
                )
                return (-inverse_log_det).detach()

            y = y.requires_grad_(True)
            latent, inverse_log_det = self.flow.forward_transform(
                y,
                context=x.detach(),
                create_graph=True,
            )
            value = -inverse_log_det

            gradient_y = torch.autograd.grad(
                value.sum(),
                y,
                retain_graph=True,
            )[0]

            jacobian_rows = [
                torch.autograd.grad(
                    latent[:, output_dimension].sum(),
                    y,
                    retain_graph=output_dimension < self.y_dim - 1,
                )[0] for output_dimension in range(self.y_dim)
            ]
            jacobian = torch.stack(jacobian_rows, dim=1)
            gradient_u = torch.linalg.solve(
                jacobian.transpose(-1, -2),
                gradient_y.unsqueeze(-1),
            ).squeeze(-1)

        return (value.detach() + ((u - u.detach()) * gradient_u.detach()).sum(dim=-1))

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
        config = ConvexPotentialFlowPredictorConfig.model_validate(data["config"])
        model = cls(config)
        model.load_state_dict(data["state_dict"])
        model.to(device=model.device, dtype=model.dtype)
        model.eval()
        return model
