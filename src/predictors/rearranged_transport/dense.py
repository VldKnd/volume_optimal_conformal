# src/predictors/rearranged_transport/dense.py

from __future__ import annotations

import torch
import torch.nn as nn

from configs.predictors.rearranged_transport.dense import (
    DenseRearrangedTransportPredictorConfig,
)
from networks.measure_preserving_flows.flow_integration import (
    GaussianSkewFieldFlow,
)
from predictors.rearranged_transport.base import (
    BaseRearrangedTransportPredictor,
)
from predictors.transport.base import BaseTransportPredictor


class DenseRearrangedTransportPredictor(
    nn.Module,
    BaseRearrangedTransportPredictor,
):
    """
    Rearranged transport T o S.

    S is a conditional Gaussian-preserving dense skew flow in latent space.
    T is an already-trained transport predictor supplied by the caller.
    """

    def __init__(
        self,
        config: DenseRearrangedTransportPredictorConfig,
        transport_predictor: BaseTransportPredictor,
    ):
        super().__init__()

        self.config = config
        self.transport_predictor = transport_predictor

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        torch.manual_seed(config.seed)

        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self._validate_transport_predictor()
        self._move_transport_predictor_to_device()

        self.rearrangement_flow = GaussianSkewFieldFlow(
            dimension=config.y_dim,
            context_dimension=config.x_dim,
            use_adjoint=config.use_adjoint,
            method=config.method,
            rtol=config.rtol,
            atol=config.atol,
            number_of_steps=config.number_of_steps,
            endpoint_alpha=0.1,
            hidden_dimension=config.hidden_dimension,
            number_of_hidden_layers=config.number_of_hidden_layers,
            time_dependent=config.time_dependent,
        ).to(device=self.device, dtype=self.dtype)

    def _validate_transport_predictor(self) -> None:
        predictor_x_dim = getattr(self.transport_predictor, "x_dim", None)
        predictor_y_dim = getattr(self.transport_predictor, "y_dim", None)

        if predictor_x_dim is not None and predictor_x_dim != self.x_dim:
            raise ValueError(
                f"Expected transport predictor x_dim={self.x_dim}, "
                f"got {predictor_x_dim}."
            )

        if predictor_y_dim is not None and predictor_y_dim != self.y_dim:
            raise ValueError(
                f"Expected transport predictor y_dim={self.y_dim}, "
                f"got {predictor_y_dim}."
            )

    def _move_transport_predictor_to_device(self) -> None:
        if isinstance(self.transport_predictor, nn.Module):
            self.transport_predictor.to(device=self.device, dtype=self.dtype)

        if hasattr(self.transport_predictor, "device"):
            self.transport_predictor.device = self.device

        if hasattr(self.transport_predictor, "dtype"):
            self.transport_predictor.dtype = self.dtype

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype)

    def rearrangement_pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        x = self.to_device(x)
        u = self.to_device(u)
        return self.rearrangement_flow.forward(u=u, x=x, end_time=1.0)

    def rearrangement_pullback(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        x = self.to_device(x)
        u = self.to_device(u)
        return self.rearrangement_flow.inverse(u=u, x=x, start_time=1.0)

    def transport_pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        return self.transport_predictor.pushforward(
            x=self.to_device(x),
            u=self.to_device(u),
        )

    def transport_pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.transport_predictor.pullback(
            x=self.to_device(x),
            y=self.to_device(y),
        )

    def pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        rearranged_u = self.rearrangement_pushforward(x=x, u=u)
        return self.transport_pushforward(x=x, u=rearranged_u)

    def pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        transport_u = self.transport_pullback(x=x, y=y)
        return self.rearrangement_pullback(x=x, u=transport_u)

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        log |det D_u (T_x o S_x)(u)|.
        """
        rearranged_u = self.rearrangement_pushforward(x=x, u=u)
        return self.transport_log_det(
            x=x,
            u=rearranged_u,
        ) + self._rearrangement_log_det_from_pushforward(
            u=self.to_device(u),
            rearranged_u=rearranged_u,
        )

    def transport_log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        return self.transport_predictor.log_det(
            x=self.to_device(x),
            u=self.to_device(u),
        )

    def rearrangement_log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        rearranged_u = self.rearrangement_pushforward(x=x, u=u)
        return self._rearrangement_log_det_from_pushforward(
            u=self.to_device(u),
            rearranged_u=rearranged_u,
        )

    def _rearrangement_log_det_from_pushforward(
        self,
        u: torch.Tensor,
        rearranged_u: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (rearranged_u.square().sum(dim=-1) - u.square().sum(dim=-1))

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.pullback(x=x, y=y)
