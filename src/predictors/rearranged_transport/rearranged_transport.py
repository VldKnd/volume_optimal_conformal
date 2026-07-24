# src/predictors/rearranged_transport/dense.py

from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn

from configs.predictors.rearranged_transport.rearranged_transport import (
    RearrangedTransportPredictorConfig,
)
from configs.predictors.transport import (
    ConvexPotentialFlowPredictorConfig,
    FlowMatchingPredictorConfig,
    NeuralOptimalTransportPredictorConfig,
    NeuralSplineFlowPredictorConfig,
    NormalizingFlowPredictorConfig,
)
from networks.measure_preserving_flows.flow_integration import (
    GaussianSkewFieldFlow,
)
from networks.measure_preserving_flows.sparse_skew_symmetric_vector_field import (
    SparseGaussianSkewVectorField,
)
from predictors.rearranged_transport.base import (
    BaseRearrangedTransportPredictor,
)
from predictors.transport import (
    ConvexPotentialFlowPredictor,
    FlowMatchingPredictor,
    NeuralOptimalTransportPredictor,
    NeuralSplineFlowPredictor,
    NormalizingFlowPredictor,
)
from predictors.transport.base import BaseTransportPredictor

_TRANSPORT_CONFIG_BY_TYPE = {
    "convex_potential_flow": ConvexPotentialFlowPredictorConfig,
    "flow_matching": FlowMatchingPredictorConfig,
    "neural_optimal_transport": NeuralOptimalTransportPredictorConfig,
    "neural_spline_flow": NeuralSplineFlowPredictorConfig,
    "normalizing_flow": NormalizingFlowPredictorConfig,
}

_TRANSPORT_PREDICTOR_BY_TYPE = {
    "convex_potential_flow": ConvexPotentialFlowPredictor,
    "flow_matching": FlowMatchingPredictor,
    "neural_optimal_transport": NeuralOptimalTransportPredictor,
    "neural_spline_flow": NeuralSplineFlowPredictor,
    "normalizing_flow": NormalizingFlowPredictor,
}


class RearrangedTransportPredictor(
    nn.Module,
    BaseRearrangedTransportPredictor,
):
    """
    Rearranged transport T o S.

    S is a conditional Gaussian-preserving flow in latent space.
    T is an already-trained transport predictor supplied by the caller.
    """

    config_class = RearrangedTransportPredictorConfig

    def __init__(
        self,
        config: RearrangedTransportPredictorConfig,
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

        vector_field = self._make_rearrangement_vector_field()
        self.rearrangement_flow = GaussianSkewFieldFlow(
            dimension=config.y_dim,
            context_dimension=self._rearrangement_context_dimension(),
            vector_field=vector_field,
            use_adjoint=config.use_adjoint,
            method=config.method,
            rtol=config.rtol,
            atol=config.atol,
            number_of_steps=config.number_of_steps,
            endpoint_alpha=0.1,
            hidden_dimension=config.hidden_dimension,
            number_of_hidden_layers=config.number_of_hidden_layers,
            time_dependent=config.time_dependent,
            activation=config.activation,
            activation_power=config.activation_power,
        ).to(device=self.device, dtype=self.dtype)

    def _make_rearrangement_vector_field(self) -> nn.Module | None:
        if self.config.vector_field_implementation == "standard":
            return None

        if self.config.vector_field_implementation == "sparse":
            return SparseGaussianSkewVectorField(
                dimension=self.config.y_dim,
                context_dimension=self._rearrangement_context_dimension(),
                hidden_dimension=self.config.hidden_dimension,
                number_of_hidden_layers=self.config.number_of_hidden_layers,
                time_dependent=self.config.time_dependent,
                activation=self.config.activation,
                activation_power=self.config.activation_power,
            )

        raise ValueError(
            "Unknown vector_field_implementation="
            f"{self.config.vector_field_implementation!r}."
        )

    def _rearrangement_context_dimension(self) -> int:
        return self.x_dim

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
        return self.rearrangement_flow.forward(u=u, x=x)

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

    def save(self, path: str) -> None:
        transport_config = getattr(self.transport_predictor, "config", None)
        if transport_config is None or not hasattr(transport_config, "model_dump"):
            raise TypeError(
                "The wrapped transport predictor must expose a Pydantic "
                "config to be saved."
            )

        transport_config_data = transport_config.model_dump()
        transport_config_data["device"] = str(self.device)
        transport_config_data["dtype"] = self.config.dtype
        transport_type = transport_config_data.get("type")
        expected_predictor_class = _TRANSPORT_PREDICTOR_BY_TYPE.get(transport_type)

        if expected_predictor_class is None:
            raise ValueError(
                f"Unsupported wrapped transport predictor type {transport_type!r}. "
                f"Expected one of {sorted(_TRANSPORT_PREDICTOR_BY_TYPE)}."
            )

        if not isinstance(self.transport_predictor, expected_predictor_class):
            raise TypeError(
                f"Transport config type {transport_type!r} does not match "
                f"predictor class {type(self.transport_predictor).__name__}."
            )

        torch.save(
            {
                "config": self.config.model_dump(),
                "rearrangement_state_dict": self.rearrangement_flow.state_dict(),
                "transport_predictor": {
                    "config": transport_config_data,
                    "state_dict": self.transport_predictor.state_dict(),
                },
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

        device = str(torch.device(map_location))
        config_data = dict(data["config"])
        config_data["device"] = device
        config = cls.config_class.model_validate(config_data)

        transport_data = data.get("transport_predictor")
        is_legacy_checkpoint = transport_data is None

        if is_legacy_checkpoint:
            transport_config_data = dict(data["transport_predictor_config"])
        else:
            transport_config_data = dict(transport_data["config"])

        transport_config_data["device"] = device
        transport_config_data["dtype"] = config.dtype
        transport_predictor = cls._make_transport_predictor(
            transport_config_data=transport_config_data,
        )

        if not is_legacy_checkpoint:
            transport_predictor.load_state_dict(transport_data["state_dict"])

        model = cls(
            config=config,
            transport_predictor=transport_predictor,
        )

        if is_legacy_checkpoint:
            model.load_state_dict(data["state_dict"])
        else:
            model.rearrangement_flow.load_state_dict(data["rearrangement_state_dict"])

        model.to(device=model.device, dtype=model.dtype)

        return model

    @staticmethod
    def _make_transport_predictor(
        transport_config_data: dict,
    ) -> BaseTransportPredictor:
        transport_type = transport_config_data.get("type")
        config_class = _TRANSPORT_CONFIG_BY_TYPE.get(transport_type)
        predictor_class = _TRANSPORT_PREDICTOR_BY_TYPE.get(transport_type)

        if config_class is None or predictor_class is None:
            raise ValueError(
                f"Unsupported wrapped transport predictor type {transport_type!r}. "
                f"Expected one of {sorted(_TRANSPORT_PREDICTOR_BY_TYPE)}."
            )

        transport_config = config_class.model_validate(transport_config_data)
        return predictor_class(transport_config)
