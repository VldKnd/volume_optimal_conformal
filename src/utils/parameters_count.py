from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn
from pydantic import BaseModel

from configs.predictors.rearranged_transport import (
    AmortizedRearrangedTransportPredictorConfig,
    RearrangedTransportPredictorConfig,
)
from configs.predictors.transport import (
    FlowMatchingPredictorConfig,
    NeuralOptimalTransportPredictorConfig,
    NeuralSplineFlowPredictorConfig,
    NormalizingFlowPredictorConfig,
)
from networks.measure_preserving_flows import GaussianSkewFieldFlow
from networks.measure_preserving_flows.sparse_skew_symmetric_vector_field import (
    SparseGaussianSkewVectorField,
)
from predictors.transport import (
    FlowMatchingPredictor,
    NeuralOptimalTransportPredictor,
    NeuralSplineFlowPredictor,
    NormalizingFlowPredictor,
)

PredictorConfig = (
    FlowMatchingPredictorConfig
    | NeuralOptimalTransportPredictorConfig
    | NeuralSplineFlowPredictorConfig
    | NormalizingFlowPredictorConfig
    | AmortizedRearrangedTransportPredictorConfig
    | RearrangedTransportPredictorConfig
)

_CONFIG_BY_TYPE = {
    "flow_matching": FlowMatchingPredictorConfig,
    "neural_optimal_transport": NeuralOptimalTransportPredictorConfig,
    "neural_spline_flow": NeuralSplineFlowPredictorConfig,
    "normalizing_flow": NormalizingFlowPredictorConfig,
    "amortized_rearranged_transport": AmortizedRearrangedTransportPredictorConfig,
    "rearranged_transport": RearrangedTransportPredictorConfig,
    "dense_rearranged_transport": RearrangedTransportPredictorConfig,
}

_PREDICTOR_BY_CONFIG_TYPE = {
    FlowMatchingPredictorConfig: FlowMatchingPredictor,
    NeuralOptimalTransportPredictorConfig: NeuralOptimalTransportPredictor,
    NeuralSplineFlowPredictorConfig: NeuralSplineFlowPredictor,
    NormalizingFlowPredictorConfig: NormalizingFlowPredictor,
}


def count_trainable_parameters(predictor: nn.Module) -> int:
    """Return the number of trainable parameters in a predictor/module."""
    return sum(
        parameter.numel()
        for parameter in predictor.parameters()
        if parameter.requires_grad
    )


def count_trainable_parameters_from_config(
    config: PredictorConfig | Mapping[str, Any],
    *,
    transport_predictor_config: PredictorConfig | Mapping[str, Any] | None = None,
) -> int:
    """Return the trainable-parameter count implied by a predictor config.

    Configs are copied to CPU before instantiation, so this utility can count a
    CUDA-configured predictor on a CPU-only machine.

    ``RearrangedTransportPredictorConfig`` does not contain the wrapped transport
    predictor. For that config, this function counts the rearrangement flow
    parameters. If ``transport_predictor_config`` is supplied, its trainable
    parameters are added as the wrapped transport contribution.
    """
    parsed_config = _parse_predictor_config(config)

    if isinstance(parsed_config, RearrangedTransportPredictorConfig):
        count = count_trainable_parameters(
            _build_rearrangement_flow_from_config(parsed_config)
        )
        if transport_predictor_config is not None:
            count += count_trainable_parameters_from_config(
                transport_predictor_config,
            )
        return count

    predictor = _build_transport_predictor_from_config(parsed_config)
    return count_trainable_parameters(predictor)


def _parse_predictor_config(
    config: PredictorConfig | Mapping[str, Any],
) -> PredictorConfig:
    if isinstance(config, tuple(_CONFIG_BY_TYPE.values())):
        return config

    if isinstance(config, Mapping):
        config_type = config.get("type")
        if config_type not in _CONFIG_BY_TYPE:
            raise ValueError(
                f"Unknown predictor config type {config_type!r}. "
                f"Expected one of {sorted(_CONFIG_BY_TYPE)}."
            )
        return _CONFIG_BY_TYPE[config_type].model_validate(config)

    raise TypeError(
        "config must be a known predictor config instance or a mapping with a "
        "'type' field."
    )


def _copy_config_to_cpu(config: PredictorConfig) -> PredictorConfig:
    if isinstance(config, BaseModel):
        return config.model_copy(update={"device": "cpu"})

    raise TypeError(f"Expected a pydantic config, got {type(config)!r}.")


def _build_transport_predictor_from_config(
    config: PredictorConfig,
) -> nn.Module:
    cpu_config = _copy_config_to_cpu(config)
    predictor_class = _PREDICTOR_BY_CONFIG_TYPE.get(type(cpu_config))
    if predictor_class is None:
        raise ValueError(
            f"Config {type(config).__name__} is not a standalone transport "
            "predictor config."
        )

    return predictor_class(cpu_config)


def _build_rearrangement_flow_from_config(
    config: RearrangedTransportPredictorConfig,
) -> nn.Module:
    cpu_config = _copy_config_to_cpu(config)
    context_dimension = cpu_config.x_dim
    if isinstance(cpu_config, AmortizedRearrangedTransportPredictorConfig):
        context_dimension += 1

    vector_field = None
    if cpu_config.vector_field_implementation == "sparse":
        vector_field = SparseGaussianSkewVectorField(
            dimension=cpu_config.y_dim,
            context_dimension=context_dimension,
            hidden_dimension=cpu_config.hidden_dimension,
            number_of_hidden_layers=cpu_config.number_of_hidden_layers,
            time_dependent=cpu_config.time_dependent,
            activation=cpu_config.activation,
            activation_power=cpu_config.activation_power,
        )

    return GaussianSkewFieldFlow(
        dimension=cpu_config.y_dim,
        context_dimension=context_dimension,
        vector_field=vector_field,
        use_adjoint=cpu_config.use_adjoint,
        method=cpu_config.method,
        rtol=cpu_config.rtol,
        atol=cpu_config.atol,
        number_of_steps=cpu_config.number_of_steps,
        endpoint_alpha=0.1,
        hidden_dimension=cpu_config.hidden_dimension,
        number_of_hidden_layers=cpu_config.number_of_hidden_layers,
        time_dependent=cpu_config.time_dependent,
        activation=cpu_config.activation,
        activation_power=cpu_config.activation_power,
    )
