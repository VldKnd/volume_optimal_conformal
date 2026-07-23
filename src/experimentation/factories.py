from __future__ import annotations

from pathlib import Path

import torch

from configs.predictors.transport import (
    FlowMatchingPredictorConfig,
    NeuralOptimalTransportPredictorConfig,
    NeuralSplineFlowPredictorConfig,
    NormalizingFlowPredictorConfig,
)
from data.datasets.synthetic import (
    BananaDataset,
    BimodalGaussianDataset,
    GaussianDatasetTarget,
    SinusoidalTransportDataset,
    StudentTDataset,
)
from experimentation.config import (
    AmortizedRearrangedTransportStageConfig,
    BaseStageConfig,
    DatasetConfig,
    RearrangementStageConfig,
    SupervisedRearrangedTransportStageConfig,
)
from predictors.rearranged_transport import (
    AmortizedRearrangedTransport,
    RearrangedTransportPredictor,
)
from predictors.transport import (
    FlowMatchingPredictor,
    NeuralOptimalTransportPredictor,
    NeuralSplineFlowPredictor,
    NormalizingFlowPredictor,
)
from trainers.rearranged_transport import (
    AmortizedRearrangedTransportTrainer,
    RearrangedTransportTrainer,
    SupervisedRearrangedTransportTrainer,
)
from trainers.transport import (
    FlowMatchingTrainer,
    NeuralOptimalTransportTrainer,
    NeuralSplineFlowTrainer,
    NormalizingFlowTrainer,
)

_DATASET_BY_TYPE = {
    "banana": BananaDataset,
    "bimodal_gaussian": BimodalGaussianDataset,
    "gaussian_dataset": GaussianDatasetTarget,
    "sinusoidal_transport": SinusoidalTransportDataset,
    "student_t_dataset": StudentTDataset,
}

_BASE_PREDICTOR_BY_TYPE = {
    "flow_matching": FlowMatchingPredictor,
    "neural_optimal_transport": NeuralOptimalTransportPredictor,
    "neural_spline_flow": NeuralSplineFlowPredictor,
    "normalizing_flow": NormalizingFlowPredictor,
}

_BASE_CONFIG_BY_TYPE = {
    "flow_matching": FlowMatchingPredictorConfig,
    "neural_optimal_transport": NeuralOptimalTransportPredictorConfig,
    "neural_spline_flow": NeuralSplineFlowPredictorConfig,
    "normalizing_flow": NormalizingFlowPredictorConfig,
}

_BASE_TRAINER_BY_TYPE = {
    "flow_matching": FlowMatchingTrainer,
    "neural_optimal_transport": NeuralOptimalTransportTrainer,
    "neural_spline_flow": NeuralSplineFlowTrainer,
    "normalizing_flow": NormalizingFlowTrainer,
}


def make_dataset(config: DatasetConfig):
    return _DATASET_BY_TYPE[config.type](config)


def make_base_predictor(stage: BaseStageConfig):
    if stage.predictor is None:
        raise ValueError("A predictor config is required to construct a base model.")
    return _BASE_PREDICTOR_BY_TYPE[stage.type](stage.predictor)


def make_base_trainer(stage: BaseStageConfig):
    if stage.trainer is None:
        raise ValueError("A trainer config is required to construct a base trainer.")
    return _BASE_TRAINER_BY_TYPE[stage.type](stage.trainer)


def load_base_predictor(
    stage: BaseStageConfig,
    path: str | Path,
    map_location: str,
):
    """Load a predictor while making map_location the runtime device."""
    data = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(data, dict) or "config" not in data or "state_dict" not in data:
        raise ValueError("Base predictor checkpoint is missing config or state_dict.")

    config_data = dict(data["config"])
    saved_type = config_data.get("type")
    if saved_type != stage.type:
        raise ValueError(
            f"Checkpoint contains predictor type {saved_type!r}, "
            f"not {stage.type!r}."
        )

    predictor_class = _BASE_PREDICTOR_BY_TYPE[stage.type]
    config_class = _BASE_CONFIG_BY_TYPE[stage.type]
    config_data["device"] = str(torch.device(map_location))
    config = config_class.model_validate(config_data)

    predictor = predictor_class(config)
    predictor.load_state_dict(data["state_dict"])
    predictor.to(device=predictor.device, dtype=predictor.dtype)
    predictor.eval()
    return predictor


def load_base_trainer(
    stage: BaseStageConfig,
    path: str | Path,
    map_location: str,
):
    return _BASE_TRAINER_BY_TYPE[stage.type].load(
        str(path),
        map_location=map_location,
    )


def make_rearrangement_predictor(
    stage: RearrangementStageConfig,
    base_predictor,
):
    if stage.predictor is None:
        raise ValueError("A predictor config is required to construct a rearrangement.")

    if isinstance(stage, AmortizedRearrangedTransportStageConfig):
        return AmortizedRearrangedTransport(stage.predictor, base_predictor)

    return RearrangedTransportPredictor(stage.predictor, base_predictor)


def make_rearrangement_trainer(stage: RearrangementStageConfig):
    if stage.trainer is None:
        raise ValueError(
            "A trainer config is required to construct a rearrangement trainer."
        )

    if isinstance(stage, AmortizedRearrangedTransportStageConfig):
        return AmortizedRearrangedTransportTrainer(stage.trainer)

    if isinstance(stage, SupervisedRearrangedTransportStageConfig):
        return SupervisedRearrangedTransportTrainer(stage.trainer)

    return RearrangedTransportTrainer(stage.trainer)


def load_rearrangement_predictor(
    stage: RearrangementStageConfig,
    path: str | Path,
    map_location: str,
):
    predictor_class = (
        AmortizedRearrangedTransport
        if isinstance(stage, AmortizedRearrangedTransportStageConfig) else
        RearrangedTransportPredictor
    )
    predictor = predictor_class.load(
        str(path),
        map_location=map_location,
    )

    is_amortized = isinstance(predictor, AmortizedRearrangedTransport)
    if is_amortized != isinstance(
        stage,
        AmortizedRearrangedTransportStageConfig,
    ):
        raise ValueError(
            "Rearrangement checkpoint type does not match the configured stage."
        )

    predictor.eval()
    return predictor


def load_rearrangement_trainer(
    stage: RearrangementStageConfig,
    path: str | Path,
    map_location: str,
):
    if isinstance(stage, AmortizedRearrangedTransportStageConfig):
        trainer_class = AmortizedRearrangedTransportTrainer
    elif isinstance(stage, SupervisedRearrangedTransportStageConfig):
        trainer_class = SupervisedRearrangedTransportTrainer
    else:
        trainer_class = RearrangedTransportTrainer

    return trainer_class.load(
        str(path),
        map_location=map_location,
    )
