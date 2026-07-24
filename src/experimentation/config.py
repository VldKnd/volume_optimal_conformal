from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, Field

from configs.conformal import TransportBasedConformalPredictorConfig
from configs.datasets import synthetic as dataset_configs
from configs.predictors import rearranged_transport as rearrangement_configs
from configs.predictors import transport as predictor_configs

DatasetConfig = Annotated[
    dataset_configs.BananaDatasetConfig
    | dataset_configs.BimodalGaussianDatasetConfig
    | dataset_configs.GaussianDatasetConfig
    | dataset_configs.SinusoidalTransportDatasetConfig
    | dataset_configs.StudentTDatasetConfig,
    Field(discriminator="type"),
]

PredictorConfig = Annotated[
    predictor_configs.ConvexPotentialFlowPredictorConfig
    | predictor_configs.FlowMatchingPredictorConfig
    | predictor_configs.NeuralOptimalTransportPredictorConfig
    | predictor_configs.NeuralSplineFlowPredictorConfig
    | predictor_configs.NormalizingFlowPredictorConfig,
    Field(discriminator="type"),
]

RearrangementConfig = Annotated[
    rearrangement_configs.RearrangedTransportPredictorConfig
    | rearrangement_configs.AmortizedRearrangedTransportPredictorConfig,
    Field(discriminator="type"),
]


class ExperimentConfig(BaseModel):
    """Configuration for one experimental training and evaluation pipeline."""

    name: str
    seed: int = 0
    save_directory: Path = Path("experiments")

    dataset_config: DatasetConfig

    predictor_config: PredictorConfig
    trainer_config: Any = None
    predictor_checkpoint: Path | None = None

    rearrangement_config: RearrangementConfig | None = None
    rearrangement_trainer_config: Any = None
    rearrangement_checkpoint: Path | None = None
    supervised_rearrangement: bool = False

    conformal_config: TransportBasedConformalPredictorConfig
    conformal_checkpoint: Path | None = None

    train_batch_size: int = Field(default=256, gt=0)
    calibration_batch_size: int = Field(default=512, gt=0)
    test_batch_size: int = Field(default=512, gt=0)
    compute_volume: bool = False

    @property
    def run_directory(self) -> Path:
        return self.save_directory / self.name


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            data = json.load(file)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(file)
        else:
            raise ValueError("Experiment config must be JSON or YAML.")

    return ExperimentConfig.model_validate(data)
