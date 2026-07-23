from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from configs.calibrators import (
    EllipticCalibratorConfig,
    NoCalibratorConfig,
    NormCalibratorConfig,
)
from configs.conformal import TransportBasedConformalPredictorConfig
from configs.datasets.synthetic import (
    BananaDatasetConfig,
    BimodalGaussianDatasetConfig,
    GaussianDatasetConfig,
    SinusoidalTransportDatasetConfig,
    StudentTDatasetConfig,
)
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
from configs.trainers.rearranged_transport import (
    AmortizedRearrangedTransportTrainerConfig,
    RearrangedTransportTrainerConfig,
    SupervisedRearrangedTransportTrainerConfig,
)
from configs.trainers.transport import (
    FlowMatchingTrainerConfig,
    NeuralOptimalTransportTrainerConfig,
    NeuralSplineFlowTrainerConfig,
    NormalizingFlowTrainerConfig,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


@lru_cache
def _strict_config_class(config_class: type[BaseModel]) -> type[BaseModel]:
    model_config = dict(config_class.model_config)
    model_config["extra"] = "forbid"
    return type(
        f"_Strict{config_class.__name__}",
        (config_class, ),
        {
            "__module__": __name__,
            "model_config": ConfigDict(**model_config),
        },
    )


def _strictly_validate_config(
    config_class: type[BaseModel],
    value,
) -> BaseModel:
    return _strict_config_class(config_class).model_validate(value)


DatasetConfig = Annotated[
    BananaDatasetConfig | BimodalGaussianDatasetConfig | GaussianDatasetConfig
    | SinusoidalTransportDatasetConfig | StudentTDatasetConfig,
    Field(discriminator="type"),
]


class _TrainableStageConfig(_StrictModel):
    mode: Literal["train", "load", "resume"] = "train"
    predictor_checkpoint: Path | None = None
    trainer_checkpoint: Path | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        predictor = getattr(self, "predictor", None)
        trainer = getattr(self, "trainer", None)

        if self.mode == "train":
            if predictor is None or trainer is None:
                raise ValueError(
                    "Train mode requires predictor and trainer configurations."
                )
            if (
                self.predictor_checkpoint is not None
                or self.trainer_checkpoint is not None
            ):
                raise ValueError(
                    "Train mode does not accept predictor or trainer checkpoints."
                )

        elif self.mode == "load":
            if self.predictor_checkpoint is None:
                raise ValueError("Load mode requires predictor_checkpoint.")
            if self.trainer_checkpoint is not None:
                raise ValueError("Load mode does not accept trainer_checkpoint.")
            if predictor is not None or trainer is not None:
                raise ValueError(
                    "Load mode reads configuration from the predictor checkpoint."
                )

        else:
            if (self.predictor_checkpoint is None or self.trainer_checkpoint is None):
                raise ValueError(
                    "Resume mode requires predictor_checkpoint and "
                    "trainer_checkpoint."
                )
            if predictor is not None or trainer is not None:
                raise ValueError(
                    "Resume mode reads configuration from the checkpoints."
                )

        return self


class FlowMatchingStageConfig(_TrainableStageConfig):
    type: Literal["flow_matching"] = "flow_matching"
    predictor: FlowMatchingPredictorConfig | None = None
    trainer: FlowMatchingTrainerConfig | None = None


class NeuralOptimalTransportStageConfig(_TrainableStageConfig):
    type: Literal["neural_optimal_transport"] = "neural_optimal_transport"
    predictor: NeuralOptimalTransportPredictorConfig | None = None
    trainer: NeuralOptimalTransportTrainerConfig | None = None


class NeuralSplineFlowStageConfig(_TrainableStageConfig):
    type: Literal["neural_spline_flow"] = "neural_spline_flow"
    predictor: NeuralSplineFlowPredictorConfig | None = None
    trainer: NeuralSplineFlowTrainerConfig | None = None


class NormalizingFlowStageConfig(_TrainableStageConfig):
    type: Literal["normalizing_flow"] = "normalizing_flow"
    predictor: NormalizingFlowPredictorConfig | None = None
    trainer: NormalizingFlowTrainerConfig | None = None


BaseStageConfig = Annotated[
    FlowMatchingStageConfig | NeuralOptimalTransportStageConfig
    | NeuralSplineFlowStageConfig | NormalizingFlowStageConfig,
    Field(discriminator="type"),
]


class _RearrangementStageConfig(_TrainableStageConfig):

    @model_validator(mode="after")
    def prevent_transport_retraining(self) -> Self:
        trainer = getattr(self, "trainer", None)
        if (
            self.mode == "train" and trainer is not None and trainer.train_transport_map
        ):
            raise ValueError(
                "ExperimentRunner trains the base predictor in its own stage; "
                "rearrangement trainer.train_transport_map must be False."
            )
        return self


class _FixedRearrangementStageConfig(_RearrangementStageConfig):
    trained_coverage_mass: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
    )

    @model_validator(mode="after")
    def validate_declared_coverage(self) -> Self:
        if self.mode != "load" and self.trained_coverage_mass is not None:
            raise ValueError(
                "trained_coverage_mass is only used when loading a fixed "
                "rearrangement predictor."
            )
        return self


class RearrangedTransportStageConfig(_FixedRearrangementStageConfig):
    type: Literal["rearranged_transport"] = "rearranged_transport"
    predictor: RearrangedTransportPredictorConfig | None = None
    trainer: RearrangedTransportTrainerConfig | None = None


class SupervisedRearrangedTransportStageConfig(_FixedRearrangementStageConfig):
    type: Literal["supervised_rearranged_transport"] = "supervised_rearranged_transport"
    predictor: RearrangedTransportPredictorConfig | None = None
    trainer: SupervisedRearrangedTransportTrainerConfig | None = None


class AmortizedRearrangedTransportStageConfig(_RearrangementStageConfig):
    type: Literal["amortized_rearranged_transport"] = "amortized_rearranged_transport"
    predictor: AmortizedRearrangedTransportPredictorConfig | None = None
    trainer: AmortizedRearrangedTransportTrainerConfig | None = None


RearrangementStageConfig = Annotated[
    RearrangedTransportStageConfig | SupervisedRearrangedTransportStageConfig
    | AmortizedRearrangedTransportStageConfig,
    Field(discriminator="type"),
]


class TransportBasedConformalStageConfig(_StrictModel):
    type: Literal["transport_based"] = "transport_based"
    mode: Literal["fit", "load"] = "fit"
    config: TransportBasedConformalPredictorConfig | None = None
    checkpoint: Path | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode == "fit":
            if self.config is None:
                raise ValueError("Fit mode requires a conformal config.")
            if self.checkpoint is not None:
                raise ValueError("Fit mode does not accept a checkpoint.")
        else:
            if self.checkpoint is None:
                raise ValueError("Load mode requires a conformal checkpoint.")
            if self.config is not None:
                raise ValueError(
                    "Load mode reads configuration from the conformal checkpoint."
                )
        return self


ConformalStageConfig = TransportBasedConformalStageConfig

_DATASET_CONFIG_BY_TYPE = {
    "banana": BananaDatasetConfig,
    "bimodal_gaussian": BimodalGaussianDatasetConfig,
    "gaussian_dataset": GaussianDatasetConfig,
    "sinusoidal_transport": SinusoidalTransportDatasetConfig,
    "student_t_dataset": StudentTDatasetConfig,
}

_BASE_COMPONENT_CONFIG_BY_TYPE = {
    "flow_matching": (
        FlowMatchingPredictorConfig,
        FlowMatchingTrainerConfig,
    ),
    "neural_optimal_transport": (
        NeuralOptimalTransportPredictorConfig,
        NeuralOptimalTransportTrainerConfig,
    ),
    "neural_spline_flow": (
        NeuralSplineFlowPredictorConfig,
        NeuralSplineFlowTrainerConfig,
    ),
    "normalizing_flow": (
        NormalizingFlowPredictorConfig,
        NormalizingFlowTrainerConfig,
    ),
}

_REARRANGEMENT_COMPONENT_CONFIG_BY_TYPE = {
    "rearranged_transport": (
        RearrangedTransportPredictorConfig,
        RearrangedTransportTrainerConfig,
    ),
    "supervised_rearranged_transport": (
        RearrangedTransportPredictorConfig,
        SupervisedRearrangedTransportTrainerConfig,
    ),
    "amortized_rearranged_transport": (
        AmortizedRearrangedTransportPredictorConfig,
        AmortizedRearrangedTransportTrainerConfig,
    ),
}

_CALIBRATOR_CONFIG_BY_TYPE = {
    "elliptic": EllipticCalibratorConfig,
    "norm": NormCalibratorConfig,
    "no_calibrator": NoCalibratorConfig,
    "none": NoCalibratorConfig,
}


class DataLoaderConfig(_StrictModel):
    train_batch_size: int = Field(default=256, gt=0)
    calibration_batch_size: int = Field(default=512, gt=0)
    test_batch_size: int = Field(default=512, gt=0)
    shuffle_train: bool = True
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False


class ArtifactConfig(_StrictModel):
    root_directory: Path = Path("experiments")
    checkpoint_every_epochs: int | None = Field(default=10, gt=0)


class EvaluationConfig(_StrictModel):
    compute_volume: bool = False


class ExperimentConfig(_StrictModel):
    """Complete configuration for one experiment run."""

    name: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    checkpoint_map_location: str = "cpu"

    dataset: DatasetConfig
    base: BaseStageConfig
    rearrangement: RearrangementStageConfig | None = None
    conformal: ConformalStageConfig

    dataloaders: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="before")
    @classmethod
    def forbid_unknown_nested_fields(cls, data):
        """Apply strict validation to the repository's reusable configs too."""
        if not isinstance(data, Mapping):
            return data

        validated = dict(data)

        dataset = validated.get("dataset")
        if isinstance(dataset, Mapping):
            config_class = _DATASET_CONFIG_BY_TYPE.get(dataset.get("type"))
            if config_class is not None:
                validated["dataset"] = _strictly_validate_config(
                    config_class,
                    dataset,
                )

        base = validated.get("base")
        validated["base"] = cls._strictly_validate_trainable_stage(
            stage=base,
            component_configs=_BASE_COMPONENT_CONFIG_BY_TYPE,
            annotation=BaseStageConfig,
        )

        rearrangement = validated.get("rearrangement")
        if rearrangement is not None:
            validated["rearrangement"] = cls._strictly_validate_trainable_stage(
                stage=rearrangement,
                component_configs=_REARRANGEMENT_COMPONENT_CONFIG_BY_TYPE,
                annotation=RearrangementStageConfig,
            )

        conformal = validated.get("conformal")
        if isinstance(conformal, Mapping):
            conformal_data = dict(conformal)
            conformal_config = conformal_data.get("config")
            if isinstance(conformal_config, Mapping):
                conformal_config_data = dict(conformal_config)
                for calibrator_field in (
                    "calibrator",
                    "calibrator_config",
                ):
                    calibrator = conformal_config_data.get(calibrator_field)
                    if isinstance(calibrator, Mapping):
                        config_class = _CALIBRATOR_CONFIG_BY_TYPE.get(
                            calibrator.get("type")
                        )
                        if config_class is not None:
                            strict_calibrator = _strictly_validate_config(
                                config_class,
                                calibrator,
                            )
                            conformal_config_data[calibrator_field] = strict_calibrator
                conformal_data["config"] = _strictly_validate_config(
                    TransportBasedConformalPredictorConfig,
                    conformal_config_data,
                )
            validated["conformal"] = (
                TransportBasedConformalStageConfig.model_validate(conformal_data)
            )

        return validated

    @staticmethod
    def _strictly_validate_trainable_stage(
        stage,
        component_configs: dict[
            str,
            tuple[type[BaseModel], type[BaseModel]],
        ],
        annotation,
    ):
        if isinstance(stage, Mapping):
            stage_data = dict(stage)
            config_classes = component_configs.get(stage_data.get("type"))
            if config_classes is not None:
                for field, config_class in zip(
                    ("predictor", "trainer"),
                    config_classes,
                    strict=True,
                ):
                    value = stage_data.get(field)
                    if value is not None:
                        stage_data[field] = _strictly_validate_config(
                            config_class,
                            value,
                        )
            stage = stage_data

        return TypeAdapter(annotation).validate_python(stage)

    @property
    def run_directory(self) -> Path:
        return self.artifacts.root_directory / self.name

    @model_validator(mode="after")
    def validate_train_dimensions_and_coverage(self) -> Self:
        if self.dataset.n_train < 1:
            raise ValueError("Dataset n_train must be positive.")
        if self.dataset.n_calibration < 0:
            raise ValueError("Dataset n_calibration cannot be negative.")
        if self.dataset.n_test < 1:
            raise ValueError("Dataset n_test must be positive.")

        if (
            self.conformal.mode == "fit" and not isinstance(
                self.conformal.config.calibrator,
                NoCalibratorConfig,
            ) and self.dataset.n_calibration < 1
        ):
            raise ValueError(
                "Empirical conformal calibration requires a non-empty "
                "calibration split."
            )

        if self.base.mode == "train":
            self._validate_dimensions(
                stage_name="base predictor",
                x_dim=self.base.predictor.x_dim,
                y_dim=self.base.predictor.y_dim,
            )

        if self.rearrangement is not None and self.rearrangement.mode == "train":
            self._validate_dimensions(
                stage_name="rearrangement predictor",
                x_dim=self.rearrangement.predictor.x_dim,
                y_dim=self.rearrangement.predictor.y_dim,
            )

            if (
                self.base.mode == "train"
                and self.base.predictor.dtype != self.rearrangement.predictor.dtype
            ):
                raise ValueError("Base and rearrangement predictor dtypes must match.")

            if (
                self.base.mode == "train" and isinstance(
                    self.base,
                    NeuralOptimalTransportStageConfig,
                ) and self.base.predictor.potential_type == "y"
            ):
                raise ValueError(
                    "Neural optimal transport with potential_type='y' cannot "
                    "train a rearrangement because log_det is unavailable."
                )

            if (
                not isinstance(
                    self.rearrangement,
                    AmortizedRearrangedTransportStageConfig,
                ) and self.conformal.mode == "fit" and not math.isclose(
                    self.rearrangement.trainer.coverage_mass,
                    self.conformal.config.coverage_mass,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Fixed rearrangement and conformal coverage_mass values "
                    "must match."
                )

        if (
            isinstance(
                self.rearrangement,
                (
                    RearrangedTransportStageConfig,
                    SupervisedRearrangedTransportStageConfig,
                ),
            ) and self.rearrangement.mode == "load"
            and self.rearrangement.trained_coverage_mass is not None
            and self.conformal.mode == "fit" and not math.isclose(
                self.rearrangement.trained_coverage_mass,
                self.conformal.config.coverage_mass,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Fixed rearrangement and conformal coverage_mass values "
                "must match."
            )

        if (
            self.evaluation.compute_volume and self.conformal.mode == "fit"
            and not self._supports_volume(self.conformal.config)
        ):
            raise ValueError(
                "Volume evaluation supports only NoCalibrator or an L2 "
                "NormCalibrator."
            )

        if (
            self.evaluation.compute_volume and self.rearrangement is None
            and self.base.mode == "train" and isinstance(
                self.base,
                NeuralOptimalTransportStageConfig,
            ) and self.base.predictor.potential_type == "y"
        ):
            raise ValueError(
                "Volume evaluation is unavailable for neural optimal "
                "transport with potential_type='y'."
            )

        return self

    def _validate_dimensions(
        self,
        stage_name: str,
        x_dim: int,
        y_dim: int,
    ) -> None:
        if x_dim != self.dataset.x_dim or y_dim != self.dataset.y_dim:
            raise ValueError(
                f"{stage_name} dimensions ({x_dim}, {y_dim}) do not match "
                f"dataset dimensions ({self.dataset.x_dim}, "
                f"{self.dataset.y_dim})."
            )

    @staticmethod
    def _supports_volume(config: TransportBasedConformalPredictorConfig, ) -> bool:
        calibrator = config.calibrator
        return isinstance(calibrator, NoCalibratorConfig) or (
            isinstance(calibrator, NormCalibratorConfig) and math.isclose(
                float(calibrator.p),
                2.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one JSON or YAML experiment configuration."""
    path = Path(path)
    suffix = path.suffix.lower()

    with path.open("r", encoding="utf-8") as file:
        if suffix == ".json":
            data = json.load(file)
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(file)
        else:
            raise ValueError("Experiment configuration must use .json, .yaml, or .yml.")

    if not isinstance(data, dict):
        raise TypeError("Experiment configuration must contain a mapping.")

    return ExperimentConfig.model_validate(data)
