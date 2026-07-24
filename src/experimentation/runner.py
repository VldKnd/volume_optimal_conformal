from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from configs.conformal import TransportBasedConformalPredictorConfig
from configs.trainers import rearranged_transport as rearranged_configs
from configs.trainers import transport as trainer_configs
from conformal import TransportBasedConformalPredictor
from data.datasets import synthetic as datasets
from data.loaders import make_xy_dataloader
from experimentation.config import ExperimentConfig
from predictors import rearranged_transport as rearranged_predictors
from predictors import transport as transport_predictors
from trainers import rearranged_transport as rearranged_trainers
from trainers import transport as transport_trainers


class ExperimentRunner:
    """Run base transport -> optional rearrangement -> conformal calibration."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.run_directory = config.run_directory

        self.dataset = None
        self.predictor = None
        self.rearrangement = None
        self.conformal_predictor = None
        self.histories: dict[str, list[dict]] = {}
        self.metrics: dict[str, float | int] = {}

    def run(self) -> ExperimentRunner:
        config = self.config
        predictor_config = config.predictor_config

        if predictor_config.type == "convex_potential_flow":
            predictor_class = transport_predictors.ConvexPotentialFlowPredictor
            trainer_class = transport_trainers.ConvexPotentialFlowTrainer
            trainer_config_class = (trainer_configs.ConvexPotentialFlowTrainerConfig)
        elif predictor_config.type == "flow_matching":
            predictor_class = transport_predictors.FlowMatchingPredictor
            trainer_class = transport_trainers.FlowMatchingTrainer
            trainer_config_class = trainer_configs.FlowMatchingTrainerConfig
        elif predictor_config.type == "neural_optimal_transport":
            predictor_class = transport_predictors.NeuralOptimalTransportPredictor
            trainer_class = transport_trainers.NeuralOptimalTransportTrainer
            trainer_config_class = (trainer_configs.NeuralOptimalTransportTrainerConfig)
        elif predictor_config.type == "neural_spline_flow":
            predictor_class = transport_predictors.NeuralSplineFlowPredictor
            trainer_class = transport_trainers.NeuralSplineFlowTrainer
            trainer_config_class = trainer_configs.NeuralSplineFlowTrainerConfig
        else:
            predictor_class = transport_predictors.NormalizingFlowPredictor
            trainer_class = transport_trainers.NormalizingFlowTrainer
            trainer_config_class = trainer_configs.NormalizingFlowTrainerConfig

        trainer_config = None
        if config.trainer_config is not None:
            trainer_data = config.trainer_config
            if hasattr(trainer_data, "model_dump"):
                trainer_data = trainer_data.model_dump()
            trainer_config = trainer_config_class.model_validate(trainer_data)

        rearrangement_class = None
        rearrangement_trainer_class = None
        rearrangement_trainer_config = None
        if config.rearrangement_config is not None:
            if config.rearrangement_config.type == "amortized_rearranged_transport":
                rearrangement_class = (
                    rearranged_predictors.AmortizedRearrangedTransport
                )
                rearrangement_trainer_class = (
                    rearranged_trainers.AmortizedRearrangedTransportTrainer
                )
                rearrangement_trainer_config_class = (
                    rearranged_configs.AmortizedRearrangedTransportTrainerConfig
                )
            else:
                rearrangement_class = (
                    rearranged_predictors.RearrangedTransportPredictor
                )
                if config.supervised_rearrangement:
                    rearrangement_trainer_class = (
                        rearranged_trainers.SupervisedRearrangedTransportTrainer
                    )
                    rearrangement_trainer_config_class = (
                        rearranged_configs.SupervisedRearrangedTransportTrainerConfig
                    )
                else:
                    rearrangement_trainer_class = (
                        rearranged_trainers.RearrangedTransportTrainer
                    )
                    rearrangement_trainer_config_class = (
                        rearranged_configs.RearrangedTransportTrainerConfig
                    )

            if config.rearrangement_trainer_config is not None:
                trainer_data = config.rearrangement_trainer_config
                if hasattr(trainer_data, "model_dump"):
                    trainer_data = trainer_data.model_dump()
                rearrangement_trainer_config = (
                    rearrangement_trainer_config_class.model_validate(trainer_data)
                )

        dataset_config = config.dataset_config
        if (
            predictor_config.x_dim != dataset_config.x_dim
            or predictor_config.y_dim != dataset_config.y_dim
        ):
            raise ValueError("Predictor and dataset dimensions must match.")

        if config.rearrangement_config is not None:
            rearrangement_config = config.rearrangement_config
            if (
                rearrangement_config.x_dim != predictor_config.x_dim
                or rearrangement_config.y_dim != predictor_config.y_dim
                or rearrangement_config.dtype != predictor_config.dtype
            ):
                raise ValueError(
                    "Base and rearrangement dimensions and dtypes must match."
                )
            if (
                rearrangement_trainer_config is not None
                and rearrangement_trainer_config.train_transport_map
            ):
                raise ValueError(
                    "The rearrangement stage must not retrain the base predictor."
                )
            if (
                rearrangement_config.type == "rearranged_transport"
                and not math.isclose(
                    rearrangement_trainer_config.coverage_mass,
                    config.conformal_config.coverage_mass,
                )
            ):
                raise ValueError(
                    "Fixed rearrangement and conformal coverage masses must match."
                )

        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.run_directory / "config.json",
            config.model_dump(mode="json"),
        )
        self._seed(config.seed)

        if dataset_config.type == "banana":
            self.dataset = datasets.BananaDataset(dataset_config)
        elif dataset_config.type == "bimodal_gaussian":
            self.dataset = datasets.BimodalGaussianDataset(dataset_config)
        elif dataset_config.type == "gaussian_dataset":
            self.dataset = datasets.GaussianDatasetTarget(dataset_config)
        elif dataset_config.type == "student_t_dataset":
            self.dataset = datasets.StudentTDataset(dataset_config)
        else:
            self.dataset = datasets.SinusoidalTransportDataset(dataset_config)

        splits = self.dataset.get_splits()
        train_loader = make_xy_dataloader(
            splits.train,
            batch_size=config.train_batch_size,
            shuffle=True,
        )
        calibration_loader = make_xy_dataloader(
            splits.calibration,
            batch_size=config.calibration_batch_size,
            shuffle=False,
        )
        test_loader = make_xy_dataloader(
            splits.test,
            batch_size=config.test_batch_size,
            shuffle=False,
        )

        base_trainer = None
        self.predictor = predictor_class(predictor_config)
        if config.predictor_checkpoint is not None:
            checkpoint = torch.load(
                config.predictor_checkpoint,
                map_location=predictor_config.device,
                weights_only=False,
            )
            self.predictor.load_state_dict(checkpoint["state_dict"])
        else:
            base_trainer = trainer_class(trainer_config)
            self._seed(config.seed)
            base_trainer.fit(self.predictor, train_loader)
        self.predictor.eval()
        self._save_stage("base", self.predictor, base_trainer)

        final_predictor = self.predictor
        rearrangement_trainer = None
        if config.rearrangement_config is not None:
            self.rearrangement = rearrangement_class(
                config.rearrangement_config,
                self.predictor,
            )
            if config.rearrangement_checkpoint is not None:
                checkpoint = torch.load(
                    config.rearrangement_checkpoint,
                    map_location=config.rearrangement_config.device,
                    weights_only=False,
                )
                self.rearrangement.rearrangement_flow.load_state_dict(
                    checkpoint["rearrangement_state_dict"]
                )
            else:
                rearrangement_trainer = rearrangement_trainer_class(
                    rearrangement_trainer_config
                )
                self._seed(config.seed)
                rearrangement_trainer.fit(self.rearrangement, train_loader)
            self.rearrangement.eval()
            self._save_stage(
                "rearrangement",
                self.rearrangement,
                rearrangement_trainer,
            )
            final_predictor = self.rearrangement

        if config.conformal_checkpoint is None:
            self.conformal_predictor = TransportBasedConformalPredictor(
                predictor=final_predictor,
                config=config.conformal_config,
            )
            self.conformal_predictor.fit(calibration_loader)
        else:
            checkpoint = torch.load(
                config.conformal_checkpoint,
                map_location=predictor_config.device,
                weights_only=False,
            )
            saved_config = TransportBasedConformalPredictorConfig.model_validate(
                checkpoint["config"]
            )
            if saved_config != config.conformal_config:
                raise ValueError(
                    "Loaded conformal checkpoint has a different configuration."
                )
            self.conformal_predictor = TransportBasedConformalPredictor(
                predictor=final_predictor,
                config=saved_config,
            )
            self.conformal_predictor.calibrator = checkpoint["calibrator"]

        conformal_directory = self.run_directory / "conformal"
        conformal_directory.mkdir(exist_ok=True)
        torch.save(
            {
                "config": self.conformal_predictor.config.model_dump(),
                "calibrator": self.conformal_predictor.calibrator,
            },
            conformal_directory / "predictor.pt",
        )

        total = 0
        contained = 0
        volumes = []
        for x_batch, y_batch in test_loader:
            inside = self.conformal_predictor.contains(x=x_batch, y=y_batch)
            total += inside.numel()
            contained += int(inside.sum().item())
            if config.compute_volume:
                volumes.append(self.conformal_predictor.volume(x_batch).detach().cpu())

        if total == 0:
            raise ValueError("The test split is empty.")

        empirical_coverage = contained / total
        coverage_error = empirical_coverage - self.conformal_predictor.coverage_mass
        self.metrics = {
            "n_test": total,
            "target_coverage": self.conformal_predictor.coverage_mass,
            "empirical_coverage": empirical_coverage,
            "coverage_error": coverage_error,
        }
        if volumes:
            volume = torch.cat(volumes)
            self.metrics["mean_volume"] = float(volume.mean())
            self.metrics["std_volume"] = float(volume.std(unbiased=False))

        self._write_json(self.run_directory / "metrics.json", self.metrics)
        return self

    def _save_stage(self, name, predictor, trainer) -> None:
        directory = self.run_directory / name
        directory.mkdir(exist_ok=True)
        predictor.save(str(directory / "predictor.pt"))
        history = []
        if trainer is not None:
            trainer.save(str(directory / "trainer.pt"))
            history = trainer.training_history
        self.histories[name] = list(history)
        self._write_json(directory / "history.json", history)

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _write_json(path: Path, data) -> None:
        path.write_text(
            json.dumps(data, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
