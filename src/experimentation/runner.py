from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from conformal import TransportBasedConformalPredictor
from conformal.calibrators.elliptic_calibrator import EllipticCalibrator
from data.datasets.base import BaseDataset
from data.loaders import make_xy_dataloader
from experimentation.config import (
    AmortizedRearrangedTransportStageConfig,
    ExperimentConfig,
    load_experiment_config,
)
from experimentation.factories import (
    load_base_predictor,
    load_base_trainer,
    load_rearrangement_predictor,
    load_rearrangement_trainer,
    make_base_predictor,
    make_base_trainer,
    make_dataset,
    make_rearrangement_predictor,
    make_rearrangement_trainer,
)
from predictors.base import BasePredictor
from trainers.base import BaseTrainer


@dataclass(frozen=True)
class ExperimentResult:
    run_directory: Path
    dataset: BaseDataset
    base_predictor: BasePredictor
    final_predictor: BasePredictor
    conformal_predictor: TransportBasedConformalPredictor
    base_trainer: BaseTrainer | None
    rearrangement_trainer: BaseTrainer | None
    histories: dict[str, list[dict[str, Any]]]
    metrics: dict[str, Any]


class ExperimentRunner:
    """Run one base -> optional rearrangement -> conformal experiment."""

    conformal_checkpoint_version = 1
    training_checkpoint_version = 1
    training_checkpoint_metadata_name = "checkpoint.json"

    def __init__(self, config: ExperimentConfig | Mapping[str, Any]):
        if not isinstance(config, ExperimentConfig):
            config = ExperimentConfig.model_validate(config)

        self.config = config
        self.run_directory = config.run_directory

        self.dataset: BaseDataset | None = None
        self.base_predictor: BasePredictor | None = None
        self.final_predictor: BasePredictor | None = None
        self.conformal_predictor: TransportBasedConformalPredictor | None = None
        self.base_trainer: BaseTrainer | None = None
        self.rearrangement_trainer: BaseTrainer | None = None
        self.metrics: dict[str, Any] = {}
        self.histories: dict[str, list[dict[str, Any]]] = {}
        self._manifest: dict[str, Any] = {}
        self._has_run = False
        self._training_context_fingerprint: str | None = None
        self._rearrangement_coverage_mass: float | None = None

    @classmethod
    def from_config_file(cls, path: str | Path) -> ExperimentRunner:
        return cls(load_experiment_config(path))

    def run(self) -> ExperimentResult:
        if self._has_run:
            raise RuntimeError("An ExperimentRunner instance can only be run once.")
        self._has_run = True

        self._prepare_run_directory()
        self._start_manifest()

        try:
            started = time.perf_counter()
            self._preflight_source_checkpoints()
            self._seed_everything(self.config.seed)

            self.dataset = make_dataset(self.config.dataset)
            splits = self.dataset.get_splits()
            loaders = self._make_dataloaders(splits)
            self._training_context_fingerprint = (
                self._make_training_context_fingerprint(splits.train)
            )

            self.base_predictor, self.base_trainer = self._run_base_stage(
                dataloader=loaders["train"],
            )

            self.final_predictor = self.base_predictor
            if self.config.rearrangement is not None:
                (
                    self.final_predictor,
                    self.rearrangement_trainer,
                ) = self._run_rearrangement_stage(
                    base_predictor=self.base_predictor,
                    dataloader=loaders["train"],
                )

            self.conformal_predictor = self._run_conformal_stage(
                predictor=self.final_predictor,
                dataloader=loaders["calibration"],
            )
            self._seed_everything(self.config.seed)
            self.metrics = self._evaluate(
                conformal_predictor=self.conformal_predictor,
                dataloader=loaders["test"],
            )
            self.metrics.update(
                base_trainable_parameter_count=self._parameter_count(
                    self.base_predictor
                ),
                final_trainable_parameter_count=self._parameter_count(
                    self.final_predictor
                ),
            )

            self._write_json(
                self.run_directory / "metrics.json",
                self.metrics,
            )
            self._manifest.update(
                status="complete",
                finished_at=self._timestamp(),
                total_duration_seconds=time.perf_counter() - started,
                metrics=self.metrics,
            )
            self._write_manifest()

        except Exception as error:
            self._manifest.update(
                status="failed",
                finished_at=self._timestamp(),
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
            self._write_manifest()
            raise

        return ExperimentResult(
            run_directory=self.run_directory,
            dataset=self.dataset,
            base_predictor=self.base_predictor,
            final_predictor=self.final_predictor,
            conformal_predictor=self.conformal_predictor,
            base_trainer=self.base_trainer,
            rearrangement_trainer=self.rearrangement_trainer,
            histories=self.histories,
            metrics=self.metrics,
        )

    def _run_base_stage(self, dataloader):
        stage = self.config.base
        stage_directory = self.run_directory / "stages" / "base"
        started = time.perf_counter()
        loaded_history: list[dict[str, Any]] = []

        if stage.mode == "train":
            predictor = make_base_predictor(stage)
            self._validate_predictor_dimensions(predictor, stage_name="base")
            self._validate_base_for_planned_pipeline(predictor)
            trainer = make_base_trainer(stage)
            self._seed_everything(self.config.seed)
            self._fit_and_checkpoint(
                predictor=predictor,
                trainer=trainer,
                dataloader=dataloader,
                stage_directory=stage_directory,
            )

        elif stage.mode == "load":
            metadata = self._prevalidate_completed_predictor_checkpoint(
                stage.predictor_checkpoint
            )
            predictor = load_base_predictor(
                stage,
                stage.predictor_checkpoint,
                map_location=self.config.checkpoint_map_location,
            )
            self._validate_loaded_predictor_metadata(
                metadata=metadata,
                predictor=predictor,
            )
            self._validate_predictor_dimensions(predictor, stage_name="base")
            self._validate_base_for_planned_pipeline(predictor)
            trainer = None
            loaded_history = self._load_source_history(
                stage.predictor_checkpoint,
                metadata,
            )
            self._save_predictor(predictor, stage_directory / "predictor.pt")
            self._write_json(stage_directory / "history.json", loaded_history)
            self._write_loaded_stage_metadata(
                stage_directory=stage_directory,
                predictor=predictor,
                source_metadata=metadata,
            )

        else:
            metadata = self._validate_resume_checkpoint_pair(
                predictor_path=stage.predictor_checkpoint,
                trainer_path=stage.trainer_checkpoint,
            )
            predictor = load_base_predictor(
                stage,
                stage.predictor_checkpoint,
                map_location=self.config.checkpoint_map_location,
            )
            self._validate_predictor_dimensions(predictor, stage_name="base")
            trainer = load_base_trainer(
                stage,
                stage.trainer_checkpoint,
                map_location=self.config.checkpoint_map_location,
            )
            self._validate_resume_objects(
                metadata=metadata,
                predictor=predictor,
                trainer=trainer,
            )
            self._validate_base_for_planned_pipeline(predictor)
            self._seed_everything(self.config.seed)
            self._fit_and_checkpoint(
                predictor=predictor,
                trainer=trainer,
                dataloader=dataloader,
                stage_directory=stage_directory,
            )

        history = loaded_history if trainer is None else trainer.training_history
        self.histories["base"] = list(history)
        self._record_stage(
            "base",
            mode=stage.mode,
            duration=time.perf_counter() - started,
            predictor_checkpoint=stage.predictor_checkpoint,
            trainer_checkpoint=stage.trainer_checkpoint,
        )
        return predictor, trainer

    def _run_rearrangement_stage(self, base_predictor, dataloader):
        stage = self.config.rearrangement
        stage_directory = self.run_directory / "stages" / "rearrangement"
        started = time.perf_counter()
        loaded_history: list[dict[str, Any]] = []

        if stage.mode == "train":
            self._validate_rearrangement_training_input(
                stage=stage,
                base_predictor=base_predictor,
            )
            predictor = make_rearrangement_predictor(stage, base_predictor)
            self._validate_predictor_dimensions(
                predictor,
                stage_name="rearrangement",
            )
            trainer = make_rearrangement_trainer(stage)
            self._set_rearrangement_coverage_mass(stage, trainer)
            self._validate_configured_conformal_coverage()
            self._seed_everything(self.config.seed)
            self._fit_and_checkpoint(
                predictor=predictor,
                trainer=trainer,
                dataloader=dataloader,
                stage_directory=stage_directory,
            )

        elif stage.mode == "load":
            metadata = self._prevalidate_completed_predictor_checkpoint(
                stage.predictor_checkpoint
            )
            predictor = load_rearrangement_predictor(
                stage,
                stage.predictor_checkpoint,
                map_location=self.config.checkpoint_map_location,
            )
            self._validate_predictor_dimensions(
                predictor,
                stage_name="rearrangement",
            )
            self._validate_loaded_predictor_metadata(
                metadata=metadata,
                predictor=predictor,
            )
            self._require_matching_base(
                expected=base_predictor,
                actual=predictor.transport_predictor,
            )
            self._rebind_rearrangement_base(
                predictor=predictor,
                base_predictor=base_predictor,
            )
            self._set_loaded_rearrangement_coverage_mass(stage, metadata)
            self._validate_configured_conformal_coverage()
            trainer = None
            loaded_history = self._load_source_history(
                stage.predictor_checkpoint,
                metadata,
            )
            self._save_predictor(predictor, stage_directory / "predictor.pt")
            self._write_json(stage_directory / "history.json", loaded_history)
            self._write_loaded_stage_metadata(
                stage_directory=stage_directory,
                predictor=predictor,
                source_metadata=metadata,
                trained_coverage_mass=self._rearrangement_coverage_mass,
            )

        else:
            metadata = self._validate_resume_checkpoint_pair(
                predictor_path=stage.predictor_checkpoint,
                trainer_path=stage.trainer_checkpoint,
            )
            predictor = load_rearrangement_predictor(
                stage,
                stage.predictor_checkpoint,
                map_location=self.config.checkpoint_map_location,
            )
            self._validate_predictor_dimensions(
                predictor,
                stage_name="rearrangement",
            )
            self._require_matching_base(
                expected=base_predictor,
                actual=predictor.transport_predictor,
            )
            trainer = load_rearrangement_trainer(
                stage,
                stage.trainer_checkpoint,
                map_location=self.config.checkpoint_map_location,
            )
            self._validate_resume_objects(
                metadata=metadata,
                predictor=predictor,
                trainer=trainer,
            )
            if trainer.config.train_transport_map:
                raise ValueError(
                    "ExperimentRunner resumes the base predictor separately; "
                    "the rearrangement trainer checkpoint must have "
                    "train_transport_map=False."
                )
            self._rebind_rearrangement_base(
                predictor=predictor,
                base_predictor=base_predictor,
            )
            self._set_rearrangement_coverage_mass(stage, trainer)
            self._validate_configured_conformal_coverage()
            self._seed_everything(self.config.seed)
            self._fit_and_checkpoint(
                predictor=predictor,
                trainer=trainer,
                dataloader=dataloader,
                stage_directory=stage_directory,
            )

        history = loaded_history if trainer is None else trainer.training_history
        self.histories["rearrangement"] = list(history)
        self._record_stage(
            "rearrangement",
            mode=stage.mode,
            duration=time.perf_counter() - started,
            predictor_checkpoint=stage.predictor_checkpoint,
            trainer_checkpoint=stage.trainer_checkpoint,
        )
        return predictor, trainer

    def _run_conformal_stage(self, predictor, dataloader):
        stage = self.config.conformal
        stage_directory = self.run_directory / "stages" / "conformal"
        checkpoint_path = stage_directory / "predictor.pt"
        started = time.perf_counter()

        if stage.mode == "fit":
            conformal_predictor = TransportBasedConformalPredictor(
                predictor=predictor,
                config=stage.config,
            )
            self._validate_rearrangement_coverage_mass(
                conformal_predictor.coverage_mass
            )
            self._validate_volume_support(conformal_predictor)
            conformal_predictor.fit(dataloader)
            self._save_conformal_predictor(
                conformal_predictor,
                checkpoint_path,
            )
        else:
            conformal_predictor = self._load_conformal_predictor(
                predictor=predictor,
                path=stage.checkpoint,
            )
            self._validate_rearrangement_coverage_mass(
                conformal_predictor.coverage_mass
            )
            self._validate_volume_support(conformal_predictor)
            self._save_conformal_predictor(
                conformal_predictor,
                checkpoint_path,
            )

        self._record_stage(
            "conformal",
            mode=stage.mode,
            duration=time.perf_counter() - started,
            predictor_checkpoint=stage.checkpoint,
        )
        return conformal_predictor

    def _fit_and_checkpoint(
        self,
        predictor,
        trainer,
        dataloader,
        stage_directory: Path,
    ) -> None:
        checkpoint_interval = self.config.artifacts.checkpoint_every_epochs

        if trainer.completed_epochs >= trainer.config.epochs:
            self._save_training_stage(
                predictor=predictor,
                trainer=trainer,
                stage_directory=stage_directory,
            )
            return

        while trainer.completed_epochs < trainer.config.epochs:
            before = trainer.completed_epochs
            max_epochs = (
                None if checkpoint_interval is None else min(
                    checkpoint_interval,
                    trainer.config.epochs - trainer.completed_epochs,
                )
            )
            trainer.fit(
                predictor,
                dataloader,
                max_epochs=max_epochs,
            )
            if trainer.completed_epochs <= before:
                raise RuntimeError(
                    "Trainer made no progress before reaching its configured "
                    "epoch count."
                )
            self._save_training_stage(
                predictor=predictor,
                trainer=trainer,
                stage_directory=stage_directory,
            )

    def _save_training_stage(
        self,
        predictor,
        trainer,
        stage_directory: Path,
    ) -> None:
        self._save_predictor(
            predictor,
            stage_directory / "predictor.pt",
        )
        self._save_trainer(
            trainer,
            stage_directory / "trainer.pt",
        )
        self._write_json(
            stage_directory / "history.json",
            trainer.training_history,
        )
        self._write_training_checkpoint_metadata(
            predictor=predictor,
            trainer=trainer,
            stage_directory=stage_directory,
        )

    def _evaluate(self, conformal_predictor, dataloader) -> dict[str, Any]:
        total = 0
        contained = 0
        volume_batches: list[torch.Tensor] = []
        threshold = float(conformal_predictor.threshold.detach().cpu())
        compute_volume = self.config.evaluation.compute_volume
        finite_volume_radius = math.isfinite(threshold)

        if math.isnan(threshold):
            raise ValueError("The calibrated threshold is NaN.")

        for x_batch, y_batch in dataloader:
            inside = conformal_predictor.contains(
                x=x_batch,
                y=y_batch,
            )
            total += int(inside.numel())
            contained += int(inside.sum().item())

            if compute_volume and finite_volume_radius:
                volume = conformal_predictor.volume(x=x_batch)
                volume_batches.append(
                    volume.detach().to(device="cpu", dtype=torch.float64)
                )

        if total == 0:
            raise ValueError("Test split must contain at least one observation.")

        coverage = contained / total
        target = conformal_predictor.coverage_mass
        metrics: dict[str, Any] = {
            "n_test": total,
            "coverage_mass": target,
            "empirical_coverage": coverage,
            "coverage_error": coverage - target,
            "absolute_coverage_error": abs(coverage - target),
            "threshold": threshold if finite_volume_radius else None,
            "threshold_is_infinite": math.isinf(threshold),
        }

        if volume_batches:
            volumes = torch.cat(volume_batches)
            metrics["volume"] = {
                "is_infinite": False,
                "mean": float(volumes.mean()),
                "std": float(volumes.std(unbiased=False)),
                "min": float(volumes.min()),
                "max": float(volumes.max()),
            }
        elif compute_volume:
            metrics["volume"] = {
                "is_infinite": True,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

        return metrics

    def _make_dataloaders(self, splits) -> dict[str, Any]:
        config = self.config.dataloaders
        common = {
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
        }
        return {
            "train":
            make_xy_dataloader(
                splits.train,
                batch_size=config.train_batch_size,
                shuffle=config.shuffle_train,
                **common,
            ),
            "calibration":
            make_xy_dataloader(
                splits.calibration,
                batch_size=config.calibration_batch_size,
                shuffle=False,
                **common,
            ),
            "test":
            make_xy_dataloader(
                splits.test,
                batch_size=config.test_batch_size,
                shuffle=False,
                **common,
            ),
        }

    def _save_conformal_predictor(
        self,
        predictor: TransportBasedConformalPredictor,
        path: Path,
    ) -> None:
        calibrator = predictor.calibrator
        state_names = ["threshold"]
        if isinstance(calibrator, EllipticCalibrator):
            state_names.extend(
                [
                    "x_train",
                    "score_train",
                    "global_covariance",
                    "global_inverse_covariance",
                ]
            )

        state = {
            name: self._checkpoint_value(getattr(calibrator, name))
            for name in state_names
        }
        data = {
            "checkpoint_kind": "transport_based_conformal_predictor",
            "format_version": self.conformal_checkpoint_version,
            "config": predictor.config.model_dump(),
            "predictor_fingerprint": self._predictor_fingerprint(predictor.predictor),
            "calibrator_state": state,
        }
        self._atomic_torch_save(data, path)

    def _load_conformal_predictor(
        self,
        predictor,
        path: str | Path,
    ) -> TransportBasedConformalPredictor:
        data = torch.load(
            path,
            map_location=self.config.checkpoint_map_location,
            weights_only=False,
        )
        if not isinstance(data, dict):
            raise TypeError("Conformal checkpoint must contain a dictionary.")
        if data.get("checkpoint_kind") != "transport_based_conformal_predictor":
            raise ValueError("Not a transport-based conformal checkpoint.")
        format_version = data.get("format_version")
        if (
            type(format_version) is not int
            or format_version != self.conformal_checkpoint_version
        ):
            raise ValueError(
                "Unsupported conformal checkpoint format version "
                f"{format_version!r}."
            )

        expected_fingerprint = data.get("predictor_fingerprint")
        actual_fingerprint = self._predictor_fingerprint(predictor)
        if expected_fingerprint != actual_fingerprint:
            raise ValueError(
                "Conformal checkpoint was fitted with a different predictor."
            )

        conformal_predictor = TransportBasedConformalPredictor(
            predictor=predictor,
            config=data["config"],
        )
        state = data.get("calibrator_state")
        if not isinstance(state, dict):
            raise ValueError("Conformal checkpoint has no calibrator state.")

        expected_state_names = {"threshold"}
        if isinstance(conformal_predictor.calibrator, EllipticCalibrator):
            expected_state_names.update(
                {
                    "x_train",
                    "score_train",
                    "global_covariance",
                    "global_inverse_covariance",
                }
            )
        if set(state) != expected_state_names:
            raise ValueError(
                "Conformal checkpoint calibrator state does not match the "
                "configured calibrator."
            )

        self._validate_calibrator_state(
            state=state,
            conformal_predictor=conformal_predictor,
        )
        for name, value in state.items():
            setattr(conformal_predictor.calibrator, name, value)

        calibrator = conformal_predictor.calibrator
        if isinstance(calibrator, EllipticCalibrator):
            if any(getattr(calibrator, name) is None for name in expected_state_names):
                raise ValueError("Elliptic conformal checkpoint has incomplete state.")
            calibrator.knn = NearestNeighbors(
                n_neighbors=calibrator.config.n_neighbors,
                n_jobs=-1,
            )
            calibrator.knn.fit(calibrator.x_train.cpu().numpy())

        if not conformal_predictor.is_calibrated:
            raise ValueError("Conformal checkpoint is not calibrated.")

        return conformal_predictor

    @staticmethod
    def _validate_calibrator_state(
        state: dict[str, Any],
        conformal_predictor: TransportBasedConformalPredictor,
    ) -> None:
        threshold = state["threshold"]
        if (
            not isinstance(threshold, torch.Tensor) or threshold.ndim != 0
            or not threshold.is_floating_point() or torch.isnan(threshold)
        ):
            raise ValueError(
                "Conformal checkpoint threshold must be a non-NaN scalar "
                "floating-point tensor."
            )

        if not isinstance(conformal_predictor.calibrator, EllipticCalibrator):
            return

        x_train = state["x_train"]
        score_train = state["score_train"]
        covariance = state["global_covariance"]
        inverse_covariance = state["global_inverse_covariance"]
        tensors = (x_train, score_train, covariance, inverse_covariance)
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError(
                "Elliptic conformal checkpoint state must contain tensors."
            )
        if (
            x_train.ndim != 2 or x_train.shape[1] != conformal_predictor.x_dim
            or score_train.ndim != 2
            or score_train.shape != (x_train.shape[0], conformal_predictor.y_dim) or
            covariance.shape != (conformal_predictor.y_dim, conformal_predictor.y_dim)
            or inverse_covariance.shape != covariance.shape
        ):
            raise ValueError(
                "Elliptic conformal checkpoint state has incompatible shapes."
            )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError(
                "Elliptic conformal checkpoint state contains non-finite values."
            )

    def _prepare_run_directory(self) -> None:
        if self.run_directory.exists() and any(self.run_directory.iterdir()):
            raise FileExistsError(
                f"Run directory {self.run_directory} is not empty. Use a new "
                "experiment name; resumed stages may still read checkpoints "
                "from an earlier run."
            )

        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.run_directory / "resolved_config.json",
            self.config.model_dump(mode="json"),
        )

    def _preflight_source_checkpoints(self) -> None:
        paths: list[tuple[str, Path | None]] = [
            ("base predictor", self.config.base.predictor_checkpoint),
            ("base trainer", self.config.base.trainer_checkpoint),
            ("conformal predictor", self.config.conformal.checkpoint),
        ]
        if self.config.rearrangement is not None:
            paths.extend(
                [
                    (
                        "rearrangement predictor",
                        self.config.rearrangement.predictor_checkpoint,
                    ),
                    (
                        "rearrangement trainer",
                        self.config.rearrangement.trainer_checkpoint,
                    ),
                ]
            )

        for label, path in paths:
            if path is not None and not path.is_file():
                raise FileNotFoundError(
                    f"{label.capitalize()} checkpoint {path} "
                    "does not exist or is not a file."
                )

    def _start_manifest(self) -> None:
        self._manifest = {
            "experiment_name": self.config.name,
            "status": "running",
            "started_at": self._timestamp(),
            "stages": {},
        }
        self._write_manifest()

    def _record_stage(
        self,
        name: str,
        mode: str,
        duration: float,
        predictor_checkpoint: str | Path | None,
        trainer_checkpoint: str | Path | None = None,
    ) -> None:
        self._manifest["stages"][name] = {
            "mode":
            mode,
            "duration_seconds":
            duration,
            "source_predictor_checkpoint":
            (None if predictor_checkpoint is None else str(predictor_checkpoint)),
            "source_trainer_checkpoint":
            (None if trainer_checkpoint is None else str(trainer_checkpoint)),
        }
        self._write_manifest()

    def _validate_rearrangement_training_input(
        self,
        stage,
        base_predictor,
    ) -> None:
        expected_dtype = getattr(torch, stage.predictor.dtype)
        if base_predictor.dtype != expected_dtype:
            raise ValueError("Base and rearrangement predictor dtypes must match.")

        base_config = getattr(base_predictor, "config", None)
        if (
            getattr(base_config, "type", None) == "neural_optimal_transport"
            and getattr(base_config, "potential_type", None) == "y"
        ):
            raise ValueError(
                "Neural optimal transport with potential_type='y' cannot "
                "train a rearrangement because log_det is unavailable."
            )

    def _validate_base_for_planned_pipeline(self, predictor) -> None:
        base_config = getattr(predictor, "config", None)
        has_y_potential = (
            getattr(base_config, "type", None) == "neural_optimal_transport"
            and getattr(base_config, "potential_type", None) == "y"
        )
        rearrangement = self.config.rearrangement

        if (
            has_y_potential and rearrangement is not None
            and rearrangement.mode in {"train", "resume"}
        ):
            raise ValueError(
                "Neural optimal transport with potential_type='y' cannot "
                "train a rearrangement because log_det is unavailable."
            )

        if has_y_potential and self.config.evaluation.compute_volume:
            raise ValueError(
                "Volume evaluation is unavailable for neural optimal "
                "transport with potential_type='y'."
            )

        if rearrangement is not None and rearrangement.mode == "train":
            expected_dtype = getattr(torch, rearrangement.predictor.dtype)
            if predictor.dtype != expected_dtype:
                raise ValueError("Base and rearrangement predictor dtypes must match.")

    def _set_rearrangement_coverage_mass(self, stage, trainer) -> None:
        if isinstance(stage, AmortizedRearrangedTransportStageConfig):
            self._rearrangement_coverage_mass = None
            return
        self._rearrangement_coverage_mass = float(trainer.config.coverage_mass)

    def _set_loaded_rearrangement_coverage_mass(
        self,
        stage,
        metadata: dict[str, Any] | None,
    ) -> None:
        if (isinstance(stage, AmortizedRearrangedTransportStageConfig)):
            self._rearrangement_coverage_mass = None
            return

        metadata_coverage = None
        if metadata is not None:
            trainer_config = metadata.get("trainer_config")
            if metadata.get("trained_coverage_mass") is not None:
                metadata_coverage = float(metadata["trained_coverage_mass"])
        else:
            trainer_config = None
        if metadata_coverage is None and isinstance(trainer_config, dict):
            coverage_mass = trainer_config.get("coverage_mass")
            if coverage_mass is not None:
                metadata_coverage = float(coverage_mass)

        declared_coverage = stage.trained_coverage_mass
        if (
            metadata_coverage is not None and declared_coverage is not None
            and not math.isclose(
                metadata_coverage,
                declared_coverage,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Configured trained_coverage_mass does not match checkpoint "
                "metadata."
            )

        coverage_mass = (
            metadata_coverage if metadata_coverage is not None else declared_coverage
        )
        if coverage_mass is None:
            raise ValueError(
                "Loading a fixed rearrangement requires runner checkpoint "
                "metadata or an explicit trained_coverage_mass."
            )
        self._rearrangement_coverage_mass = float(coverage_mass)

    def _validate_rearrangement_coverage_mass(
        self,
        conformal_coverage_mass: float,
    ) -> None:
        if self._rearrangement_coverage_mass is None:
            return
        if not math.isclose(
            self._rearrangement_coverage_mass,
            conformal_coverage_mass,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Fixed rearrangement and conformal coverage_mass values "
                "must match."
            )

    def _validate_configured_conformal_coverage(self) -> None:
        conformal = self.config.conformal
        if conformal.mode == "fit":
            self._validate_rearrangement_coverage_mass(conformal.config.coverage_mass)

    @staticmethod
    def _rebind_rearrangement_base(
        predictor,
        base_predictor,
    ) -> None:
        predictor.transport_predictor = base_predictor
        predictor._move_transport_predictor_to_device()

    def _validate_volume_support(
        self,
        conformal_predictor: TransportBasedConformalPredictor,
    ) -> None:
        if not self.config.evaluation.compute_volume:
            return

        calibrator_config = conformal_predictor.config.calibrator
        calibrator_type = calibrator_config.type
        if calibrator_type not in {"no_calibrator", "none", "norm"}:
            raise ValueError(
                "Volume evaluation supports only NoCalibrator or an L2 "
                "NormCalibrator."
            )
        if (
            calibrator_type == "norm" and not math.isclose(
                float(calibrator_config.p),
                2.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Volume evaluation supports only an L2 NormCalibrator.")

        predictor = conformal_predictor.predictor
        transport_predictor = getattr(
            predictor,
            "transport_predictor",
            predictor,
        )
        transport_config = getattr(transport_predictor, "config", None)
        if (
            getattr(transport_config, "type", None) == "neural_optimal_transport"
            and getattr(transport_config, "potential_type", None) == "y"
        ):
            raise ValueError(
                "Volume evaluation is unavailable for neural optimal "
                "transport with potential_type='y'."
            )

    def _write_training_checkpoint_metadata(
        self,
        predictor,
        trainer,
        stage_directory: Path,
    ) -> None:
        if self._training_context_fingerprint is None:
            raise RuntimeError("Training context fingerprint is not initialized.")

        predictor_path = stage_directory / "predictor.pt"
        trainer_path = stage_directory / "trainer.pt"
        history_path = stage_directory / "history.json"
        metadata = {
            "checkpoint_kind": "experiment_training_stage",
            "format_version": self.training_checkpoint_version,
            "predictor_file": predictor_path.name,
            "trainer_file": trainer_path.name,
            "predictor_sha256": self._file_sha256(predictor_path),
            "trainer_sha256": self._file_sha256(trainer_path),
            "history_file": history_path.name,
            "history_sha256": self._file_sha256(history_path),
            "predictor_fingerprint": self._predictor_fingerprint(predictor),
            "trainer_type": self._qualified_type_name(trainer),
            "trainer_config": trainer.config.model_dump(mode="json"),
            "completed_epochs": trainer.completed_epochs,
            "total_epochs": trainer.config.epochs,
            "complete": trainer.completed_epochs >= trainer.config.epochs,
            "training_context_fingerprint": (self._training_context_fingerprint),
        }
        self._write_json(
            stage_directory / self.training_checkpoint_metadata_name,
            metadata,
        )

    def _write_loaded_stage_metadata(
        self,
        stage_directory: Path,
        predictor,
        source_metadata: dict[str, Any] | None,
        trained_coverage_mass: float | None = None,
    ) -> None:
        predictor_path = stage_directory / "predictor.pt"
        history_path = stage_directory / "history.json"
        metadata = {
            "checkpoint_kind":
            "experiment_loaded_stage",
            "format_version":
            self.training_checkpoint_version,
            "predictor_file":
            predictor_path.name,
            "predictor_sha256":
            self._file_sha256(predictor_path),
            "history_file":
            history_path.name,
            "history_sha256":
            self._file_sha256(history_path),
            "predictor_fingerprint":
            self._predictor_fingerprint(predictor),
            "complete":
            True,
            "source_checkpoint_kind": (
                None
                if source_metadata is None else source_metadata.get("checkpoint_kind")
            ),
        }
        if trained_coverage_mass is not None:
            metadata["trained_coverage_mass"] = trained_coverage_mass
        self._write_json(
            stage_directory / self.training_checkpoint_metadata_name,
            metadata,
        )

    def _validate_resume_checkpoint_pair(
        self,
        predictor_path: str | Path,
        trainer_path: str | Path,
    ) -> dict[str, Any]:
        predictor_path = Path(predictor_path)
        trainer_path = Path(trainer_path)
        if predictor_path.parent != trainer_path.parent:
            raise ValueError(
                "Resume predictor and trainer checkpoints must share a "
                "runner stage directory."
            )

        metadata_path = (predictor_path.parent / self.training_checkpoint_metadata_name)
        if not metadata_path.is_file():
            raise ValueError(
                "Resume requires checkpoint.json written by ExperimentRunner."
            )

        metadata = self._read_json(metadata_path)
        self._validate_training_checkpoint_metadata(metadata)
        if (
            metadata.get("predictor_file") != predictor_path.name
            or metadata.get("trainer_file") != trainer_path.name
        ):
            raise ValueError("Resume checkpoint paths do not match checkpoint.json.")
        if metadata.get("predictor_sha256") != self._file_sha256(predictor_path):
            raise ValueError(
                "Predictor checkpoint does not match its checkpoint metadata."
            )
        if metadata.get("trainer_sha256") != self._file_sha256(trainer_path):
            raise ValueError(
                "Trainer checkpoint does not match its checkpoint metadata."
            )
        self._validate_checkpoint_history(
            checkpoint_directory=predictor_path.parent,
            metadata=metadata,
        )
        if (
            metadata.get("training_context_fingerprint")
            != self._training_context_fingerprint
        ):
            raise ValueError(
                "Resume checkpoint was created with different training data "
                "or dataloader settings."
            )
        return metadata

    def _validate_resume_objects(
        self,
        metadata: dict[str, Any],
        predictor,
        trainer,
    ) -> None:
        if (
            metadata.get("predictor_fingerprint")
            != self._predictor_fingerprint(predictor)
        ):
            raise ValueError("Loaded predictor does not match checkpoint metadata.")
        if metadata.get("trainer_type") != self._qualified_type_name(trainer):
            raise ValueError("Loaded trainer does not match checkpoint metadata.")
        if metadata.get("trainer_config") != trainer.config.model_dump(mode="json"):
            raise ValueError(
                "Loaded trainer config does not match checkpoint metadata."
            )
        if metadata.get("completed_epochs") != trainer.completed_epochs:
            raise ValueError(
                "Loaded trainer progress does not match checkpoint metadata."
            )
        if metadata.get("total_epochs") != trainer.config.epochs:
            raise ValueError(
                "Loaded trainer horizon does not match checkpoint metadata."
            )

    def _prevalidate_completed_predictor_checkpoint(
        self,
        path: str | Path,
    ) -> dict[str, Any] | None:
        path = Path(path)
        metadata_path = path.parent / self.training_checkpoint_metadata_name
        if not metadata_path.is_file():
            return None

        metadata = self._read_json(metadata_path)
        self._validate_completed_checkpoint_metadata(metadata)
        if metadata.get("predictor_file") != path.name:
            raise ValueError(
                "Predictor checkpoint path does not match checkpoint.json."
            )
        if metadata.get("predictor_sha256") != self._file_sha256(path):
            raise ValueError("Predictor checkpoint does not match checkpoint.json.")
        self._validate_checkpoint_history(
            checkpoint_directory=path.parent,
            metadata=metadata,
        )
        if metadata.get("complete") is not True:
            raise ValueError(
                "Load mode requires a completed stage checkpoint; use resume "
                "for an intermediate checkpoint."
            )
        return metadata

    def _validate_loaded_predictor_metadata(
        self,
        metadata: dict[str, Any] | None,
        predictor,
    ) -> None:
        if metadata is None:
            return
        if (
            metadata.get("predictor_fingerprint")
            != self._predictor_fingerprint(predictor)
        ):
            raise ValueError("Loaded predictor does not match checkpoint metadata.")

    def _validate_completed_checkpoint_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        if metadata.get("checkpoint_kind") not in {
            "experiment_training_stage",
            "experiment_loaded_stage",
        }:
            raise ValueError("Not an ExperimentRunner stage checkpoint.")
        if (
            type(metadata.get("format_version")) is not int
            or metadata["format_version"] != self.training_checkpoint_version
        ):
            raise ValueError(
                "Unsupported stage checkpoint format version "
                f"{metadata.get('format_version')!r}."
            )
        if type(metadata.get("complete")) is not bool:
            raise ValueError("Stage checkpoint complete flag must be a boolean.")

    def _validate_training_checkpoint_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        if metadata.get("checkpoint_kind") != "experiment_training_stage":
            raise ValueError("Not an ExperimentRunner training checkpoint.")
        if (
            type(metadata.get("format_version")) is not int
            or metadata["format_version"] != self.training_checkpoint_version
        ):
            raise ValueError(
                "Unsupported training checkpoint format version "
                f"{metadata.get('format_version')!r}."
            )
        if type(metadata.get("complete")) is not bool:
            raise ValueError("Training checkpoint complete flag must be a boolean.")
        for field in ("completed_epochs", "total_epochs"):
            if type(metadata.get(field)) is not int:
                raise ValueError(f"Training checkpoint {field} must be an integer.")

    def _validate_checkpoint_history(
        self,
        checkpoint_directory: Path,
        metadata: dict[str, Any],
    ) -> None:
        history_file = metadata.get("history_file")
        if history_file != "history.json":
            raise ValueError("Stage checkpoint history_file must be 'history.json'.")

        history_path = checkpoint_directory / history_file
        if not history_path.is_file():
            raise ValueError("Stage checkpoint history file is missing.")
        if metadata.get("history_sha256") != self._file_sha256(history_path):
            raise ValueError("Training history does not match checkpoint.json.")

    def _make_training_context_fingerprint(self, data) -> str:
        digest = hashlib.sha256()
        dataset_config = self.config.dataset.model_dump(mode="json")
        dataset_config.pop("device", None)
        payload = {
            "dataset": dataset_config,
            "dataloader": {
                "batch_size": self.config.dataloaders.train_batch_size,
                "shuffle": self.config.dataloaders.shuffle_train,
            },
        }
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        self._update_digest_with_tensor(digest, "x", data.x)
        self._update_digest_with_tensor(digest, "y", data.y)
        return digest.hexdigest()

    def _validate_predictor_dimensions(
        self,
        predictor,
        stage_name: str,
    ) -> None:
        if (
            predictor.x_dim != self.dataset.x_dim
            or predictor.y_dim != self.dataset.y_dim
        ):
            raise ValueError(
                f"Loaded {stage_name} predictor dimensions "
                f"({predictor.x_dim}, {predictor.y_dim}) do not match dataset "
                f"dimensions ({self.dataset.x_dim}, {self.dataset.y_dim})."
            )

    def _require_matching_base(self, expected, actual) -> None:
        if (
            self._predictor_fingerprint(expected)
            != self._predictor_fingerprint(actual)
        ):
            raise ValueError(
                "Loaded rearrangement wraps a different base predictor "
                "configuration or state."
            )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    @staticmethod
    def _parameter_count(predictor) -> int:
        parameters = getattr(predictor, "parameters", None)
        if not callable(parameters):
            return 0
        return sum(
            parameter.numel() for parameter in parameters() if parameter.requires_grad
        )

    @classmethod
    def _predictor_fingerprint(cls, predictor) -> str:
        state_dict = getattr(predictor, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(
                "Predictor must expose state_dict() for checkpoint validation."
            )

        digest = hashlib.sha256()
        digest.update(
            f"{type(predictor).__module__}.{type(predictor).__qualname__}".encode()
        )
        digest.update(
            json.dumps(
                cls._predictor_config_payload(predictor),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for name, value in state_dict().items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Predictor state {name!r} is not a torch.Tensor.")
            cls._update_digest_with_tensor(digest, name, value)
        return digest.hexdigest()

    @classmethod
    def _predictor_config_payload(cls, predictor) -> dict[str, Any]:
        config = getattr(predictor, "config", None)
        if config is None or not hasattr(config, "model_dump"):
            raise TypeError(
                "Predictor must expose a Pydantic config for checkpoint "
                "validation."
            )

        config_data = config.model_dump(mode="json")
        config_data.pop("device", None)
        payload: dict[str, Any] = {
            "predictor_type": cls._qualified_type_name(predictor),
            "config": config_data,
        }

        transport_predictor = getattr(
            predictor,
            "transport_predictor",
            None,
        )
        if transport_predictor is not None:
            payload["transport_predictor"] = cls._predictor_config_payload(
                transport_predictor
            )
        return payload

    @staticmethod
    def _update_digest_with_tensor(
        digest,
        name: str,
        value: torch.Tensor,
    ) -> None:
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

    @staticmethod
    def _qualified_type_name(value) -> str:
        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"

    @staticmethod
    def _file_sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _checkpoint_value(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

    def _save_predictor(self, predictor, path: Path) -> None:
        self._atomic_component_save(predictor, path)

    def _save_trainer(self, trainer, path: Path) -> None:
        self._atomic_component_save(trainer, path)

    @staticmethod
    def _atomic_component_save(component, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            component.save(str(temporary_path))
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_torch_save(data: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(data, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _write_manifest(self) -> None:
        self._write_json(
            self.run_directory / "manifest.json",
            self._manifest,
        )

    @staticmethod
    def _load_source_history(
        predictor_path: str | Path,
        metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if metadata is None:
            return []

        history_path = Path(predictor_path).parent / metadata["history_file"]

        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list) or any(
            not isinstance(entry, dict) for entry in history
        ):
            raise ValueError(f"Expected a list of records in {history_path}.")
        return history

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object in {path}.")
        return data

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(
                    data,
                    indent=2,
                    sort_keys=True,
                    default=str,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
