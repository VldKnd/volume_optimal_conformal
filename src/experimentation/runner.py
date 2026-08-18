from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs.conformal import ResidualConformalPredictorConfig
from configs.trainers import RandomForestTrainerConfig
from configs.trainers import rearranged_transport as rearranged_configs
from configs.trainers import transport as trainer_configs
from conformal import (
    ResidualConformalPredictor,
    TransportBasedConformalPredictor,
)
from data.datasets import real as real_datasets
from data.datasets import synthetic as datasets
from data.loaders import make_xy_dataloader
from evaluation import (
    excess_coverage_risk_from_data,
    log_volume,
)
from evaluation.wsc import wsc_unbiased
from experimentation.config import ExperimentConfig
from experimentation.tracking import (
    ExperimentTracker,
    NullTracker,
    create_experiment_tracker,
)
from predictors import RandomForestPredictor
from predictors import rearranged_transport as rearranged_predictors
from predictors import transport as transport_predictors
from trainers import RandomForestTrainer
from trainers import rearranged_transport as rearranged_trainers
from trainers import transport as transport_trainers


def _conformal_calibration_settings(config: Any) -> dict[str, Any]:
    """Return only settings that affect the fitted conformal state."""
    return config.model_dump(
        exclude={
            "volume_mc_samples",
            "volume_batch_size",
            "volume_seed",
        }
    )


class ExperimentRunner:
    """Train, calibrate, evaluate, and save one configured experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        source_config_path: str | Path | None = None,
    ):
        self.config = config
        self.run_directory = config.run_directory
        self.source_config_path = (
            None if source_config_path is None else Path(source_config_path)
        )

        self.dataset = None
        self.predictor = None
        self.rearrangement = None
        self.conformal_predictor = None
        self.histories: dict[str, list[dict]] = {}
        self.metrics: dict[str, object] = {}
        self.tracker: ExperimentTracker = NullTracker()
        self._tracking_log_failed = False

    def run(self) -> ExperimentRunner:
        try:
            result = self._run()
        except BaseException:
            try:
                self.tracker.finish(exit_code=1)
            except Exception as tracking_error:
                print(
                    "[wandb] Failed to finish the unsuccessful run: "
                    f"{tracking_error}",
                    flush=True,
                )
            raise

        try:
            self.tracker.finish(exit_code=0)
        except Exception as tracking_error:
            print(
                f"[wandb] Failed to finish the completed run: {tracking_error}",
                flush=True,
            )
        return result

    def _run(self) -> ExperimentRunner:
        config = self.config
        predictor_config = config.predictor_config

        if predictor_config.type == "random_forest":
            predictor_class = RandomForestPredictor
            trainer_class = RandomForestTrainer
            trainer_config_class = RandomForestTrainerConfig
        elif predictor_config.type == "convex_potential_flow":
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

        trainer_data = config.trainer_config
        if hasattr(trainer_data, "model_dump"):
            trainer_data = trainer_data.model_dump()
        trainer_config = trainer_config_class.model_validate(trainer_data or {})

        rearrangement_class = None
        rearrangement_trainer_class = None
        rearrangement_trainer_config = None
        if config.rearrangement_config is not None:
            trainer_data = config.rearrangement_trainer_config
            if hasattr(trainer_data, "model_dump"):
                trainer_data = trainer_data.model_dump()
            requested_rearrangement_trainer = (
                trainer_data.get("type") if isinstance(trainer_data, dict) else None
            )
            if config.rearrangement_config.type == "amortized_rearranged_transport":
                if requested_rearrangement_trainer is not None:
                    raise ValueError(
                        "Amortized rearrangement does not accept trainer type "
                        f"{requested_rearrangement_trainer!r}."
                    )
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
                if requested_rearrangement_trainer is not None:
                    raise ValueError(
                        "Fixed rearrangement does not accept trainer type "
                        f"{requested_rearrangement_trainer!r}."
                    )
                elif config.supervised_rearrangement:
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

            rearrangement_trainer_config = (
                rearrangement_trainer_config_class.model_validate(trainer_data or {})
            )

        dataset_config = config.dataset_config
        if (
            predictor_config.x_dim != dataset_config.x_dim
            or predictor_config.y_dim != dataset_config.y_dim
        ):
            raise ValueError("Predictor and dataset dimensions must match.")

        if config.rearrangement_config is not None:
            if predictor_config.type == "random_forest":
                raise ValueError(
                    "A rearrangement layer requires a transport predictor."
                )

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
                config.supervised_rearrangement
                and rearrangement_config.type != "amortized_rearranged_transport"
                and not math.isclose(
                    rearrangement_trainer_config.coverage_mass,
                    config.conformal_config.coverage_mass,
                )
            ):
                raise ValueError(
                    "Supervised rearrangement and conformal coverage masses must "
                    "match."
                )

        residual_conformal = isinstance(
            config.conformal_config,
            ResidualConformalPredictorConfig,
        )
        if residual_conformal != (predictor_config.type == "random_forest"):
            raise ValueError(
                "Random-forest regression requires residual conformal prediction; "
                "transport predictors require transport-based conformal prediction."
            )

        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.run_directory / "config.json",
            config.model_dump(mode="json"),
        )

        resolved_run_config = config.model_dump(mode="json")
        resolved_run_config["trainer_config"] = trainer_config.model_dump(mode="json")
        resolved_run_config["rearrangement_trainer_config"] = (
            None if rearrangement_trainer_config is None else
            rearrangement_trainer_config.model_dump(mode="json")
        )
        self.tracker = create_experiment_tracker(
            config.wandb,
            run_config=resolved_run_config,
            run_directory=self.run_directory,
            source_config_path=self.source_config_path,
        )
        self._seed(config.seed)

        if dataset_config.type == "atp1d":
            self.dataset = real_datasets.ATP1dDataset(dataset_config)
        elif dataset_config.type == "atp7d":
            self.dataset = real_datasets.ATP7dDataset(dataset_config)
        elif dataset_config.type == "bio":
            self.dataset = real_datasets.BioDataset(dataset_config)
        elif dataset_config.type == "blog":
            self.dataset = real_datasets.BlogDataset(dataset_config)
        elif dataset_config.type == "qm9":
            self.dataset = real_datasets.QM9Dataset(dataset_config)
        elif dataset_config.type == "scm1d":
            self.dataset = real_datasets.SCM1dDataset(dataset_config)
        elif dataset_config.type == "scm20d":
            self.dataset = real_datasets.SCM20dDataset(dataset_config)
        elif dataset_config.type == "rf1":
            self.dataset = real_datasets.RF1Dataset(dataset_config)
        elif dataset_config.type == "rf2":
            self.dataset = real_datasets.RF2Dataset(dataset_config)
        elif dataset_config.type == "sgemm":
            self.dataset = real_datasets.SGEMMDataset(dataset_config)
        elif dataset_config.type == "banana":
            self.dataset = datasets.BananaDataset(dataset_config)
        elif dataset_config.type == "bimodal_gaussian":
            self.dataset = datasets.BimodalGaussianDataset(dataset_config)
        elif dataset_config.type == "gaussian_dataset":
            self.dataset = datasets.GaussianDatasetTarget(dataset_config)
        elif dataset_config.type == "student_t_dataset":
            self.dataset = datasets.StudentTDataset(dataset_config)
        elif dataset_config.type in {"star_shaped_gaussian", "star_shaped"}:
            self.dataset = datasets.StarShapedGaussianDataset(dataset_config)
        else:
            self.dataset = datasets.SinusoidalTransportDataset(dataset_config)

        splits = self.dataset.get_splits()
        train_loader = make_xy_dataloader(
            splits.train,
            batch_size=config.train_batch_size,
            shuffle=True,
            drop_last=True,
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
        if config.predictor_checkpoint is not None:
            self.predictor = predictor_class.load(
                str(config.predictor_checkpoint),
                map_location=predictor_config.device,
            )
        else:
            self.predictor = predictor_class(predictor_config)
            base_trainer = trainer_class(trainer_config)
            self._attach_live_tracking(base_trainer, stage="base")
            self._seed(config.seed)
            base_trainer.fit(self.predictor, train_loader)
        eval_method = getattr(self.predictor, "eval", None)
        if callable(eval_method):
            eval_method()
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
                collect_solver_diagnostics = (
                    self.tracker.enabled and not self._tracking_log_failed
                    and config.wandb.log_solver_diagnostics
                )
                rearrangement_flow = self.rearrangement.rearrangement_flow
                if collect_solver_diagnostics:
                    rearrangement_flow.enable_solver_diagnostics()
                try:
                    self._attach_live_tracking(
                        rearrangement_trainer,
                        stage="rearrangement",
                    )
                    rearrangement_train_loader = make_xy_dataloader(
                        splits.train,
                        batch_size=(
                            config.rearrangement_train_batch_size
                            or config.train_batch_size
                        ),
                        shuffle=True,
                    )
                    self._seed(config.seed)
                    rearrangement_trainer.fit(
                        self.rearrangement,
                        rearrangement_train_loader,
                    )
                finally:
                    if collect_solver_diagnostics:
                        rearrangement_flow.disable_solver_diagnostics()
            self.rearrangement.eval()
            self._save_stage(
                "rearrangement",
                self.rearrangement,
                rearrangement_trainer,
            )
            final_predictor = self.rearrangement

        conformal_class = (
            ResidualConformalPredictor
            if residual_conformal else TransportBasedConformalPredictor
        )
        conformal_config_class = type(config.conformal_config)

        if config.conformal_checkpoint is None:
            self.conformal_predictor = conformal_class(
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
            saved_config = conformal_config_class.model_validate(checkpoint["config"])
            if _conformal_calibration_settings(saved_config
                                               ) != _conformal_calibration_settings(
                                                   config.conformal_config
                                               ):
                raise ValueError(
                    "Loaded conformal checkpoint has different calibration "
                    "settings."
                )
            self.conformal_predictor = conformal_class(
                predictor=final_predictor,
                config=config.conformal_config,
            )
            self.conformal_predictor.calibrator = checkpoint["calibrator"]
            if residual_conformal:
                self.conformal_predictor.calibration_x = checkpoint["calibration_x"]
                self.conformal_predictor.calibration_residuals = checkpoint[
                    "calibration_residuals"]
                self.conformal_predictor.volume_neighbors = checkpoint[
                    "volume_neighbors"]
            else:
                self.conformal_predictor.calibration_x = checkpoint.get("calibration_x")
                self.conformal_predictor.calibration_y = checkpoint.get("calibration_y")
                self.conformal_predictor.volume_neighbors = checkpoint.get(
                    "volume_neighbors"
                )

        conformal_directory = self.run_directory / "conformal"
        conformal_directory.mkdir(exist_ok=True)
        conformal_checkpoint = {
            "config": self.conformal_predictor.config.model_dump(),
            "calibrator": self.conformal_predictor.calibrator,
        }
        if residual_conformal:
            conformal_checkpoint.update(
                calibration_x=self.conformal_predictor.calibration_x,
                calibration_residuals=(self.conformal_predictor.calibration_residuals),
                volume_neighbors=self.conformal_predictor.volume_neighbors,
            )
        elif self.conformal_predictor.calibration_y is not None:
            conformal_checkpoint.update(
                calibration_x=self.conformal_predictor.calibration_x,
                calibration_y=self.conformal_predictor.calibration_y,
                volume_neighbors=self.conformal_predictor.volume_neighbors,
            )
        torch.save(
            conformal_checkpoint,
            conformal_directory / "predictor.pt",
        )

        metrics_start = time.perf_counter()
        if config.metrics_verbose:
            print("[metrics] Computing coverage indicators...", flush=True)

        coverage_start = time.perf_counter()
        representations = []
        coverage_indicators = []
        for x_batch, y_batch in test_loader:
            with torch.no_grad():
                inside = self.conformal_predictor.contains(
                    x_batch,
                    y_batch,
                ).detach().reshape(-1)
            if inside.numel() != x_batch.shape[0]:
                raise ValueError("contains must return one value per observation.")
            representations.append(x_batch.detach().cpu())
            coverage_indicators.append(inside.cpu())

        if not coverage_indicators:
            raise ValueError("The test split is empty.")

        representations = torch.cat(representations)
        coverage_indicators = torch.cat(coverage_indicators)
        if config.metrics_verbose:
            elapsed = time.perf_counter() - coverage_start
            print(
                f"[metrics] Coverage indicators completed in {elapsed:.1f}s.",
                flush=True,
            )

        marginal_mean = float(coverage_indicators.float().mean())
        marginal_std = None
        if config.metrics_verbose:
            print(
                f"[metrics] Marginal coverage: {marginal_mean:.4f}.",
                flush=True,
            )

        if config.metrics_verbose:
            print("[metrics] Computing worst-slab coverage...", flush=True)
        metric_start = time.perf_counter()
        slab_mean, slab_std = wsc_unbiased(
            representations=representations.numpy(),
            coverages=coverage_indicators.numpy(),
            delta=0.1,
            M=1_000,
            test_size=0.75,
            random_state=config.seed,
        )
        if config.metrics_verbose:
            elapsed = time.perf_counter() - metric_start
            print(
                "[metrics] Worst-slab coverage completed in "
                f"{elapsed:.1f}s: {slab_mean:.4f} ± {slab_std:.4f}.",
                flush=True,
            )

        if config.metrics_verbose:
            print("[metrics] Computing excess coverage risk...", flush=True)
        metric_start = time.perf_counter()
        risk_mean, risk_std = excess_coverage_risk_from_data(
            x=representations.numpy(),
            coverage_indicators=coverage_indicators.numpy(),
            target_coverage=self.conformal_predictor.coverage_mass,
        )
        if config.metrics_verbose:
            elapsed = time.perf_counter() - metric_start
            print(
                "[metrics] Excess coverage risk completed in "
                f"{elapsed:.1f}s: {risk_mean:.4f} ± {risk_std:.4f}.",
                flush=True,
            )

        self.metrics = {
            "n_test": splits.test.x.shape[0],
            "target_coverage": self.conformal_predictor.coverage_mass,
            "marginal_coverage": {
                "mean": marginal_mean,
                "std": marginal_std,
            },
            "worst_slab_coverage": {
                "mean": slab_mean,
                "std": slab_std,
            },
            "excess_coverage_risk": {
                "mean": risk_mean,
                "std": risk_std,
            },
        }
        if config.compute_volume:
            if config.metrics_verbose:
                print(
                    "[metrics] Computing log-volume per dimension...",
                    flush=True,
                )
            metric_start = time.perf_counter()
            volume_mean, volume_std = log_volume(
                test_loader,
                self.conformal_predictor,
            )
            self.metrics["log_volume_per_dimension"] = {
                "mean": volume_mean,
                "std": volume_std,
            }
            if config.metrics_verbose:
                elapsed = time.perf_counter() - metric_start
                print(
                    "[metrics] Log-volume per dimension completed in "
                    f"{elapsed:.1f}s: {volume_mean:.4f} ± {volume_std:.4f}.",
                    flush=True,
                )

        self._write_json(self.run_directory / "metrics.json", self.metrics)
        self._log_tracking_metrics("evaluation", self.metrics)
        if config.metrics_verbose:
            elapsed = time.perf_counter() - metrics_start
            print(
                f"[metrics] All metrics completed in {elapsed:.1f}s.",
                flush=True,
            )
        return self

    def _attach_live_tracking(self, trainer, stage: str) -> None:
        if not self.tracker.enabled or self._tracking_log_failed:
            return

        log_every_n_steps = self.config.wandb.log_every_n_steps

        def log_training_metrics(
            event: str,
            metrics: dict[str, Any],
            step: int,
        ) -> None:
            if (event == "batch" and step != 1 and step % log_every_n_steps != 0):
                return

            prefix = f"{event}_"

            def metric_name(key: str) -> str:
                if event == "epoch" and key == "epoch":
                    return "epoch"
                if key.startswith(prefix):
                    return key
                return f"{prefix}{key}"

            payload = {metric_name(key): value for key, value in metrics.items()}
            self._log_tracking_metrics(stage, payload, step=step)
            if self._tracking_log_failed:
                trainer.set_metric_callback(None)

        trainer.set_metric_callback(log_training_metrics)

    def _log_tracking_metrics(
        self,
        stage: str,
        metrics: dict[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        if self._tracking_log_failed or not self.tracker.enabled:
            return

        try:
            self.tracker.log(stage, metrics, step=step)
        except Exception as tracking_error:
            self._tracking_log_failed = True
            if stage == "rearrangement" and self.rearrangement is not None:
                self.rearrangement.rearrangement_flow.disable_solver_diagnostics()
            print(
                "[wandb] Live logging failed; training will continue without "
                f"further W&B metrics: {tracking_error}",
                flush=True,
            )

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
