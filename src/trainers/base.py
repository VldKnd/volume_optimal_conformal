from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar, Self

import numpy as np
import torch
from pydantic import BaseModel
from predictors.base import BasePredictor


class BaseTrainer(ABC):
    """Base trainer with epoch-boundary checkpoint and resume support.

    Trainer checkpoints contain the trainer configuration, history, optimizer,
    scheduler, progress, and random-number-generator states. Predictor weights
    remain in the predictor's own checkpoint; load the matching predictor before
    continuing training with a loaded trainer. Concrete trainers accept
    ``max_epochs`` to run only that many additional epochs while keeping
    ``config.epochs`` as the total training and scheduler horizon. Explicit
    DataLoader/sampler generators and persistent worker states remain the
    caller's responsibility.
    """

    checkpoint_version: ClassVar[int] = 1
    config_class: ClassVar[type[BaseModel] | None] = None
    trainer_type: ClassVar[str | None] = None

    def __init__(self, config: BaseModel):
        self.config = config
        self.training_history: list[dict] = []

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self.completed_epochs = 0
        self.global_step = 0
        self.steps_per_epoch: int | None = None
        self.initialization_complete = False

        self._optimized_parameter_signature: list[dict[str, Any]] | None = None
        self._predictor_type: str | None = None
        self._pending_optimizer_state_dict: dict[str, Any] | None = None
        self._pending_scheduler_state_dict: dict[str, Any] | None = None
        self._pending_rng_state: dict[str, Any] | None = None

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def state_dict(self) -> dict[str, Any]:
        optimizer_state = (
            self.optimizer.state_dict()
            if self.optimizer is not None else self._pending_optimizer_state_dict
        )
        scheduler_state = (
            self.scheduler.state_dict()
            if self.scheduler is not None else self._pending_scheduler_state_dict
        )
        rng_state = (
            self._pending_rng_state
            if self._pending_rng_state is not None else self._capture_rng_state()
        )

        return {
            "checkpoint_kind":
            "trainer",
            "format_version":
            self.checkpoint_version,
            "trainer_type":
            self._resolved_trainer_type(),
            "config":
            self.config.model_dump(),
            "training_history":
            copy.deepcopy(self.training_history),
            "progress": {
                "completed_epochs": self.completed_epochs,
                "global_step": self.global_step,
                "steps_per_epoch": self.steps_per_epoch,
                "initialization_complete": self.initialization_complete,
            },
            "predictor_type":
            self._predictor_type,
            "optimized_parameter_signature":
            copy.deepcopy(self._optimized_parameter_signature),
            "optimizer_state_dict":
            optimizer_state,
            "scheduler_state_dict":
            scheduler_state,
            "rng_state":
            rng_state,
        }

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> Self:
        data = torch.load(path, map_location=map_location, weights_only=False)

        if not isinstance(data, dict):
            raise TypeError("Trainer checkpoint must contain a dictionary.")

        checkpoint_kind = data.get("checkpoint_kind")
        if checkpoint_kind != "trainer":
            raise ValueError(f"Expected a trainer checkpoint, got {checkpoint_kind!r}.")

        format_version = data.get("format_version")
        if (
            isinstance(format_version, bool) or format_version != cls.checkpoint_version
        ):
            raise ValueError(
                f"Unsupported trainer checkpoint version {format_version}."
            )

        expected_trainer_type = cls._resolved_trainer_type()
        saved_trainer_type = data.get("trainer_type")
        if saved_trainer_type != expected_trainer_type:
            raise ValueError(
                f"Checkpoint contains trainer type {saved_trainer_type!r}, "
                f"not {expected_trainer_type!r}."
            )

        if cls.config_class is None:
            raise TypeError(f"{cls.__name__} does not define config_class.")

        if "config" not in data:
            raise ValueError("Trainer checkpoint does not contain a config.")

        config = cls.config_class.model_validate(data["config"])
        trainer = cls(config)

        training_history = data.get("training_history")
        if not isinstance(training_history, list):
            raise ValueError("Trainer checkpoint training_history must be a list.")
        trainer.training_history = copy.deepcopy(training_history)

        progress = data.get("progress", {})
        if not isinstance(progress, dict):
            raise ValueError("Trainer checkpoint progress must be a dictionary.")

        trainer.completed_epochs = cls._validate_non_negative_int(
            progress.get("completed_epochs"),
            name="completed_epochs",
        )
        trainer.global_step = cls._validate_non_negative_int(
            progress.get("global_step"),
            name="global_step",
        )

        steps_per_epoch = progress.get("steps_per_epoch")
        if steps_per_epoch is not None:
            steps_per_epoch = cls._validate_non_negative_int(
                steps_per_epoch,
                name="steps_per_epoch",
            )
            if steps_per_epoch == 0:
                raise ValueError("steps_per_epoch must be positive when saved.")
        trainer.steps_per_epoch = steps_per_epoch

        if trainer.completed_epochs > trainer.config.epochs:
            raise ValueError(
                "Trainer checkpoint has more completed epochs than configured "
                "epochs."
            )

        if len(trainer.training_history) != trainer.completed_epochs:
            raise ValueError(
                "Trainer checkpoint history length does not match "
                "completed_epochs."
            )

        initialization_complete = progress.get("initialization_complete")
        if not isinstance(initialization_complete, bool):
            raise ValueError(
                "Trainer checkpoint initialization_complete must be a boolean."
            )
        trainer.initialization_complete = initialization_complete

        trainer._predictor_type = data.get("predictor_type")
        trainer._optimized_parameter_signature = copy.deepcopy(
            data.get("optimized_parameter_signature")
        )
        trainer._pending_optimizer_state_dict = data.get("optimizer_state_dict")
        trainer._pending_scheduler_state_dict = data.get("scheduler_state_dict")
        if (
            trainer._pending_scheduler_state_dict is not None
            and trainer._pending_optimizer_state_dict is None
        ):
            raise ValueError(
                "Trainer checkpoint contains scheduler state without optimizer "
                "state."
            )

        rng_state = data.get("rng_state")
        if rng_state is not None and not isinstance(rng_state, dict):
            raise ValueError("Trainer checkpoint rng_state must be a dictionary.")
        trainer._pending_rng_state = rng_state

        return trainer

    def _setup_optimization(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        steps_per_epoch: int,
        predictor: BasePredictor,
    ) -> tuple[
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler | None,
    ]:
        self._validate_steps_per_epoch(steps_per_epoch)

        named_parameters = list(named_parameters)
        parameters = [parameter for _, parameter in named_parameters]
        parameter_signature = [
            {
                "name": name,
                "shape": list(parameter.shape),
            } for name, parameter in named_parameters
        ]
        predictor_type = self._qualified_type_name(predictor)

        optimizer_state = (
            self._pending_optimizer_state_dict
            if self._pending_optimizer_state_dict is not None else
            self.optimizer.state_dict() if self.optimizer is not None else None
        )
        scheduler_state = (
            self._pending_scheduler_state_dict
            if self._pending_scheduler_state_dict is not None else
            self.scheduler.state_dict() if self.scheduler is not None else None
        )

        if optimizer_state is not None or scheduler_state is not None:
            if (
                self._predictor_type is not None
                and self._predictor_type != predictor_type
            ):
                raise ValueError(
                    f"Trainer checkpoint expects predictor type "
                    f"{self._predictor_type!r}, got {predictor_type!r}."
                )

            if (
                self._optimized_parameter_signature is not None
                and self._optimized_parameter_signature != parameter_signature
            ):
                raise ValueError(
                    "Trainer checkpoint parameters do not match the supplied "
                    "predictor."
                )

            if (
                self.steps_per_epoch is not None
                and self.steps_per_epoch != steps_per_epoch
            ):
                raise ValueError(
                    "Cannot resume with a different number of batches per epoch: "
                    f"checkpoint has {self.steps_per_epoch}, dataloader has "
                    f"{steps_per_epoch}."
                )

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = None
        if self.config.use_cosine_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs * steps_per_epoch,
            )

        if scheduler_state is not None:
            if scheduler is None:
                raise ValueError(
                    "Checkpoint contains scheduler state, but the trainer "
                    "configuration disables the scheduler."
                )
            scheduler.load_state_dict(scheduler_state)
        elif optimizer_state is not None and scheduler is not None:
            raise ValueError(
                "Checkpoint contains optimizer state but no scheduler state."
            )

        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
            except (ValueError, RuntimeError) as error:
                raise ValueError(
                    "Could not bind the checkpoint optimizer state to the "
                    "supplied predictor."
                ) from error

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.steps_per_epoch = steps_per_epoch
        self._predictor_type = predictor_type
        self._optimized_parameter_signature = parameter_signature
        self._pending_optimizer_state_dict = None
        self._pending_scheduler_state_dict = None

        return optimizer, scheduler

    def _fit_end_epoch(self, max_epochs: int | None) -> int:
        if max_epochs is None:
            return self.config.epochs

        if (
            isinstance(max_epochs, bool) or not isinstance(max_epochs, int)
            or max_epochs < 0
        ):
            raise ValueError("max_epochs must be a non-negative integer when provided.")

        return min(
            self.config.epochs,
            self.completed_epochs + max_epochs,
        )

    def _restore_rng_state(self) -> None:
        if self._pending_rng_state is None:
            return

        cpu_state = self._pending_rng_state.get("cpu")
        if cpu_state is not None:
            torch.set_rng_state(cpu_state.cpu())

        python_state = self._pending_rng_state.get("python")
        if python_state is not None:
            random.setstate(python_state)

        numpy_state = self._pending_rng_state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        cuda_states = self._pending_rng_state.get("cuda")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

        mps_state = self._pending_rng_state.get("mps")
        if mps_state is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(mps_state.cpu())

        self._pending_rng_state = None

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        return {
            "python":
            random.getstate(),
            "numpy":
            np.random.get_state(),
            "cpu":
            torch.get_rng_state(),
            "cuda":
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps":
            (torch.mps.get_rng_state() if torch.backends.mps.is_available() else None),
        }

    @staticmethod
    def _qualified_type_name(value: object) -> str:
        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"

    @staticmethod
    def _validate_non_negative_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Trainer checkpoint {name} must be a non-negative integer."
            )
        return value

    @staticmethod
    def _validate_steps_per_epoch(steps_per_epoch: int) -> None:
        if steps_per_epoch < 1:
            raise ValueError("dataloader must contain at least one batch.")

    @classmethod
    def _resolved_trainer_type(cls) -> str:
        return cls.trainer_type or cls.__name__

    @abstractmethod
    def fit(
        self,
        predictor: BasePredictor,
        *args,
        **kwargs,
    ) -> BasePredictor:
        """
        Fit the predictor in-place.

        Returns:
            The trained predictor.
        """
