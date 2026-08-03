from __future__ import annotations

import importlib
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from experimentation.config import WandbConfig

Scalar = bool | int | float | str


def normalize_scalar(value: Any) -> Scalar:
    """Convert a scalar-like value into a value safe for metric logging."""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            item_value = item()
        except (RuntimeError, TypeError, ValueError) as error:
            raise TypeError(
                "Only scalar tensors and arrays can be logged as metrics."
            ) from error
        if item_value is value:
            raise TypeError(f"Unsupported metric value {value!r}.")
        return normalize_scalar(item_value)

    raise TypeError(
        f"Unsupported metric value of type {type(value).__name__}; "
        "expected a scalar."
    )


def flatten_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str | None = None,
    separator: str = "/",
) -> dict[str, Scalar]:
    """Flatten nested metric mappings and omit unavailable (``None``) values."""
    flattened: dict[str, Scalar] = {}

    def visit(values: Mapping[str, Any], parent: str | None) -> None:
        for raw_key, value in values.items():
            key = str(raw_key).strip(separator)
            if not key:
                raise ValueError("Metric names must not be empty.")
            full_key = separator.join(part for part in (parent, key) if part)
            if isinstance(value, Mapping):
                visit(value, full_key)
            elif value is not None:
                if full_key in flattened:
                    raise ValueError(f"Duplicate flattened metric name: {full_key!r}.")
                flattened[full_key] = normalize_scalar(value)

    normalized_prefix = prefix.strip(separator) if prefix else None
    visit(metrics, normalized_prefix)
    return flattened


@runtime_checkable
class ExperimentTracker(Protocol):
    """Backend-neutral interface used by trainers and experiment runners."""

    @property
    def enabled(self) -> bool:
        ...

    def log(
        self,
        stage: str,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        ...

    def finish(self, exit_code: int = 0) -> None:
        ...


class NullTracker:
    """No-op tracker used when experiment tracking is disabled."""

    @property
    def enabled(self) -> bool:
        return False

    def log(
        self,
        stage: str,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        del stage, metrics, step, commit

    def finish(self, exit_code: int = 0) -> None:
        del exit_code


class WandbTracker:
    """Thin adapter around an initialized ``wandb.Run``."""

    def __init__(self, run: Any):
        self.run = run
        self._defined_step_axes: set[str] = set()
        self._finished = False

    @property
    def enabled(self) -> bool:
        return True

    @staticmethod
    def _normalize_stage(stage: str) -> str:
        normalized = stage.strip("/")
        if not normalized:
            raise ValueError("Tracking stage must not be empty.")
        if "/" in normalized:
            raise ValueError("Tracking stage must be a single path component.")
        return normalized

    def _define_step_axis(self, stage: str) -> str:
        step_metric = f"{stage}/global_step"
        if stage not in self._defined_step_axes:
            self.run.define_metric(step_metric)
            self.run.define_metric(
                f"{stage}/*",
                step_metric=step_metric,
            )
            self._defined_step_axes.add(stage)
        return step_metric

    def log(
        self,
        stage: str,
        metrics: Mapping[str, Any],
        *,
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        if self._finished:
            raise RuntimeError("Cannot log to a finished W&B run.")

        normalized_stage = self._normalize_stage(stage)
        payload = flatten_metrics(metrics, prefix=normalized_stage)
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, Integral):
                raise TypeError("Tracking step must be an integer.")
            if step < 0:
                raise ValueError("Tracking step must be non-negative.")
            step_metric = f"{normalized_stage}/global_step"
            if step_metric in payload:
                raise ValueError(
                    "Do not include 'global_step' in metrics when passing step=."
                )
            self._define_step_axis(normalized_stage)
            payload[step_metric] = int(step)

        if payload:
            self.run.log(payload, commit=commit)

    def finish(self, exit_code: int = 0) -> None:
        if not self._finished:
            self.run.finish(exit_code=exit_code)
            self._finished = True


def _default_group(source_config_path: str | Path | None) -> str | None:
    if source_config_path is None:
        return None

    parent = Path(source_config_path).parent
    parts = parent.parts
    try:
        configurations_index = len(parts) - 1 - parts[::-1].index("configurations")
    except ValueError:
        return None

    relative_parts = parts[configurations_index + 1:]
    return "/".join(relative_parts) or None


def _load_wandb() -> Any:
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as error:
        if error.name != "wandb":
            raise
        raise RuntimeError(
            "W&B logging is enabled, but the optional 'wandb' dependency is "
            "not installed. Install it with `uv sync --extra tracking`, or set "
            "wandb.mode to 'disabled'."
        ) from error


def create_experiment_tracker(
    wandb_config: WandbConfig | None,
    *,
    run_config: Mapping[str, Any],
    run_directory: str | Path,
    source_config_path: str | Path | None = None,
) -> ExperimentTracker:
    """Create a tracker without importing W&B on the disabled path."""
    if wandb_config is None or wandb_config.mode == "disabled":
        return NullTracker()

    wandb = _load_wandb()
    serialized_config = dict(run_config)
    if source_config_path is not None:
        serialized_config["source_config_path"] = str(source_config_path)

    init_arguments: dict[str, Any] = {
        "project": wandb_config.project,
        "dir": str(run_directory),
        "config": serialized_config,
        "mode": wandb_config.mode,
        "name": wandb_config.name or serialized_config.get("name"),
        "group": wandb_config.group or _default_group(source_config_path),
        "reinit": "create_new",
    }
    optional_arguments = {
        "entity": wandb_config.entity,
        "tags": list(wandb_config.tags) if wandb_config.tags else None,
        "notes": wandb_config.notes,
        "job_type": wandb_config.job_type,
    }
    init_arguments.update(
        {
            key: value
            for key, value in optional_arguments.items() if value is not None
        }
    )

    run = wandb.init(**init_arguments)
    if run is None:
        raise RuntimeError("wandb.init() did not return a run.")
    return WandbTracker(run)


__all__ = [
    "ExperimentTracker",
    "NullTracker",
    "WandbTracker",
    "create_experiment_tracker",
    "flatten_metrics",
    "normalize_scalar",
]
