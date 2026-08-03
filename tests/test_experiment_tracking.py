from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from experimentation.config import WandbConfig
from experimentation.tracking import (
    NullTracker,
    WandbTracker,
    create_experiment_tracker,
    flatten_metrics,
    normalize_scalar,
)


class _ScalarLike:

    def __init__(self, value: float):
        self.value = value

    def item(self) -> float:
        return self.value


class ExperimentTrackingTest(unittest.TestCase):

    def test_wandb_config_is_disabled_by_default_and_validates_fields(self) -> None:
        config = WandbConfig()

        self.assertEqual(config.mode, "disabled")
        self.assertEqual(config.log_every_n_steps, 20)
        self.assertTrue(config.log_solver_diagnostics)
        self.assertEqual(config.tags, [])

        with self.assertRaises(ValidationError):
            WandbConfig(mode="sometimes")
        with self.assertRaises(ValidationError):
            WandbConfig(log_every_n_steps=0)
        with self.assertRaises(ValidationError):
            WandbConfig(unknown_option=True)

    def test_disabled_factory_does_not_import_wandb(self) -> None:
        with mock.patch(
            "experimentation.tracking.importlib.import_module",
        ) as import_module:
            tracker = create_experiment_tracker(
                WandbConfig(),
                run_config={"name": "smoke"},
                run_directory=Path("unused"),
            )

        self.assertIsInstance(tracker, NullTracker)
        self.assertFalse(tracker.enabled)
        import_module.assert_not_called()

    def test_null_tracker_accepts_live_logging_calls(self) -> None:
        tracker = NullTracker()

        tracker.log("train", {"loss": 1.0}, step=3, commit=False)
        tracker.finish(exit_code=1)

    def test_scalar_normalization_and_nested_metric_flattening(self) -> None:
        self.assertEqual(normalize_scalar(_ScalarLike(2.5)), 2.5)
        self.assertEqual(normalize_scalar(Path("checkpoint.pt")), "checkpoint.pt")
        self.assertEqual(
            flatten_metrics(
                {
                    "loss": _ScalarLike(1.25),
                    "coverage": {
                        "mean": 0.9,
                        "std": None
                    },
                },
                prefix="rearrangement",
            ),
            {
                "rearrangement/loss": 1.25,
                "rearrangement/coverage/mean": 0.9,
            },
        )

        with self.assertRaises(TypeError):
            normalize_scalar([1.0, 2.0])

    def test_wandb_tracker_uses_independent_stage_axes(self) -> None:
        run = mock.Mock()
        tracker = WandbTracker(run)

        tracker.log("base", {"loss": 2.0}, step=10)
        tracker.log("base", {"loss": 1.0}, step=11, commit=False)
        tracker.log("rearrangement", {"loss": 4.0}, step=0)

        self.assertEqual(
            run.define_metric.call_args_list,
            [
                mock.call("base/global_step"),
                mock.call("base/*", step_metric="base/global_step"),
                mock.call("rearrangement/global_step"),
                mock.call(
                    "rearrangement/*",
                    step_metric="rearrangement/global_step",
                ),
            ],
        )
        self.assertEqual(
            run.log.call_args_list,
            [
                mock.call(
                    {
                        "base/loss": 2.0,
                        "base/global_step": 10
                    },
                    commit=True,
                ),
                mock.call(
                    {
                        "base/loss": 1.0,
                        "base/global_step": 11
                    },
                    commit=False,
                ),
                mock.call(
                    {
                        "rearrangement/loss": 4.0,
                        "rearrangement/global_step": 0,
                    },
                    commit=True,
                ),
            ],
        )

    def test_online_factory_passes_identity_and_serialized_config(self) -> None:
        run = mock.Mock()
        wandb = mock.Mock()
        wandb.init.return_value = run
        source_path = Path("benchmark/configurations/scm20d/suite/method/seed_03.yaml")

        with mock.patch(
            "experimentation.tracking.importlib.import_module",
            return_value=wandb,
        ):
            tracker = create_experiment_tracker(
                WandbConfig(mode="online", tags=["scm20d"]),
                run_config={
                    "name": "seed_03",
                    "epochs": 5
                },
                run_directory=Path("results/seed_03"),
                source_config_path=source_path,
            )

        self.assertIsInstance(tracker, WandbTracker)
        wandb.init.assert_called_once_with(
            project="minimal-volume-conformal-prediction",
            dir="results/seed_03",
            config={
                "name": "seed_03",
                "epochs": 5,
                "source_config_path": str(source_path),
            },
            mode="online",
            name="seed_03",
            group="scm20d/suite/method",
            reinit="create_new",
            tags=["scm20d"],
        )

    def test_enabled_factory_has_clear_missing_dependency_error(self) -> None:
        missing_wandb = ModuleNotFoundError(
            "No module named 'wandb'",
            name="wandb",
        )
        with mock.patch(
            "experimentation.tracking.importlib.import_module",
            side_effect=missing_wandb,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "uv sync --extra tracking",
            ):
                create_experiment_tracker(
                    WandbConfig(mode="online"),
                    run_config={"name": "smoke"},
                    run_directory=Path("results/smoke"),
                )

    def test_finish_is_idempotent(self) -> None:
        run = mock.Mock()
        tracker = WandbTracker(run)

        tracker.finish(exit_code=1)
        tracker.finish(exit_code=0)

        run.finish.assert_called_once_with(exit_code=1)


if __name__ == "__main__":
    unittest.main()
