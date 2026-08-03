from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimentation.runner import ExperimentRunner


class _RecordingTracker:
    enabled = True

    def __init__(self):
        self.logs = []
        self.finish_calls = []

    def log(self, stage, metrics, *, step=None, commit=True):
        self.logs.append((stage, metrics, step, commit))

    def finish(self, exit_code=0):
        self.finish_calls.append(exit_code)


def _runner(log_every_n_steps=3):
    config = SimpleNamespace(
        run_directory=Path("unused"),
        wandb=SimpleNamespace(log_every_n_steps=log_every_n_steps),
    )
    return ExperimentRunner(config)


class RunnerLiveTrackingTest(unittest.TestCase):

    def test_batch_metrics_are_throttled_but_first_batch_and_epochs_log(self):
        runner = _runner(log_every_n_steps=3)
        tracker = _RecordingTracker()
        trainer = mock.Mock()
        runner.tracker = tracker

        runner._attach_live_tracking(trainer, stage="rearrangement")
        callback = trainer.set_metric_callback.call_args.args[0]
        callback("batch", {"loss": 5.0, "epoch": 1, "batch_time": 2.5}, 1)
        callback("batch", {"loss": 4.0, "epoch": 1}, 2)
        callback("batch", {"loss": 3.0, "epoch": 1}, 3)
        callback("epoch", {"log_volume_loss": 3.5, "epoch": 1}, 3)

        self.assertEqual(
            tracker.logs,
            [
                (
                    "rearrangement",
                    {
                        "batch_loss": 5.0,
                        "batch_epoch": 1,
                        "batch_time": 2.5,
                    },
                    1,
                    True,
                ),
                (
                    "rearrangement",
                    {
                        "batch_loss": 3.0,
                        "batch_epoch": 1,
                    },
                    3,
                    True,
                ),
                (
                    "rearrangement",
                    {
                        "epoch_log_volume_loss": 3.5,
                        "epoch": 1,
                    },
                    3,
                    True,
                ),
            ],
        )

    def test_success_and_failure_finish_with_matching_exit_code(self):
        for error, expected_exit_code in ((None, 0), (RuntimeError("boom"), 1)):
            with self.subTest(expected_exit_code=expected_exit_code):
                runner = _runner()
                tracker = _RecordingTracker()

                def execute():
                    runner.tracker = tracker
                    if error is not None:
                        raise error
                    return runner

                runner._run = execute
                if error is None:
                    self.assertIs(runner.run(), runner)
                else:
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        runner.run()

                self.assertEqual(tracker.finish_calls, [expected_exit_code])

    def test_finish_failure_does_not_mask_training_failure(self):
        runner = _runner()
        tracker = mock.Mock(enabled=True)
        tracker.finish.side_effect = RuntimeError("finish failed")

        def execute():
            runner.tracker = tracker
            raise ValueError("training failed")

        runner._run = execute
        with mock.patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "training failed"):
                runner.run()

        tracker.finish.assert_called_once_with(exit_code=1)

    def test_log_failure_disables_further_logs_without_stopping_training(self):
        runner = _runner()
        tracker = mock.Mock(enabled=True)
        tracker.log.side_effect = RuntimeError("logger unavailable")
        trainer = mock.Mock()
        runner.tracker = tracker

        runner._attach_live_tracking(trainer, stage="base")
        callback = trainer.set_metric_callback.call_args.args[0]
        with mock.patch("builtins.print") as print_mock:
            callback("batch", {"loss": 1.0}, 1)
            callback("epoch", {"epoch": 1, "loss": 1.0}, 1)

        tracker.log.assert_called_once()
        self.assertTrue(runner._tracking_log_failed)
        trainer.set_metric_callback.assert_called_with(None)
        print_mock.assert_called_once()

    def test_finish_failure_does_not_fail_a_completed_experiment(self):
        runner = _runner()
        tracker = mock.Mock(enabled=True)
        tracker.finish.side_effect = RuntimeError("finish failed")

        def execute():
            runner.tracker = tracker
            return runner

        runner._run = execute
        with mock.patch("builtins.print") as print_mock:
            self.assertIs(runner.run(), runner)

        tracker.finish.assert_called_once_with(exit_code=0)
        print_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
