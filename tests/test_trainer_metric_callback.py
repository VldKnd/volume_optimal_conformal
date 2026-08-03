from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from trainers.base import BaseTrainer


class _TrainerConfig(BaseModel):
    epochs: int = 3


class _Trainer(BaseTrainer):
    config_class = _TrainerConfig
    trainer_type = "metric_callback_test_trainer"

    def fit(self, predictor, *args, **kwargs):
        del args, kwargs
        return predictor


class TrainerMetricCallbackTest(unittest.TestCase):

    def test_callback_is_live_runtime_state_and_is_not_checkpointed(self) -> None:
        trainer = _Trainer(_TrainerConfig())
        events = []
        trainer.set_metric_callback(
            lambda event, metrics, step: events.append((event, metrics, step))
        )

        trainer.global_step = 2
        trainer._record_batch({"loss": 2.0})
        trainer._record_epoch({"epoch": 1, "loss": 1.5})

        self.assertEqual(
            events,
            [
                ("batch", {
                    "loss": 2.0
                }, 2),
                ("epoch", {
                    "epoch": 1,
                    "loss": 1.5
                }, 2),
            ],
        )
        self.assertNotIn("metric_callback", trainer.state_dict())

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "trainer.pt"
            trainer.save(str(checkpoint))
            loaded = _Trainer.load(str(checkpoint))

        self.assertIsNone(loaded._metric_callback)
        self.assertEqual(loaded.training_history, [{"epoch": 1, "loss": 1.5}])
        self.assertEqual(loaded.completed_epochs, 1)
        self.assertEqual(loaded.global_step, 2)

        resumed_events = []
        loaded.set_metric_callback(
            lambda event, metrics, step: resumed_events.append((event, metrics, step))
        )
        loaded.global_step = 3
        loaded._record_epoch({"epoch": 2, "loss": 1.0})

        self.assertEqual(
            resumed_events,
            [("epoch", {
                "epoch": 2,
                "loss": 1.0
            }, 3)],
        )


if __name__ == "__main__":
    unittest.main()
