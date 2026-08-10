import tempfile
import unittest
from pathlib import Path

import torch

from configs.predictors.transport import NormalizingFlowPredictorConfig
from predictors.transport import NormalizingFlowPredictor


class NormalizingFlowCheckpointDeviceTest(unittest.TestCase):

    def test_map_location_overrides_saved_cuda_device(self) -> None:
        source = NormalizingFlowPredictor(
            NormalizingFlowPredictorConfig(
                x_dim=1,
                y_dim=2,
                hidden_dim=2,
                num_hidden_layers=1,
                num_flow_layers=1,
                device="cpu",
            )
        )
        saved_config = source.config.model_dump()
        saved_config["device"] = "cuda"

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "predictor.pt"
            torch.save(
                {
                    "config": saved_config,
                    "state_dict": source.state_dict(),
                },
                checkpoint,
            )
            loaded = NormalizingFlowPredictor.load(
                checkpoint,
                map_location="cpu",
            )

        self.assertEqual(loaded.config.device, "cpu")
        self.assertEqual(loaded.device, torch.device("cpu"))
        self.assertTrue(
            all(parameter.device.type == "cpu" for parameter in loaded.parameters())
        )


if __name__ == "__main__":
    unittest.main()
