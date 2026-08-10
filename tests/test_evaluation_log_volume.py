import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation import log_volume


class _FixedVolumePredictor:

    y_dim = 2

    def __init__(self, log_volumes: torch.Tensor):
        self.log_volumes = log_volumes

    def estimate_log_volume(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_volumes[:x.shape[0]]


class LogVolumeTest(unittest.TestCase):

    def setUp(self) -> None:
        x = torch.zeros(4, 1)
        y = torch.zeros(4, 1)
        self.dataloader = DataLoader(TensorDataset(x, y), batch_size=4)

    def test_excludes_exactly_zero_volume_estimates(self) -> None:
        predictor = _FixedVolumePredictor(
            torch.tensor([
                math.log(4.0),
                -torch.inf,
                math.log(16.0),
                -torch.inf,
            ])
        )

        mean, std = log_volume(self.dataloader, predictor)

        self.assertAlmostEqual(mean, 1.5 * math.log(2.0))
        self.assertAlmostEqual(std, 0.5 * math.log(2.0))

    def test_returns_negative_infinity_when_all_volumes_are_zero(self) -> None:
        predictor = _FixedVolumePredictor(torch.full((4, ), -torch.inf))

        mean, std = log_volume(self.dataloader, predictor)

        self.assertEqual(mean, -torch.inf)
        self.assertEqual(std, 0.0)


if __name__ == "__main__":
    unittest.main()
