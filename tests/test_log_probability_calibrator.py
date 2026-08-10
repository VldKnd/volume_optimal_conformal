import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.calibrators import LogProbabilityCalibratorConfig
from configs.conformal import TransportBasedConformalPredictorConfig
from conformal import TransportBasedConformalPredictor
from conformal.calibrators import LogProbabilityCalibrator


class _DensityTransport:

    x_dim = 1
    y_dim = 1
    device = torch.device("cpu")
    dtype = torch.float64

    def __init__(self) -> None:
        self.log_prob_batch_sizes: list[int] = []

    def eval(self) -> None:
        return None

    def pushforward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def pullback(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x

    def log_det(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        del x
        return torch.zeros(u.shape[0], dtype=u.dtype, device=u.device)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.log_prob_batch_sizes.append(x.shape[0])
        return -(y[:, 0] - x[:, 0]).square()


class _TransportWithoutDensity:

    x_dim = 1
    y_dim = 1
    device = torch.device("cpu")
    dtype = torch.float32

    def pushforward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        del x
        return u

    def pullback(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del x
        return y

    def log_det(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        del x
        return torch.zeros(u.shape[0], dtype=u.dtype, device=u.device)


class LogProbabilityCalibratorTest(unittest.TestCase):

    def test_stores_lower_log_probability_cutoff(self) -> None:
        calibrator = LogProbabilityCalibrator(LogProbabilityCalibratorConfig())
        log_probabilities = torch.tensor([0.0, -1.0, -4.0, -9.0])

        calibrator.fit(
            x=torch.zeros(4, 1),
            scores=log_probabilities[:, None],
            coverage_mass=0.5,
        )

        self.assertEqual(calibrator.threshold.item(), -4.0)
        inside = calibrator.contains(
            x=torch.zeros(3, 1),
            scores=torch.tensor([-1.0, -4.0, -9.0]),
        )
        torch.testing.assert_close(
            inside,
            torch.tensor([True, True, False]),
        )

    def test_transport_predictor_uses_log_prob_for_membership(self) -> None:
        predictor = _DensityTransport()
        conformal_predictor = TransportBasedConformalPredictor(
            predictor=predictor,
            config=TransportBasedConformalPredictorConfig(
                coverage_mass=0.5,
                calibrator=LogProbabilityCalibratorConfig(),
                volume_n_neighbors=4,
            ),
        )
        x = torch.zeros(4, 1, dtype=torch.float64)
        y = torch.arange(4, dtype=torch.float64)[:, None]
        conformal_predictor.calibrate(DataLoader(TensorDataset(x, y), batch_size=2))

        self.assertEqual(conformal_predictor.threshold.item(), -4.0)
        query_y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        torch.testing.assert_close(
            conformal_predictor.scalar_score(x=torch.zeros_like(query_y), y=query_y),
            torch.tensor([-1.0, -4.0, -9.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            conformal_predictor.contains(x=torch.zeros_like(query_y), y=query_y),
            torch.tensor([True, True, False]),
        )

    def test_requires_base_predictor_log_prob(self) -> None:
        with self.assertRaisesRegex(TypeError, "log_prob"):
            TransportBasedConformalPredictor(
                predictor=_TransportWithoutDensity(),
                config=TransportBasedConformalPredictorConfig(
                    calibrator=LogProbabilityCalibratorConfig(),
                ),
            )

    def test_volume_uses_bounding_box_rejection_monte_carlo(self) -> None:
        predictor = _DensityTransport()
        conformal_predictor = TransportBasedConformalPredictor(
            predictor=predictor,
            config=TransportBasedConformalPredictorConfig(
                coverage_mass=0.5,
                calibrator=LogProbabilityCalibratorConfig(),
                volume_mc_samples=20_000,
                volume_batch_size=127,
                volume_n_neighbors=5,
                volume_seed=7,
            ),
        )
        x = torch.zeros(5, 1, dtype=torch.float64)
        y = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        conformal_predictor.calibrate(DataLoader(TensorDataset(x, y), batch_size=5))
        predictor.log_prob_batch_sizes.clear()

        log_volume = conformal_predictor.estimate_log_volume(
            torch.zeros(1, 1, dtype=torch.float64)
        )

        self.assertAlmostEqual(math.exp(log_volume.item()), 2.0, delta=0.05)
        self.assertLessEqual(max(predictor.log_prob_batch_sizes), 127)


if __name__ == "__main__":
    unittest.main()
