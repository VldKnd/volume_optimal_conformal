import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.calibrators import CDFCalibratorConfig
from configs.conformal import TransportBasedConformalPredictorConfig
from conformal import TransportBasedConformalPredictor
from conformal.calibrators import CDFCalibrator


class _DeterministicDensityTransport:

    x_dim = 1
    y_dim = 1
    device = torch.device("cpu")
    dtype = torch.float64

    def __init__(self) -> None:
        self.sample_sizes: list[int] = []

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
        return -(y[:, 0] - x[:, 0]).square()

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        self.sample_sizes.append(n_samples)
        values = torch.arange(
            n_samples,
            device=x.device,
            dtype=x.dtype,
        )
        return x[:, None, :] + values[None, :, None]


class _DensityTransportWithoutSampling(_DeterministicDensityTransport):
    sample = None


class CDFCalibratorTest(unittest.TestCase):

    def test_calibrates_empirical_cdf_scores_as_nonconformity(self) -> None:
        calibrator = CDFCalibrator(CDFCalibratorConfig(n_cdf_samples=4))
        scores = torch.tensor([0.25, 0.5, 0.75, 1.0])

        calibrator.fit(
            x=torch.zeros(4, 1),
            scores=scores[:, None],
            coverage_mass=0.5,
        )

        self.assertEqual(calibrator.threshold.item(), 0.75)
        torch.testing.assert_close(
            calibrator.contains(
                x=torch.zeros(2, 1),
                scores=torch.tensor([0.75, 1.0]),
            ),
            torch.tensor([True, False]),
        )

    def test_transport_score_matches_requested_density_rank(self) -> None:
        predictor = TransportBasedConformalPredictor(
            predictor=_DeterministicDensityTransport(),
            config=TransportBasedConformalPredictorConfig(
                coverage_mass=0.5,
                calibrator=CDFCalibratorConfig(n_cdf_samples=4),
                volume_n_neighbors=4,
            ),
        )
        x = torch.zeros(4, 1, dtype=torch.float64)
        y = torch.arange(4, dtype=torch.float64)[:, None]
        predictor.calibrate(DataLoader(TensorDataset(x, y), batch_size=2))

        expected = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        torch.testing.assert_close(predictor.cdf_score(x=x, y=y), expected)
        self.assertEqual(predictor.threshold.item(), 0.75)
        torch.testing.assert_close(
            predictor.contains(x=x, y=y),
            torch.tensor([True, True, True, False]),
        )

    def test_requires_conditional_sampling(self) -> None:
        with self.assertRaisesRegex(TypeError, "sample"):
            TransportBasedConformalPredictor(
                predictor=_DensityTransportWithoutSampling(),
                config=TransportBasedConformalPredictorConfig(
                    calibrator=CDFCalibratorConfig(n_cdf_samples=4),
                ),
            )

    def test_cdf_samples_are_evaluated_in_bounded_chunks(self) -> None:
        base_predictor = _DeterministicDensityTransport()
        predictor = TransportBasedConformalPredictor(
            predictor=base_predictor,
            config=TransportBasedConformalPredictorConfig(
                calibrator=CDFCalibratorConfig(
                    n_cdf_samples=4,
                    cdf_batch_size=4,
                ),
            ),
        )

        predictor.cdf_score(
            x=torch.zeros(2, 1, dtype=torch.float64),
            y=torch.zeros(2, 1, dtype=torch.float64),
        )

        self.assertEqual(base_predictor.sample_sizes, [2, 2])

    def test_volume_uses_observation_space_monte_carlo(self) -> None:
        predictor = TransportBasedConformalPredictor(
            predictor=_DeterministicDensityTransport(),
            config=TransportBasedConformalPredictorConfig(
                coverage_mass=0.5,
                calibrator=CDFCalibratorConfig(n_cdf_samples=4),
                volume_mc_samples=100,
                volume_batch_size=13,
                volume_n_neighbors=4,
                volume_seed=3,
            ),
        )
        x = torch.zeros(4, 1, dtype=torch.float64)
        y = torch.arange(4, dtype=torch.float64)[:, None]
        predictor.calibrate(DataLoader(TensorDataset(x, y), batch_size=4))

        volume = predictor.estimate_volume(x=torch.zeros(1, 1, dtype=torch.float64))
        torch.testing.assert_close(
            volume,
            torch.tensor([3.0], dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
