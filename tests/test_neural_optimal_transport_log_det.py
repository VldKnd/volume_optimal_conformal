import inspect
import unittest

import torch
import torch.nn as nn

from configs.predictors.transport import NeuralOptimalTransportPredictorConfig
from predictors.transport import NeuralOptimalTransportPredictor


class _DiagonalQuadraticPotential(nn.Module):

    def __init__(self, curvature: torch.Tensor):
        super().__init__()
        self.register_buffer("curvature", curvature)

    def forward(
        self,
        condition: torch.Tensor,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        del condition
        return 0.5 * (self.curvature * tensor.square()).sum(
            dim=-1,
            keepdim=True,
        )


class NeuralOptimalTransportLogDetTest(unittest.TestCase):

    def test_log_det_jitter_defaults_are_zero(self) -> None:
        self.assertEqual(
            inspect.signature(NeuralOptimalTransportPredictor.log_det)
            .parameters["jitter"]
            .default,
            0.0,
        )
        self.assertEqual(
            inspect.signature(
                NeuralOptimalTransportPredictor.estimate_log_det_d2_phi
            ).parameters["jitter"].default,
            0.0,
        )

    def test_default_uses_unmodified_hessian(self) -> None:
        curvature = torch.tensor([0.5, 2.0], dtype=torch.float64)
        predictor = NeuralOptimalTransportPredictor(
            NeuralOptimalTransportPredictorConfig(
                x_dim=1,
                y_dim=2,
                hidden_dim=2,
                num_hidden_layers=1,
                dtype="float64",
            )
        )
        predictor.potential_network = _DiagonalQuadraticPotential(curvature)

        x = torch.zeros(3, 1, dtype=torch.float64)
        u = torch.tensor(
            [[0.0, 0.0], [1.0, -2.0], [-0.5, 0.25]],
            dtype=torch.float64,
        )

        actual = predictor.estimate_log_det_d2_phi(x=x, u=u)
        expected = curvature.log().sum().expand_as(actual)
        torch.testing.assert_close(actual, expected)

        jitter = 0.25
        jittered = predictor.estimate_log_det_d2_phi(
            x=x,
            u=u,
            jitter=jitter,
        )
        expected_jittered = (curvature + jitter).log().sum().expand_as(jittered)
        torch.testing.assert_close(jittered, expected_jittered)


if __name__ == "__main__":
    unittest.main()
