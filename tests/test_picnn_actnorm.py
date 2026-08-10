import unittest

import torch

from networks.picnn import ActNorm


class ActNormTest(unittest.TestCase):

    def test_initialization_stores_log_data_scale(self):
        inputs = torch.tensor(
            [
                [[1.0, 3.0], [2.0, 7.0]],
                [[5.0, 9.0], [8.0, 13.0]],
            ],
            dtype=torch.float64,
        )
        layer = ActNorm(feature_dimension=2).to(dtype=torch.float64)

        outputs = layer(inputs)

        reduce_dims = (0, 1)
        data_std = inputs.std(dim=reduce_dims, unbiased=False)
        expected_scale = 1.0 / (data_std + layer.epsilon)
        expected_bias = inputs.mean(dim=reduce_dims) * expected_scale
        torch.testing.assert_close(layer.log_scale, expected_scale.log())
        torch.testing.assert_close(layer.scale, expected_scale)
        torch.testing.assert_close(
            outputs,
            inputs * expected_scale - expected_bias,
        )

    def test_effective_scale_is_strictly_positive(self):
        layer = ActNorm(feature_dimension=3)
        with torch.no_grad():
            layer.log_scale.copy_(torch.tensor([-20.0, 0.0, 5.0]))
            layer.initialized.fill_(True)

        self.assertTrue(torch.all(layer.scale > 0.0))
        self.assertIn("log_scale", layer.state_dict())
        self.assertNotIn("scale", layer.state_dict())

    def test_loads_legacy_scale_checkpoint(self):
        legacy_scale = torch.tensor([0.5, 2.0])
        layer = ActNorm(feature_dimension=2)

        layer.load_state_dict(
            {
                "scale": legacy_scale,
                "bias": torch.tensor([1.0, -1.0]),
                "initialized": torch.tensor(True),
            }
        )

        torch.testing.assert_close(layer.scale, legacy_scale)
        torch.testing.assert_close(layer.log_scale, legacy_scale.log())
        self.assertTrue(layer.initialized.item())

    def test_rejects_non_positive_legacy_scale(self):
        layer = ActNorm(feature_dimension=2)

        with self.assertRaisesRegex(RuntimeError, "non-positive values"):
            layer.load_state_dict(
                {
                    "scale": torch.tensor([-0.5, 2.0]),
                    "bias": torch.zeros(2),
                    "initialized": torch.tensor(True),
                }
            )


if __name__ == "__main__":
    unittest.main()
