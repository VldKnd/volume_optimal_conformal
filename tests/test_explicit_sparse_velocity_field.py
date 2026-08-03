from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

from networks.measure_preserving_flows.explicit_sparse_velocity_field import (
    ExplicitSparseGaussianSkewVectorField,
)
from networks.measure_preserving_flows.mlp import ScaledTanh
from networks.measure_preserving_flows.sparse_skew_symmetric_vector_field import (
    SparseGaussianSkewVectorField,
)


class ExplicitSparseVelocityFieldSmokeTest(unittest.TestCase):

    def test_scaled_tanh_is_trainable_and_bounded(self) -> None:
        activation = ScaledTanh(initial_scale=2.5).double()
        inputs = torch.tensor([-100.0, 0.4, 100.0], dtype=torch.float64)
        outputs = activation(inputs)

        self.assertEqual(activation.alpha.shape, torch.Size([]))
        self.assertTrue(activation.alpha.requires_grad)
        self.assertGreater(activation.scale.item(), 0.0)
        self.assertTrue((outputs.abs() <= activation.scale.abs()).all())

        outputs[1].backward()
        torch.testing.assert_close(
            activation.alpha.grad,
            activation.scale.detach() * torch.tanh(inputs[1]),
        )

    def _make_matching_fields(
        self,
        activation: str,
    ) -> tuple[
        SparseGaussianSkewVectorField,
        ExplicitSparseGaussianSkewVectorField,
    ]:
        field_kwargs = {
            "dimension": 5,
            "hidden_dimension": 6,
            "number_of_hidden_layers": 2,
            "context_dimension": 3,
            "time_dependent": True,
            "activation": activation,
            "activation_power": 2.0,
        }
        reference = SparseGaussianSkewVectorField(**field_kwargs).double()
        explicit = ExplicitSparseGaussianSkewVectorField(**field_kwargs).double()
        self.assertIsInstance(reference.network.net[-1], ScaledTanh)
        self.assertIsInstance(explicit.network.net[-1], ScaledTanh)

        generator = torch.Generator().manual_seed(731)
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(mean=0.0, std=0.2, generator=generator)
            reference.network.net[-1].alpha.fill_(math.log(1.7))

        self.assertEqual(
            list(reference.state_dict()),
            list(explicit.state_dict()),
        )
        explicit.load_state_dict(reference.state_dict())
        return reference, explicit

    def test_matches_autograd_velocity_and_training_gradients(self) -> None:
        generator = torch.Generator().manual_seed(913)

        for activation in (
            "elu",
            "gelu",
            "leaky_relu",
            "prelu",
            "relu",
            "silu",
            "softplus",
            "tanh",
        ):
            with self.subTest(activation=activation):
                reference, explicit = self._make_matching_fields(activation)
                u_value = torch.randn(3, 5, dtype=torch.float64, generator=generator)
                x_value = torch.randn(3, 3, dtype=torch.float64, generator=generator)
                projection = torch.randn(
                    3,
                    5,
                    dtype=torch.float64,
                    generator=generator,
                )

                reference_u = u_value.clone().requires_grad_(True)
                explicit_u = u_value.clone().requires_grad_(True)
                reference_x = x_value.clone().requires_grad_(True)
                explicit_x = x_value.clone().requires_grad_(True)
                reference_t = torch.tensor(
                    0.37,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                explicit_t = reference_t.detach().clone().requires_grad_(True)

                reference_velocity = reference(
                    u=reference_u,
                    x=reference_x,
                    t=reference_t,
                )
                explicit_velocity = explicit(
                    u=explicit_u,
                    x=explicit_x,
                    t=explicit_t,
                )
                torch.testing.assert_close(
                    explicit_velocity,
                    reference_velocity,
                    rtol=1e-10,
                    atol=1e-11,
                )

                reference_inputs = (
                    reference_u,
                    reference_x,
                    reference_t,
                    *tuple(reference.parameters()),
                )
                explicit_inputs = (
                    explicit_u,
                    explicit_x,
                    explicit_t,
                    *tuple(explicit.parameters()),
                )
                reference_gradients = torch.autograd.grad(
                    (reference_velocity * projection).sum(),
                    reference_inputs,
                )
                explicit_gradients = torch.autograd.grad(
                    (explicit_velocity * projection).sum(),
                    explicit_inputs,
                )

                for explicit_gradient, reference_gradient in zip(
                    explicit_gradients,
                    reference_gradients,
                    strict=True,
                ):
                    torch.testing.assert_close(
                        explicit_gradient,
                        reference_gradient,
                        rtol=2e-9,
                        atol=2e-10,
                    )

    def test_uses_only_two_final_derivatives_without_autograd(self) -> None:
        _, explicit = self._make_matching_fields("softplus")
        u = torch.randn(4, 5, dtype=torch.float64)
        x = torch.randn(4, 3, dtype=torch.float64)
        time = torch.full((4, 1), 0.2, dtype=torch.float64)

        potential, left_derivatives, right_derivatives = (
            explicit.network.forward_with_edge_derivatives(
                state=u,
                x=x,
                t=time,
            )
        )
        self.assertEqual(potential.shape, (4, 4))
        self.assertEqual(left_derivatives.shape, (4, 4))
        self.assertEqual(right_derivatives.shape, (4, 4))

        with mock.patch(
            "torch.autograd.grad",
            side_effect=AssertionError("explicit field called torch.autograd.grad"),
        ):
            with torch.no_grad():
                velocity = explicit(u=u, x=x, t=0.2)

        self.assertEqual(velocity.shape, u.shape)
        self.assertFalse(velocity.requires_grad)


if __name__ == "__main__":
    unittest.main()
