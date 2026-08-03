from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from networks.measure_preserving_flows.flow_integration import VectorFieldFlow
from trainers.rearranged_transport import RearrangedTransportTrainer


class _LinearVectorField(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor | None,
        _time: torch.Tensor,
    ) -> torch.Tensor:
        result = self.scale * state
        if context is not None:
            result = result + 0.01 * context
        return result


class FlowIntegrationDiagnosticsTest(unittest.TestCase):

    @staticmethod
    def _make_flow(use_adjoint: bool) -> VectorFieldFlow:
        return VectorFieldFlow(
            vector_field=_LinearVectorField(),
            endpoint_alpha=0.0,
            use_adjoint=use_adjoint,
            method="dopri5",
            rtol=1e-7,
            atol=1e-9,
        ).double()

    @staticmethod
    def _assert_adaptive_counts_are_consistent(
        test_case: unittest.TestCase,
        summary: dict[str, int],
        prefix: str,
    ) -> None:
        attempted = summary[f"{prefix}_attempted_steps"]
        accepted = summary[f"{prefix}_accepted_steps"]
        rejected = summary[f"{prefix}_rejected_steps"]
        test_case.assertGreater(attempted, 0)
        test_case.assertEqual(attempted, accepted + rejected)

    def test_diagnostics_are_disabled_and_absent_from_state_dict_by_default(
        self,
    ) -> None:
        flow = self._make_flow(use_adjoint=False)
        state_before = {
            name: value.clone()
            for name, value in flow.state_dict().items()
        }

        self.assertFalse(flow.solver_diagnostics_enabled)
        self.assertIsNone(flow.solver_diagnostics_summary())

        input_value = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
        output_without_diagnostics = flow(input_value)

        flow.enable_solver_diagnostics()
        output_with_diagnostics = flow(input_value)

        self.assertTrue(flow.solver_diagnostics_enabled)
        torch.testing.assert_close(
            output_with_diagnostics,
            output_without_diagnostics,
        )
        self.assertEqual(set(flow.state_dict()), set(state_before))
        for name, expected in state_before.items():
            torch.testing.assert_close(flow.state_dict()[name], expected)

    def test_dopri5_direct_autograd_counts_only_forward_solver_work(self) -> None:
        flow = self._make_flow(use_adjoint=False)
        flow.enable_solver_diagnostics()
        input_value = torch.tensor(
            [[1.0, -0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )

        output = flow(input_value)
        after_forward = flow.solver_diagnostics_summary()
        self.assertIsNotNone(after_forward)
        assert after_forward is not None
        self.assertGreater(after_forward["forward_nfe"], 0)
        self.assertEqual(after_forward["backward_nfe"], 0)
        self.assertEqual(
            after_forward["total_nfe"],
            after_forward["forward_nfe"],
        )
        self._assert_adaptive_counts_are_consistent(
            self,
            after_forward,
            "forward",
        )

        output.square().sum().backward()
        after_backward = flow.solver_diagnostics_summary()
        self.assertEqual(after_backward, after_forward)
        assert after_backward is not None
        self.assertEqual(after_backward["adjoint_attempted_steps"], 0)
        self.assertEqual(after_backward["adjoint_accepted_steps"], 0)
        self.assertEqual(after_backward["adjoint_rejected_steps"], 0)

    def test_dopri5_adjoint_counts_forward_and_backward_separately(self) -> None:
        flow = self._make_flow(use_adjoint=True)
        flow.enable_solver_diagnostics()
        input_value = torch.tensor(
            [[1.0, -0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )

        output = flow(input_value)
        after_forward = flow.solver_diagnostics_summary()
        self.assertIsNotNone(after_forward)
        assert after_forward is not None
        self.assertGreater(after_forward["forward_nfe"], 0)
        self.assertEqual(after_forward["backward_nfe"], 0)
        self._assert_adaptive_counts_are_consistent(
            self,
            after_forward,
            "forward",
        )

        output.square().sum().backward()
        after_backward = flow.solver_diagnostics_summary()
        self.assertIsNotNone(after_backward)
        assert after_backward is not None
        self.assertGreater(after_backward["backward_nfe"], 0)
        self.assertEqual(
            after_backward["total_nfe"],
            after_backward["forward_nfe"] + after_backward["backward_nfe"],
        )
        self._assert_adaptive_counts_are_consistent(
            self,
            after_backward,
            "adjoint",
        )
        self.assertIsNotNone(input_value.grad)
        self.assertIsNotNone(flow.vector_field.scale.grad)

    def test_adjoint_diagnostics_preserve_outputs_and_gradients(self) -> None:
        reference = self._make_flow(use_adjoint=True)
        instrumented = self._make_flow(use_adjoint=True)
        instrumented.load_state_dict(reference.state_dict())
        instrumented.enable_solver_diagnostics()

        reference_input = torch.tensor(
            [[0.3, -0.7]],
            dtype=torch.float64,
            requires_grad=True,
        )
        instrumented_input = reference_input.detach().clone().requires_grad_(True)

        reference_output = reference(reference_input)
        instrumented_output = instrumented(instrumented_input)
        torch.testing.assert_close(instrumented_output, reference_output)

        reference_output.square().sum().backward()
        instrumented_output.square().sum().backward()
        torch.testing.assert_close(instrumented_input.grad, reference_input.grad)
        torch.testing.assert_close(
            instrumented.vector_field.scale.grad,
            reference.vector_field.scale.grad,
        )

    def test_summary_can_reset_and_diagnostics_can_be_disabled(self) -> None:
        flow = self._make_flow(use_adjoint=False)
        flow.enable_solver_diagnostics()
        flow(torch.ones(1, 2, dtype=torch.float64))

        measured = flow.solver_diagnostics_summary(reset=True)
        self.assertIsNotNone(measured)
        assert measured is not None
        self.assertGreater(measured["total_nfe"], 0)
        self.assertEqual(
            flow.solver_diagnostics_summary(),
            {
                "forward_nfe": 0,
                "backward_nfe": 0,
                "total_nfe": 0,
                "forward_attempted_steps": 0,
                "forward_accepted_steps": 0,
                "forward_rejected_steps": 0,
                "adjoint_attempted_steps": 0,
                "adjoint_accepted_steps": 0,
                "adjoint_rejected_steps": 0,
            },
        )

        flow.disable_solver_diagnostics()
        self.assertFalse(flow.solver_diagnostics_enabled)
        self.assertIsNone(flow.solver_diagnostics_summary())

    def test_rearrangement_live_metrics_include_and_reset_solver_counts(self, ) -> None:
        flow = mock.Mock()
        flow.last_end_time = 0.95
        flow.vector_field = SimpleNamespace(network=None)
        flow.solver_diagnostics_summary.return_value = {
            "forward_nfe": 25,
            "backward_nfe": 37,
            "total_nfe": 62,
        }
        predictor = SimpleNamespace(
            rearrangement_flow=flow,
            device=torch.device("cpu"),
        )

        metrics = RearrangedTransportTrainer._rearrangement_flow_metrics(predictor)

        self.assertEqual(metrics["integration_end_time"], 0.95)
        self.assertEqual(metrics["solver_forward_nfe"], 25)
        self.assertEqual(metrics["solver_backward_nfe"], 37)
        self.assertEqual(metrics["solver_total_nfe"], 62)
        flow.solver_diagnostics_summary.assert_called_once_with(reset=True)


if __name__ == "__main__":
    unittest.main()
