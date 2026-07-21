# src/predictors/transport/flow_matching.py

from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn
from torchdiffeq import odeint

from configs.predictors.transport.flow_matching import FlowMatchingPredictorConfig
from predictors.transport.base import BaseTransportPredictor
from networks.mlp_vector_field import MLPVectorField
from networks.standard_scaler import FrozenStandardScaler


class FlowMatchingPredictor(nn.Module, BaseTransportPredictor):

    def __init__(self, config: FlowMatchingPredictorConfig):
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        torch.manual_seed(config.seed)

        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self.vector_field = MLPVectorField(
            x_dim=config.x_dim,
            y_dim=config.y_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
        ).to(device=self.device, dtype=self.dtype)

        self.y_scaler = FrozenStandardScaler(config.y_dim).to(
            device=self.device,
            dtype=self.dtype,
        )

        self.use_hutchinson_trace_estimator = config.use_hutchinson_trace_estimator
        self.hutchinson_num_samples = config.hutchinson_num_samples
        self.hutchinson_noise = config.hutchinson_noise

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype)

    def _make_time_batch(
        self,
        t: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=self.dtype)
        else:
            t = t.to(device=self.device, dtype=self.dtype)

        return t.reshape(1, 1).expand(batch_size, 1)

    def _ode_time_span(self) -> torch.Tensor:
        return torch.tensor(
            [0.0, 1.0],
            device=self.device,
            dtype=self.dtype,
        )

    def _exact_divergence(
        self,
        velocity: torch.Tensor,
        state: torch.Tensor,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """
        Exact divergence:

            div v(z, t) = Tr[dv / dz]

        Cost: O(y_dim) reverse-mode autodiff calls.
        """
        divergence = torch.zeros(
            state.shape[0],
            device=state.device,
            dtype=state.dtype,
        )

        for j in range(state.shape[1]):
            grad_j = torch.autograd.grad(
                velocity[:, j].sum(),
                state,
                create_graph=create_graph,
                retain_graph=create_graph or (j + 1 < state.shape[1]),
            )[0][:, j]

            divergence = divergence + grad_j

        return divergence

    def _sample_hutchinson_noise(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns Hutchinson probe vectors.

        Shape:

            (num_probes, batch, y_dim)

        The probes are sampled once per ODE solve and reused for all ODE
        function evaluations. This keeps the augmented ODE deterministic
        conditional on the sampled probes.
        """
        shape = (
            self.hutchinson_num_samples,
            state.shape[0],
            state.shape[1],
        )

        if self.hutchinson_noise == "gaussian":
            return torch.randn(
                shape,
                device=state.device,
                dtype=state.dtype,
            )

        if self.hutchinson_noise == "rademacher":
            return (
                torch.empty(
                    shape,
                    device=state.device,
                    dtype=state.dtype,
                ).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            )

        raise ValueError(
            f"Unknown hutchinson_noise={self.hutchinson_noise!r}. "
            "Expected 'rademacher' or 'gaussian'."
        )

    def _hutchinson_divergence(
        self,
        velocity: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """
        Hutchinson estimator for

            div v(z, t) = Tr[dv / dz]

        using

            Tr(J) ≈ mean_k eps_k^T J eps_k.

        Args:
            velocity: (batch, y_dim)
            state:    (batch, y_dim)
            noise:    (num_probes, batch, y_dim)

        Returns:
            divergence estimate: (batch,)
        """
        estimates = []

        num_probes = noise.shape[0]

        for k in range(num_probes):
            eps = noise[k]

            vjp = torch.autograd.grad(
                outputs=velocity,
                inputs=state,
                grad_outputs=eps,
                create_graph=create_graph,
                retain_graph=create_graph or (k + 1 < num_probes),
            )[0]

            estimate = (vjp * eps).sum(dim=1)
            estimates.append(estimate)

        return torch.stack(estimates, dim=0).mean(dim=0)

    def _divergence(
        self,
        velocity: torch.Tensor,
        state: torch.Tensor,
        hutchinson_noise: torch.Tensor | None = None,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """
        Chooses exact trace or Hutchinson trace estimator depending on config.
        """
        if self.use_hutchinson_trace_estimator:
            if hutchinson_noise is None:
                raise ValueError(
                    "hutchinson_noise must be provided when "
                    "use_hutchinson_trace_estimator=True."
                )

            return self._hutchinson_divergence(
                velocity=velocity,
                state=state,
                noise=hutchinson_noise,
                create_graph=create_graph,
            )

        return self._exact_divergence(
            velocity=velocity,
            state=state,
            create_graph=create_graph,
        )

    @torch.no_grad()
    def warmup_y_scaler(self, dataloader) -> None:
        self.y_scaler.reset_running_stats()

        for _, y_batch in dataloader:
            y_batch = self.to_device(y_batch)
            self.y_scaler.update(y_batch)

        self.y_scaler.eval()

    def scale_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.y_scaler(y)

    def unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return (
            y_scaled * torch.sqrt(self.y_scaler.running_var + self.y_scaler.eps) +
            self.y_scaler.running_mean
        )

    def predict_vector_field(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        state = self.to_device(state)
        x = self.to_device(x)
        t = self.to_device(t)

        return self.vector_field(
            state=state,
            x=x,
            t=t,
        )

    @torch.no_grad()
    def pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        ode_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Push latent u to observation space.

            y = T_x(u)

        Args:
            x: (batch, x_dim)
            u: (batch, y_dim)

        Returns:
            y: (batch, y_dim)
        """
        self.eval()

        x = self.to_device(x)
        state = self.to_device(u)

        steps = self.config.ode_steps if ode_steps is None else ode_steps
        time_span = self._ode_time_span()

        def dynamics(
            t_scalar: torch.Tensor,
            state: torch.Tensor,
        ) -> torch.Tensor:
            t = self._make_time_batch(
                t=t_scalar,
                batch_size=state.shape[0],
            )

            return self.predict_vector_field(
                state=state,
                x=x,
                t=t,
            )

        trajectory = odeint(
            dynamics,
            state,
            time_span,
            method="rk4",
            options={"step_size": 1.0 / steps},
        )

        state = trajectory[-1]

        return self.unscale_y(state).detach()

    @torch.no_grad()
    def pushforward_with_log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        ode_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Push latent u to observation space and compute the forward log-det.

            y = T_x(u)

        Returns:

            y:       (batch, y_dim)
            log_det: (batch,)

        The returned log_det is

            log |det D_u T_x(u)|.

        Since pushforward returns unscaled y, this includes the final
        diagonal Jacobian contribution from unscale_y.
        """
        self.eval()

        x = self.to_device(x)
        state = self.to_device(u)

        steps = self.config.ode_steps if ode_steps is None else ode_steps
        time_span = self._ode_time_span()

        log_det = torch.zeros(
            state.shape[0],
            device=self.device,
            dtype=self.dtype,
        )

        hutchinson_noise = None
        if self.use_hutchinson_trace_estimator:
            hutchinson_noise = self._sample_hutchinson_noise(state)

        def augmented_dynamics(
            t_scalar: torch.Tensor,
            augmented_state: tuple[torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            state, log_det = augmented_state

            t = self._make_time_batch(
                t=t_scalar,
                batch_size=state.shape[0],
            )

            with torch.enable_grad():
                state_for_div = state.detach().requires_grad_(True)

                velocity = self.vector_field(
                    state=state_for_div,
                    x=x,
                    t=t,
                )

                divergence = self._divergence(
                    velocity=velocity,
                    state=state_for_div,
                    hutchinson_noise=hutchinson_noise,
                )

            return velocity.detach(), divergence.detach()

        state_trajectory, log_det_trajectory = odeint(
            augmented_dynamics,
            (state, log_det),
            time_span,
            method="rk4",
            options={"step_size": 1.0 / steps},
        )

        state = state_trajectory[-1]
        log_det = log_det_trajectory[-1]

        # Log-det contribution of unscale_y:
        #
        #   y = y_scaled * sqrt(running_var + eps) + running_mean
        #
        # Therefore:
        #
        #   log |det D unscale_y|
        #       = sum_j log sqrt(running_var_j + eps)
        #       = 0.5 * sum_j log(running_var_j + eps)
        scaler_log_det = 0.5 * torch.log(self.y_scaler.running_var +
                                         self.y_scaler.eps).sum()

        log_det = log_det + scaler_log_det

        y = self.unscale_y(state)

        return y.detach(), log_det.detach()

    @torch.enable_grad()
    def _differentiable_pushforward_with_log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        ode_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        x = self.to_device(x)
        state = self.to_device(u)

        steps = self.config.ode_steps if ode_steps is None else ode_steps
        time_span = self._ode_time_span()

        log_det = torch.zeros(
            state.shape[0],
            device=self.device,
            dtype=self.dtype,
        )

        hutchinson_noise = None
        if self.use_hutchinson_trace_estimator:
            hutchinson_noise = self._sample_hutchinson_noise(state)

        def augmented_dynamics(
            t_scalar: torch.Tensor,
            augmented_state: tuple[torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            state, log_det = augmented_state

            t = self._make_time_batch(
                t=t_scalar,
                batch_size=state.shape[0],
            )

            state = state.requires_grad_(True)
            velocity = self.vector_field(
                state=state,
                x=x,
                t=t,
            )
            divergence = self._divergence(
                velocity=velocity,
                state=state,
                hutchinson_noise=hutchinson_noise,
                create_graph=True,
            )

            return velocity, divergence

        state_trajectory, log_det_trajectory = odeint(
            augmented_dynamics,
            (state, log_det),
            time_span,
            method="rk4",
            options={"step_size": 1.0 / steps},
        )

        state = state_trajectory[-1]
        log_det = log_det_trajectory[-1]
        scaler_log_det = 0.5 * torch.log(self.y_scaler.running_var +
                                         self.y_scaler.eps).sum()

        return self.unscale_y(state), log_det + scaler_log_det

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        ode_steps: int | None = None,
    ) -> torch.Tensor:
        _, log_det = self._differentiable_pushforward_with_log_det(
            x=x,
            u=u,
            ode_steps=ode_steps,
        )
        return log_det

    @torch.no_grad()
    def pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ode_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Pull observation y back to latent space.

            u = T_x^{-1}(y)

        Args:
            x: (batch, x_dim)
            y: (batch, y_dim)

        Returns:
            u: (batch, y_dim)
        """
        self.eval()

        x = self.to_device(x)
        state = self.scale_y(self.to_device(y))

        steps = self.config.ode_steps if ode_steps is None else ode_steps
        time_span = self._ode_time_span()

        def reverse_dynamics(
            s_scalar: torch.Tensor,
            state: torch.Tensor,
        ) -> torch.Tensor:
            # Artificial time s in [0, 1], physical flow time t = 1 - s.
            t_scalar = 1.0 - s_scalar

            t = self._make_time_batch(
                t=t_scalar,
                batch_size=state.shape[0],
            )

            velocity = self.predict_vector_field(
                state=state,
                x=x,
                t=t,
            )

            return -velocity

        trajectory = odeint(
            reverse_dynamics,
            state,
            time_span,
            method="rk4",
            options={"step_size": 1.0 / steps},
        )

        state = trajectory[-1]

        return state.detach()

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default transport score:

            S_multi(x, y) = T_x^{-1}(y)
        """
        return self.pullback(x=x, y=y)

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        x = self.to_device(x)

        batch_size = x.shape[0]

        u = torch.randn(
            batch_size,
            n_samples,
            self.y_dim,
            device=self.device,
            dtype=self.dtype,
        )

        x_rep = (
            x[:,
              None, :].expand(batch_size, n_samples,
                              self.x_dim).reshape(batch_size * n_samples, self.x_dim)
        )

        u_flat = u.reshape(batch_size * n_samples, self.y_dim)

        y_flat = self.pushforward(x=x_rep, u=u_flat)

        return y_flat.reshape(batch_size, n_samples, self.y_dim)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.model_dump(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> Self:
        data = torch.load(path, map_location=map_location, weights_only=False)

        config = FlowMatchingPredictorConfig.model_validate(data["config"])
        model = cls(config)
        model.load_state_dict(data["state_dict"])
        model.to(device=model.device, dtype=model.dtype)

        return model
