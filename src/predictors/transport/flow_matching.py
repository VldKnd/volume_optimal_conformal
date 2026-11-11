# src/predictors/transport/flow_matching.py

from __future__ import annotations

import math
from typing import Self

import torch
import torch.nn as nn
from torchdiffeq import odeint

from configs.predictors.transport.flow_matching import FlowMatchingPredictorConfig
from predictors.transport.base import (
    BaseTransportPredictor,
    checkpoint_config_on_device,
)
from networks.mlp_vector_field import MLPVectorField
from networks.standard_scaler import FrozenStandardScaler


class FlowMatchingPredictor(nn.Module, BaseTransportPredictor):
    """
    Conditional flow matching transport predictor.

    A learned vector field v(z, x, t) defines an ODE on t in [0, 1] that moves
    standard Gaussian latent noise u = z_0 to scaled targets z_1. The public
    transport map is

        y = T_x(u),

    where u is standard Gaussian latent noise and y is returned in the original
    target coordinates. Internally, the flow maps u to scaled y, matching the
    scaler convention used by the other transport predictors.

    Log-determinants and conditional log densities follow from the
    instantaneous change-of-variables formula: the divergence of the vector
    field is integrated along the same fixed-step RK4 trajectory as the state.
    """

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

    def _ode_options(self, ode_steps: int | None) -> dict[str, float]:
        steps = self.config.ode_steps if ode_steps is None else ode_steps

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"ode_steps must be a positive integer, got {steps!r}.")

        return {"step_size": 1.0 / steps}

    @staticmethod
    def _physical_time(
        t_scalar: torch.Tensor,
        reverse: bool,
    ) -> torch.Tensor:
        # Reverse solves run in artificial time s in [0, 1] with physical flow
        # time t = 1 - s.
        return 1.0 - t_scalar if reverse else t_scalar

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
        use_hutchinson_trace_estimator: bool | None = None,
    ) -> torch.Tensor:
        """
        Chooses exact trace or Hutchinson trace estimator depending on config,
        unless ``use_hutchinson_trace_estimator`` overrides it for this call.
        """
        if use_hutchinson_trace_estimator is None:
            use_hutchinson_trace_estimator = self.use_hutchinson_trace_estimator

        if use_hutchinson_trace_estimator:
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

    def _unscale_y_log_det(self) -> torch.Tensor:
        return 0.5 * torch.log(self.y_scaler.running_var + self.y_scaler.eps).sum()

    def _scale_y_log_det(self) -> torch.Tensor:
        return -self._unscale_y_log_det()

    def _validate_condition_and_point_shapes(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got shape {tuple(x.shape)}.")

        if point.ndim != 2:
            raise ValueError(
                f"Expected point to be 2D, got shape {tuple(point.shape)}."
            )

        if x.shape[-1] != self.x_dim:
            raise ValueError(f"Expected x.shape[-1] = {self.x_dim}, got {x.shape[-1]}.")

        if point.shape[-1] != self.y_dim:
            raise ValueError(
                f"Expected point.shape[-1] = {self.y_dim}, "
                f"got {point.shape[-1]}."
            )

        if x.shape[0] != point.shape[0]:
            raise ValueError(
                f"Expected x.shape[0] == point.shape[0], got "
                f"{x.shape[0]} and {point.shape[0]}."
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

    def _integrate(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        *,
        reverse: bool,
        ode_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Integrate the flow ODE from t = 0 to t = 1 (``reverse=False``, latent
        to scaled target) or from t = 1 back to t = 0 (``reverse=True``, scaled
        target to latent) with fixed-step RK4.
        """
        sign = -1.0 if reverse else 1.0

        def dynamics(
            t_scalar: torch.Tensor,
            current_state: torch.Tensor,
        ) -> torch.Tensor:
            t = self._make_time_batch(
                t=self._physical_time(t_scalar, reverse),
                batch_size=current_state.shape[0],
            )

            return sign * self.vector_field(
                state=current_state,
                x=x,
                t=t,
            )

        trajectory = odeint(
            dynamics,
            state,
            self._ode_time_span(),
            method="rk4",
            options=self._ode_options(ode_steps),
        )

        return trajectory[-1]

    def _integrate_with_log_det(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        *,
        reverse: bool,
        ode_steps: int | None = None,
        create_graph: bool = False,
        use_hutchinson_trace_estimator: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the state together with the accumulated log-determinant

            d/dt log |det D z_t| = div v(z_t, x, t).

        For ``reverse=False`` the second output is log |det D_u T_x(u)| in
        scaled coordinates. For ``reverse=True`` the ODE is solved backwards and
        the second output is log |det D_y T_x^{-1}(y)| in scaled coordinates,
        i.e. the negative divergence integral along the trajectory.

        With ``create_graph=False`` every ODE function evaluation is detached,
        so the solve retains no autograd graph. With ``create_graph=True`` the
        second-order graph is kept and both outputs are differentiable with
        respect to ``state``.

        ``ode_steps`` and ``use_hutchinson_trace_estimator`` override the
        configured values for this solve only.
        """
        sign = -1.0 if reverse else 1.0

        if use_hutchinson_trace_estimator is None:
            use_hutchinson_trace_estimator = self.use_hutchinson_trace_estimator

        log_det = torch.zeros(
            state.shape[0],
            device=state.device,
            dtype=state.dtype,
        )

        hutchinson_noise = None
        if use_hutchinson_trace_estimator:
            hutchinson_noise = self._sample_hutchinson_noise(state)

        def augmented_dynamics(
            t_scalar: torch.Tensor,
            augmented_state: tuple[torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            current_state, _ = augmented_state

            t = self._make_time_batch(
                t=self._physical_time(t_scalar, reverse),
                batch_size=current_state.shape[0],
            )

            with torch.enable_grad():
                state_for_div = (
                    current_state if create_graph else current_state.detach()
                )
                if not state_for_div.requires_grad:
                    state_for_div = state_for_div.requires_grad_(True)

                velocity = self.vector_field(
                    state=state_for_div,
                    x=x,
                    t=t,
                )

                divergence = self._divergence(
                    velocity=velocity,
                    state=state_for_div,
                    hutchinson_noise=hutchinson_noise,
                    create_graph=create_graph,
                    use_hutchinson_trace_estimator=use_hutchinson_trace_estimator,
                )

            if not create_graph:
                velocity = velocity.detach()
                divergence = divergence.detach()

            return sign * velocity, sign * divergence

        state_trajectory, log_det_trajectory = odeint(
            augmented_dynamics,
            (state, log_det),
            self._ode_time_span(),
            method="rk4",
            options=self._ode_options(ode_steps),
        )

        return state_trajectory[-1], log_det_trajectory[-1]

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
        self._validate_condition_and_point_shapes(x=x, point=state)

        y_scaled = self._integrate(
            x=x,
            state=state,
            reverse=False,
            ode_steps=ode_steps,
        )

        return self.unscale_y(y_scaled).detach()

    @torch.no_grad()
    def pushforward_with_log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        ode_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Push latent u to observation space and compute the forward log-det.

        Returns:

            y:       (batch, y_dim)
            log_det: (batch,)

        The returned log_det is log |det D_u T_x(u)| for the public unscaled
        map, so it includes the diagonal Jacobian contribution of unscale_y.
        """
        self.eval()

        x = self.to_device(x)
        state = self.to_device(u)
        self._validate_condition_and_point_shapes(x=x, point=state)

        y_scaled, log_det = self._integrate_with_log_det(
            x=x,
            state=state,
            reverse=False,
            ode_steps=ode_steps,
            create_graph=False,
        )
        y = self.unscale_y(y_scaled)

        return y.detach(), (log_det + self._unscale_y_log_det()).detach()

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
        y = self.to_device(y)
        self._validate_condition_and_point_shapes(x=x, point=y)

        u = self._integrate(
            x=x,
            state=self.scale_y(y),
            reverse=True,
            ode_steps=ode_steps,
        )

        return u.detach()

    @torch.no_grad()
    def pullback_with_log_det(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ode_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pull observation y back to latent space and compute the inverse log-det.

        Returns:

            u:       (batch, y_dim)
            log_det: (batch,)

        The returned log_det is log |det D_y T_x^{-1}(y)| for the public map,
        so it includes the diagonal Jacobian contribution of scale_y.
        """
        self.eval()

        x = self.to_device(x)
        y = self.to_device(y)
        self._validate_condition_and_point_shapes(x=x, point=y)

        u, log_det = self._integrate_with_log_det(
            x=x,
            state=self.scale_y(y),
            reverse=True,
            ode_steps=ode_steps,
            create_graph=False,
        )

        return u.detach(), (log_det + self._scale_y_log_det()).detach()

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        ode_steps: int | None = None,
        use_hutchinson_trace_estimator: bool | None = None,
    ) -> torch.Tensor:
        """
        Return log |det D_u T_x(u)| for the public unscaled pushforward map.

        When ``u`` requires gradients and grad mode is enabled, the result stays
        differentiable with respect to ``u``; the rearrangement trainers rely
        on this. Otherwise the ODE solve retains no graph and the result is
        detached, matching the other transport predictors.

        ``ode_steps`` and ``use_hutchinson_trace_estimator`` override the
        configured solver and divergence estimator for this call only. The
        rearrangement trainers use them for a cheaper differentiable estimate
        during training while evaluation keeps the configured exact solve.
        """
        differentiable = torch.is_grad_enabled() and u.requires_grad

        self.eval()

        x = self.to_device(x)
        state = self.to_device(u)
        self._validate_condition_and_point_shapes(x=x, point=state)

        with torch.set_grad_enabled(differentiable):
            _, log_det = self._integrate_with_log_det(
                x=x,
                state=state,
                reverse=False,
                ode_steps=ode_steps,
                create_graph=differentiable,
                use_hutchinson_trace_estimator=use_hutchinson_trace_estimator,
            )
            log_det = log_det + self._unscale_y_log_det()

        return log_det if differentiable else log_det.detach()

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ode_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Conditional log density log p(y | x) under the flow ODE.

        The observation is pulled back to the latent u along the reverse ODE
        while the negative divergence of the vector field is accumulated, and
        the standard Gaussian base density is added. With the exact divergence
        this is the log density of the learned flow up to ODE discretisation
        error; with the Hutchinson estimator the log-determinant term is an
        unbiased estimate.

        Gradient handling mirrors :meth:`log_det`: the result is differentiable
        with respect to ``y`` only when ``y`` requires gradients and grad mode
        is enabled.
        """
        differentiable = torch.is_grad_enabled() and y.requires_grad

        self.eval()

        x = self.to_device(x)
        y = self.to_device(y)
        self._validate_condition_and_point_shapes(x=x, point=y)

        with torch.set_grad_enabled(differentiable):
            u, inverse_log_det = self._integrate_with_log_det(
                x=x,
                state=self.scale_y(y),
                reverse=True,
                ode_steps=ode_steps,
                create_graph=differentiable,
            )
            inverse_log_det = inverse_log_det + self._scale_y_log_det()

            base_log_prob = (-0.5 * u.square() -
                             0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
            log_prob = base_log_prob + inverse_log_det

        return log_prob if differentiable else log_prob.detach()

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

        config = FlowMatchingPredictorConfig.model_validate(
            checkpoint_config_on_device(data["config"], map_location)
        )
        model = cls(config)
        model.load_state_dict(data["state_dict"])
        model.to(device=model.device, dtype=model.dtype)
        model.eval()

        return model
