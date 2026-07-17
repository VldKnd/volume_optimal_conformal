# src/predictors/transport/neural_optimal_transport.py

from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn

from configs.predictors.transport.neural_optimal_transport import (
    NeuralOptimalTransportPredictorConfig,
)
from predictors.transport.base import BaseTransportPredictor
from networks.picnn import PISCNN


class NeuralOptimalTransportPredictor(nn.Module, BaseTransportPredictor):
    """
    Conditional neural quantile regression predictor.

    It learns a conditional convex potential and defines

        pushforward(x, u) = T_x(u),
        pullback(x, y)    = T_x^{-1}(y).

    The multivariate score is

        S_multi(x, y) = pullback(x, y).
    """

    def __init__(self, config: NeuralOptimalTransportPredictorConfig):
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        torch.manual_seed(config.seed)

        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self.potential_type = config.potential_type

        self.potential_network = PISCNN(
            feature_dimension=config.x_dim,
            response_dimension=config.y_dim,
            hidden_dimension=config.hidden_dim,
            number_of_hidden_layers=config.num_hidden_layers,
        ).to(device=self.device, dtype=self.dtype)

        self.y_scaler = nn.BatchNorm1d(
            config.y_dim,
            affine=False,
        ).to(device=self.device, dtype=self.dtype)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def warmup_y_scaler(self, dataloader) -> None:
        self.y_scaler.train()

        for _, y_batch in dataloader:
            y_batch = self.to_device(y_batch)
            _ = self.y_scaler(y_batch)

        self.y_scaler.eval()

    def scale_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.y_scaler(y)

    def unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return (
            y_scaled * torch.sqrt(self.y_scaler.running_var + self.y_scaler.eps) +
            self.y_scaler.running_mean
        )

    @torch.enable_grad()
    def gradient_inverse(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes gradient of the learned potential wrt point.
        """
        x = self.to_device(x)
        point = self.to_device(point).detach().clone().requires_grad_(True)

        potential = self.potential_network.forward(
            condition=x,
            tensor=point,
        ).sum()

        grad = torch.autograd.grad(
            potential,
            point,
            create_graph=False,
        )[0]

        return grad.detach()

    def c_transform_inverse(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximately solves

            argmin_z phi_x(z) - <point, z>

        using LBFGS, with Adam fallback.
        """
        x = self.to_device(x)
        point = self.to_device(point)

        inverse = torch.nn.Parameter(point.detach().clone().contiguous())

        optimizer = torch.optim.LBFGS(
            [inverse],
            lr=self.config.c_transform_lr,
            max_iter=self.config.c_transform_max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-7,
        )

        def closure():
            optimizer.zero_grad()

            inner = (point * inverse).sum(dim=-1, keepdim=True)
            potential = self.potential_network.forward(
                condition=x,
                tensor=inverse,
            )

            objective = (potential - inner).mean()

            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite c-transform objective.")

            objective.backward()
            torch.nn.utils.clip_grad_norm_([inverse], max_norm=10.0)

            return objective

        try:
            optimizer.step(closure)

        except (FloatingPointError, RuntimeError):
            inverse = torch.nn.Parameter(point.detach().clone().contiguous())

            fallback = torch.optim.Adam([inverse], lr=1e-2)

            for _ in range(self.config.c_transform_max_iter):
                fallback.zero_grad()

                inner = (point * inverse).sum(dim=-1, keepdim=True)
                potential = self.potential_network.forward(
                    condition=x,
                    tensor=inverse,
                )

                objective = (potential - inner).mean()

                if not torch.isfinite(objective):
                    break

                objective.backward()
                torch.nn.utils.clip_grad_norm_([inverse], max_norm=10.0)
                fallback.step()

        return inverse.detach()

    def estimate_psi(
        self,
        x: torch.Tensor,
        y_scaled: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.potential_type == "y":
            return self.potential_network(
                condition=x,
                tensor=y_scaled,
            )

        if u is None:
            raise ValueError("u must be provided when potential_type='u'.")

        return (
            (y_scaled * u).sum(dim=-1, keepdim=True) -
            self.potential_network(condition=x, tensor=u)
        )

    def estimate_phi(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y_scaled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.potential_type == "u":
            return self.potential_network(
                condition=x,
                tensor=u,
            )

        if y_scaled is None:
            raise ValueError("y_scaled must be provided when potential_type='y'.")

        return (
            (y_scaled * u).sum(dim=-1, keepdim=True) -
            self.potential_network(condition=x, tensor=y_scaled)
        )

    @torch.enable_grad()
    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        jitter: float = 1e-6,
        create_graph: bool | None = None,
    ) -> torch.Tensor:
        """
        Estimate log |det D_u T_x(u)| for the public pushforward map.

        This includes both the scaled-coordinate determinant from
        estimate_log_det_d2_phi and the final diagonal Jacobian contribution
        from unscale_y.
        """
        log_det = self.estimate_log_det_d2_phi(
            x=x,
            u=u,
            jitter=jitter,
            create_graph=create_graph,
        )
        return log_det + self._unscale_y_log_det()

    @torch.enable_grad()
    def estimate_log_det_d2_phi(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        jitter: float = 1e-6,
        create_graph: bool | None = None,
    ) -> torch.Tensor:
        """
        Estimate log det D_2 phi_x(u).

        This is only implemented when potential_type='u', where phi_x is
        represented directly by the potential network. The determinant is
        computed in the model's scaled target coordinates.

        Args:
            x: (batch, x_dim)
            u: (batch, y_dim)
            jitter: non-negative diagonal jitter added before taking the
                determinant.

        Returns:
            log_det: (batch,)
        """
        if jitter < 0.0:
            raise ValueError(f"jitter must be non-negative, got {jitter}.")

        self.eval()

        x = self.to_device(x)
        u = self.to_device(u)
        if create_graph is None:
            create_graph = u.requires_grad

        self._validate_condition_and_point_shapes(x=x, point=u)

        if self.potential_type == "y":
            raise NotImplementedError(
                "log det estimate for the y potential is not implemented."
            )

        return self._potential_hessian_log_det(
            x=x,
            point=u,
            jitter=jitter,
            create_graph=create_graph,
        )

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

    @torch.enable_grad()
    def _potential_hessian_log_det(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
        jitter: float,
        create_graph: bool,
    ) -> torch.Tensor:
        point = self.to_device(point)
        if not point.requires_grad:
            point = point.detach().clone().requires_grad_(True)
        x = self.to_device(x)

        potential = self.potential_network(
            condition=x,
            tensor=point,
        ).sum()

        grad = torch.autograd.grad(
            potential,
            point,
            create_graph=True,
        )[0]

        hessian_rows = []
        for dim in range(self.y_dim):
            row = torch.autograd.grad(
                grad[:, dim].sum(),
                point,
                create_graph=create_graph,
                retain_graph=create_graph or dim < self.y_dim - 1,
            )[0]
            hessian_rows.append(row)

        hessian = torch.stack(hessian_rows, dim=1)
        hessian = 0.5 * (hessian + hessian.transpose(-1, -2))

        eye = torch.eye(
            self.y_dim,
            device=hessian.device,
            dtype=hessian.dtype,
        )
        hessian = hessian + jitter * eye.unsqueeze(0)

        sign, log_abs_det = torch.linalg.slogdet(hessian)
        log_det = torch.where(
            sign > 0,
            log_abs_det,
            torch.full_like(log_abs_det, torch.nan),
        )
        if create_graph:
            return log_det

        return log_det.detach()

    def _unscale_y_log_det(self) -> torch.Tensor:
        return 0.5 * torch.log(self.y_scaler.running_var + self.y_scaler.eps).sum()

    @torch.enable_grad()
    def pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pull y back to latent u.

        Args:
            x: (batch, x_dim)
            y: (batch, y_dim)

        Returns:
            u: (batch, y_dim)
        """
        self.eval()

        x = self.to_device(x)
        y_scaled = self.scale_y(self.to_device(y))

        if self.potential_type == "y":
            u = self.gradient_inverse(x=x, point=y_scaled)
        else:
            u = self.c_transform_inverse(x=x, point=y_scaled)

        return u.detach()

    @torch.enable_grad()
    def pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Push latent u to y.

        Args:
            x: (batch, x_dim)
            u: (batch, y_dim)

        Returns:
            y: (batch, y_dim)
        """
        self.eval()

        x = self.to_device(x)
        u = self.to_device(u)

        if self.potential_type == "u":
            y_scaled = self.gradient_inverse(x=x, point=u)
        else:
            y_scaled = self.c_transform_inverse(x=x, point=u)

        return self.unscale_y(y_scaled).detach()

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.pullback(x=x, y=y)

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
        data = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        config = NeuralOptimalTransportPredictorConfig.model_validate(data["config"])
        model = cls(config)
        model.load_state_dict(data["state_dict"])

        return model
