from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Self

import torch
from torch.utils.data import DataLoader

from configs.conformal import ConformalPredictorConfig
from conformal.calibrators.base import BaseCalibrator
from conformal.calibrators.factory import make_calibrator
from conformal.calibrators.no_calibrator import NoCalibrator
from conformal.calibrators.norm_calibrator import NormCalibrator
from predictors.rearranged_transport.amortized_rearranged_transport import (
    AmortizedRearrangedTransport,
)


class ConformalPredictor:
    """Calibrated prediction-region wrapper around a transport predictor.

    The base predictor maps latent scores ``u`` to observations ``y``. The
    calibrator turns the pullback scores into a scalar threshold defining the
    latent conformal region.
    """

    def __init__(
        self,
        predictor: Any,
        config: ConformalPredictorConfig | Mapping[str, Any],
    ):
        if not isinstance(config, ConformalPredictorConfig):
            config = ConformalPredictorConfig.model_validate(config)

        self._validate_predictor(predictor)

        self.predictor = predictor
        self.base_predictor = predictor
        self.config = config

        self.x_dim = predictor.x_dim
        self.y_dim = predictor.y_dim
        self.device = torch.device(getattr(predictor, "device", "cpu"))
        self.dtype = getattr(predictor, "dtype", torch.float32)
        if not isinstance(self.dtype, torch.dtype):
            self.dtype = getattr(torch, str(self.dtype))

        self.calibrator = make_calibrator(config.calibrator)
        self._coverage_conditioned = self._is_coverage_conditioned(predictor)

        if isinstance(self.calibrator, NoCalibrator):
            self._initialize_analytic_calibrator()

    @property
    def coverage_mass(self) -> float:
        return self.config.coverage_mass

    @property
    def threshold(self) -> torch.Tensor | None:
        return getattr(self.calibrator, "threshold", None)

    @property
    def is_calibrated(self) -> bool:
        return self.threshold is not None

    def calibrate(
        self,
        dataloader: DataLoader,
    ) -> Self:
        """Fit the calibrator from batched ``(x, y)`` observations.

        Pullbacks are evaluated one dataloader batch at a time on the
        predictor device. Detached covariates and scores are accumulated on
        CPU, so calibration does not require the full dataset to fit in the
        accelerator memory.
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided for calibration.")

        if isinstance(self.calibrator, NoCalibrator):
            return self

        calibration_x: list[torch.Tensor] = []
        calibration_scores: list[torch.Tensor] = []
        cpu_dtype = self._calibration_cpu_dtype()

        for batch in dataloader:
            x_batch, y_batch = self._extract_xy_batch(batch)
            x_batch, y_batch = self._prepare_inputs(
                x=x_batch,
                point=y_batch,
                point_name="y",
            )

            if x_batch.shape[0] == 0:
                continue

            with torch.no_grad():
                scores = self._call_predictor(
                    method_name="pullback",
                    x=x_batch,
                    point=y_batch,
                    point_name="y",
                )

            self._validate_calibration_scores(
                scores=scores,
                batch_size=x_batch.shape[0],
            )
            calibration_x.append(x_batch.detach().to(
                device="cpu",
                dtype=cpu_dtype,
            ))
            calibration_scores.append(
                scores.detach().to(
                    device="cpu",
                    dtype=cpu_dtype,
                )
            )

        if not calibration_scores:
            raise ValueError(
                "Calibration dataloader must contain at least one observation."
            )

        self.calibrator.fit(
            x=torch.cat(calibration_x, dim=0),
            scores=torch.cat(calibration_scores, dim=0),
            coverage_mass=self.coverage_mass,
        )
        return self

    def fit(
        self,
        dataloader: DataLoader,
    ) -> Self:
        """Alias for :meth:`calibrate`."""
        return self.calibrate(dataloader=dataloader)

    def pushforward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Push latent points through the wrapped predictor."""
        x, u = self._prepare_inputs(x=x, point=u, point_name="u")
        return self._call_predictor(
            method_name="pushforward",
            x=x,
            point=u,
            point_name="u",
        )

    def pullback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Pull observations back to the wrapped predictor's latent space."""
        x, y = self._prepare_inputs(x=x, point=y, point_name="y")
        return self._call_predictor(
            method_name="pullback",
            x=x,
            point=y,
            point_name="y",
        )

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.pullback(x=x, y=y)

    def scalar_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.pullback(x=x, y=y)
        return self.calibrator.scalar_score(
            x=self._to_device(x),
            scores=scores,
        )

    def contains(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether each observation lies in its conformal region."""
        self._require_calibrated()
        with torch.no_grad():
            scores = self.pullback(x=x, y=y)
            return self.calibrator.contains(
                x=self._to_device(x),
                scores=scores,
            )

    def log_det(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Return the forward-map log absolute Jacobian determinant."""
        x, u = self._prepare_inputs(x=x, point=u, point_name="u")
        return self._call_predictor(
            method_name="log_det",
            x=x,
            point=u,
            point_name="u",
        )

    def estimate_log_volume(
        self,
        x: torch.Tensor,
        number_of_samples: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Estimate one log-volume per covariate by Monte Carlo integration.

        For a latent Euclidean ball ``B_r`` this estimates

        ``log Vol(B_r) + log E[exp(log |det D T_x(U)|)]``

        for ``U`` uniform in ``B_r``.

        ``batch_size`` bounds the number of flattened ``(x, u)`` pairs passed
        to the predictor in any one call.
        """
        radius = self._euclidean_ball_radius()
        x = self._prepare_x(x)
        if x.shape[0] == 0:
            return x.new_empty(0)

        number_of_samples = self._positive_integer(
            self.config.volume_mc_samples
            if number_of_samples is None else number_of_samples,
            name="number_of_samples",
        )
        batch_size = self._positive_integer(
            self.config.volume_batch_size if batch_size is None else batch_size,
            name="batch_size",
        )
        seed = self.config.volume_seed if seed is None else seed
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer.")

        radius = torch.as_tensor(
            radius,
            device=self.device,
            dtype=self.dtype,
        )
        if radius.ndim != 0:
            raise ValueError("The calibrated Euclidean-ball threshold must be scalar.")
        if not torch.isfinite(radius) or radius < 0.0:
            raise ValueError(
                "The calibrated Euclidean-ball threshold must be finite and "
                "non-negative."
            )

        if radius == 0.0:
            return x.new_full((x.shape[0], ), -torch.inf)

        sampling_device = (
            self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
        )
        generator = torch.Generator(device=sampling_device)
        generator.manual_seed(seed)

        log_integral = x.new_full((x.shape[0], ), -torch.inf)
        completed_samples = 0

        while completed_samples < number_of_samples:
            samples_per_x = min(
                batch_size,
                number_of_samples - completed_samples,
            )
            covariates_per_call = max(1, batch_size // samples_per_x)

            for x_start in range(0, x.shape[0], covariates_per_call):
                x_end = min(
                    x.shape[0],
                    x_start + covariates_per_call,
                )
                x_chunk = x[x_start:x_end]
                samples = self._sample_uniform_ball(
                    batch_size=x_chunk.shape[0],
                    number_of_samples=samples_per_x,
                    radius=radius.to(sampling_device),
                    device=sampling_device,
                    generator=generator,
                ).to(device=self.device, dtype=self.dtype)

                expanded_x = (
                    x_chunk[:, None, :].expand(
                        x_chunk.shape[0],
                        samples_per_x,
                        self.x_dim,
                    ).reshape(
                        x_chunk.shape[0] * samples_per_x,
                        self.x_dim,
                    )
                )
                flat_samples = samples.reshape(-1, self.y_dim)
                log_det = self.log_det(
                    x=expanded_x,
                    u=flat_samples,
                ).reshape(-1)

                expected_values = x_chunk.shape[0] * samples_per_x
                if log_det.numel() != expected_values:
                    raise ValueError(
                        "predictor.log_det must return one value per latent "
                        f"point; expected {expected_values}, got "
                        f"{log_det.numel()}."
                    )

                chunk_log_integral = torch.logsumexp(
                    log_det.reshape(
                        x_chunk.shape[0],
                        samples_per_x,
                    ).detach(),
                    dim=1,
                )
                log_integral[x_start:x_end] = torch.logaddexp(
                    log_integral[x_start:x_end],
                    chunk_log_integral,
                )

            completed_samples += samples_per_x

        log_ball_volume = (
            self.y_dim * torch.log(radius) + 0.5 * self.y_dim * math.log(math.pi) -
            math.lgamma(0.5 * self.y_dim + 1.0)
        )
        return (log_ball_volume + log_integral - math.log(number_of_samples))

    def estimate_volume(
        self,
        x: torch.Tensor,
        number_of_samples: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        return torch.exp(
            self.estimate_log_volume(
                x=x,
                number_of_samples=number_of_samples,
                batch_size=batch_size,
                seed=seed,
            )
        )

    def log_volume(
        self,
        x: torch.Tensor,
        number_of_samples: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        return self.estimate_log_volume(
            x=x,
            number_of_samples=number_of_samples,
            batch_size=batch_size,
            seed=seed,
        )

    def volume(
        self,
        x: torch.Tensor,
        number_of_samples: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        return self.estimate_volume(
            x=x,
            number_of_samples=number_of_samples,
            batch_size=batch_size,
            seed=seed,
        )

    def _initialize_analytic_calibrator(self) -> None:
        x = torch.empty(
            1,
            self.x_dim,
            device=self.device,
            dtype=self.dtype,
        )
        scores = torch.empty(
            1,
            self.y_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.calibrator.fit(
            x=x,
            scores=scores,
            coverage_mass=self.coverage_mass,
        )

    def _euclidean_ball_radius(self) -> torch.Tensor:
        self._require_calibrated()

        if isinstance(self.calibrator, NoCalibrator):
            return self.threshold

        if isinstance(self.calibrator, NormCalibrator):
            if not math.isclose(
                float(self.calibrator.config.p),
                2.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise NotImplementedError(
                    "Volume estimation currently supports only an L2 "
                    "NormCalibrator; an Lp region with p != 2 is not a "
                    "Euclidean ball."
                )
            return self.threshold

        raise NotImplementedError(
            "Volume estimation currently supports only NoCalibrator and an "
            "L2 NormCalibrator. The configured calibrator defines a "
            "non-Euclidean latent region."
        )

    def _sample_uniform_ball(
        self,
        batch_size: int,
        number_of_samples: int,
        radius: torch.Tensor,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        directions = torch.randn(
            batch_size,
            number_of_samples,
            self.y_dim,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )
        directions = directions / directions.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(self.dtype).eps)

        radial_fractions = torch.rand(
            batch_size,
            number_of_samples,
            1,
            device=device,
            dtype=self.dtype,
            generator=generator,
        ).pow(1.0 / self.y_dim)

        return radius * radial_fractions * directions

    def _call_predictor(
        self,
        method_name: str,
        x: torch.Tensor,
        point: torch.Tensor,
        point_name: str,
    ) -> torch.Tensor:
        self._set_predictor_eval()
        method = getattr(self.predictor, method_name)
        kwargs = {
            "x": x,
            point_name: point,
        }
        if self._coverage_conditioned:
            kwargs["coverage_mass"] = self.coverage_mass
        return method(**kwargs)

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        point: torch.Tensor,
        point_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare_x(x)
        point = self._to_device(point)

        if point.ndim != 2:
            raise ValueError(
                f"Expected {point_name} to be 2D, got shape "
                f"{tuple(point.shape)}."
            )
        if point.shape[-1] != self.y_dim:
            raise ValueError(
                f"Expected {point_name}.shape[-1] = {self.y_dim}, got "
                f"{point.shape[-1]}."
            )
        if point.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected x and {point_name} to have the same batch size, "
                f"got {x.shape[0]} and {point.shape[0]}."
            )

        return x, point

    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_device(x)
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got shape {tuple(x.shape)}.")
        if x.shape[-1] != self.x_dim:
            raise ValueError(f"Expected x.shape[-1] = {self.x_dim}, got {x.shape[-1]}.")
        return x

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(tensor).__name__}.")
        return tensor.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _extract_xy_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, Mapping):
            if "x" in batch and "y" in batch:
                return batch["x"], batch["y"]

            raise ValueError(
                "Calibration mapping batches must contain 'x' and 'y' keys."
            )

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise ValueError(
            "Expected calibration batches as (x_batch, y_batch) pairs or "
            "mappings containing 'x' and 'y'."
        )

    def _calibration_cpu_dtype(self) -> torch.dtype:
        if self.dtype in {torch.float32, torch.float64}:
            return self.dtype

        return torch.float32

    def _validate_calibration_scores(
        self,
        scores: Any,
        batch_size: int,
    ) -> None:
        if not isinstance(scores, torch.Tensor):
            raise TypeError(
                "predictor.pullback must return a torch.Tensor, got "
                f"{type(scores).__name__}."
            )

        expected_shape = (batch_size, self.y_dim)
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                "predictor.pullback must return scores with shape "
                f"{expected_shape}, got {tuple(scores.shape)}."
            )

    def _set_predictor_eval(self) -> None:
        eval_method = getattr(self.predictor, "eval", None)
        if callable(eval_method):
            eval_method()

    def _require_calibrated(self) -> None:
        if not self.is_calibrated:
            raise RuntimeError(
                "ConformalPredictor must be calibrated before this operation."
            )

    @staticmethod
    def _positive_integer(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    @staticmethod
    def _is_coverage_conditioned(predictor: Any) -> bool:
        if isinstance(predictor, AmortizedRearrangedTransport):
            return True

        if bool(getattr(predictor, "coverage_conditioned", False)):
            return True

        config = getattr(predictor, "config", None)
        return (getattr(config, "type", None) == "amortized_rearranged_transport")

    @staticmethod
    def _validate_predictor(predictor: Any) -> None:
        for dimension_name in ("x_dim", "y_dim"):
            dimension = getattr(predictor, dimension_name, None)
            minimum = 0 if dimension_name == "x_dim" else 1
            if (
                isinstance(dimension, bool) or not isinstance(dimension, int)
                or dimension < minimum
            ):
                raise TypeError(
                    f"predictor must expose integer {dimension_name} >= "
                    f"{minimum}."
                )

        for method_name in ("pushforward", "pullback", "log_det"):
            if not callable(getattr(predictor, method_name, None)):
                raise TypeError(f"predictor must implement a callable {method_name}().")
