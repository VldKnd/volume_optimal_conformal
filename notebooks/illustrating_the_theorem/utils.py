"""Sampling utilities for the theorem illustration experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import ot
import torch
from scipy.special import gammainccinv, gammaincinv
from scipy.spatial import cKDTree


def _validate_dimension(dimension: int) -> None:
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an integer.")
    if dimension < 1:
        raise ValueError("dimension must be positive.")


def _validate_number_of_samples(number_of_samples: int) -> None:
    if not isinstance(number_of_samples, int) or isinstance(number_of_samples, bool):
        raise TypeError("number_of_samples must be an integer.")
    if number_of_samples < 1:
        raise ValueError("number_of_samples must be positive.")


@dataclass(frozen=True)
class DiscreteOTMatching:
    """A compact, memorized discrete optimal-transport solution.

    ``source_to_target[i]`` contains the index of the target point matched to
    ``source_points[i]``. The target cloud itself remains in its original
    order, while :attr:`matched_target_points` exposes it in source order.
    """

    source_points: torch.Tensor
    target_points: torch.Tensor
    source_to_target: torch.Tensor
    mean_squared_cost: float

    @property
    def number_of_points(self) -> int:
        return self.source_points.shape[0]

    @property
    def matched_target_points(self) -> torch.Tensor:
        """Return target points ordered by their matched source points."""
        return self.target_points[self.source_to_target]

    def target_points_for_source_ball(self, radius: float) -> torch.Tensor:
        """Return targets induced by source points inside a centered ball."""
        if not math.isfinite(radius) or not 0.0 <= radius <= 1.0:
            raise ValueError("radius must lie between zero and one.")
        source_mask = self.source_points.norm(dim=-1) <= radius
        return self.matched_target_points[source_mask]

    def dense_coupling(self) -> torch.Tensor:
        """Materialize the uniform-weight permutation coupling if required."""
        coupling = torch.zeros(
            self.number_of_points,
            self.number_of_points,
            device=self.source_points.device,
            dtype=self.source_points.dtype,
        )
        source_indices = torch.arange(
            self.number_of_points,
            device=self.source_points.device,
        )
        coupling[source_indices, self.source_to_target] = 1.0 / self.number_of_points
        return coupling


@dataclass(frozen=True)
class LazyExactOTResult:
    """Exact empirical OT returned by POT's coordinate-based lazy solver.

    The sparse coupling is stored as coordinate triples. For an equal-size,
    equal-weight problem, ``source_to_target`` is populated when that sparse
    coupling is a genuine one-to-one matching; otherwise it remains ``None``.
    No dense cost or coupling matrix is retained.
    """

    source_points: torch.Tensor
    target_points: torch.Tensor
    optimal_transport_cost: float
    sparse_source_indices: torch.Tensor
    sparse_target_indices: torch.Tensor
    sparse_masses: torch.Tensor
    source_marginals: torch.Tensor
    target_marginals: torch.Tensor
    source_to_target: torch.Tensor | None
    status: str | None

    @property
    def has_permutation(self) -> bool:
        return self.source_to_target is not None

    @property
    def matched_target_points(self) -> torch.Tensor | None:
        """Return targets in source order when the coupling is a permutation."""
        if self.source_to_target is None:
            return None
        return self.target_points[self.source_to_target]

    def as_discrete_matching(self) -> DiscreteOTMatching:
        """Convert a permutation result to the established matching type."""
        if self.source_to_target is None:
            raise RuntimeError("The lazy exact coupling is not a permutation.")
        return DiscreteOTMatching(
            source_points=self.source_points,
            target_points=self.target_points,
            source_to_target=self.source_to_target,
            mean_squared_cost=self.optimal_transport_cost,
        )


@dataclass(frozen=True)
class SinkhornOTResult:
    """Lazy entropic OT solution represented by its barycentric map."""

    source_points: torch.Tensor
    target_points: torch.Tensor
    transported_source_points: torch.Tensor
    transport_cost: float
    regularized_transport_cost: float
    regularization: float
    source_marginals: torch.Tensor
    target_marginals: torch.Tensor
    max_source_marginal_error: float
    max_target_marginal_error: float
    number_of_iterations: int | None
    final_error: float
    converged: bool
    compute_backend: str = "cpu"
    compute_device: str = "cpu"


OTSolverBackend: TypeAlias = Literal[
    "dense_exact",
    "lazy_exact",
    "sinkhorn",
]
SinkhornComputeBackend: TypeAlias = Literal["cpu", "cuda", "geomloss"]
EmpiricalOTResult: TypeAlias = (
    DiscreteOTMatching | LazyExactOTResult | SinkhornOTResult
)


@dataclass(frozen=True)
class MonteCarloOTRegionVolume:
    """Log-domain Monte Carlo volume estimate and reusable diagnostics."""

    log_volume: float
    log_inclusion_fraction: float
    log_bounding_box_volume: float
    log_volume_mc_standard_error: float
    number_inside: int
    number_of_samples: int
    source_radius: float
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor

    @property
    def volume(self) -> float:
        """Exponentiate the estimate only when ordinary volume is required."""
        return math.exp(self.log_volume)

    @property
    def inclusion_fraction(self) -> float:
        """Exponentiate the stored log inclusion fraction."""
        return math.exp(self.log_inclusion_fraction)

    @property
    def bounding_box_volume(self) -> float:
        """Exponentiate the stored log bounding-box volume."""
        return math.exp(self.log_bounding_box_volume)


@dataclass(frozen=True)
class LogVolumeRatioExperiment:
    """Scalar outputs from one seeded HDR-versus-OT comparison."""

    dimension: int
    k: float
    seed: int
    significance_level: float
    number_of_ot_points: int
    number_of_monte_carlo_samples: int
    hdr_log_volume: float
    ot_log_volume: float
    log_volume_ratio: float
    log_volume_mc_standard_error: float
    log_inclusion_fraction: float
    log_bounding_box_volume: float
    number_inside: int
    number_of_source_region_points: int
    mean_squared_transport_cost: float
    ot_backend: str = "dense_exact"
    regularization: float | None = None
    solver_converged: bool | None = None
    sinkhorn_compute_backend: str | None = None


def log_ball_volume(dimension: int, radius: float = 1.0) -> float:
    """Return the log Lebesgue volume of a Euclidean ball."""
    _validate_dimension(dimension)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive.")

    return (
        dimension * math.log(radius) + 0.5 * dimension * math.log(math.pi) -
        math.lgamma(0.5 * dimension + 1.0)
    )


def ball_volume(dimension: int, radius: float = 1.0) -> float:
    """Return ball volume by exponentiating :func:`log_ball_volume`."""
    return math.exp(log_ball_volume(dimension=dimension, radius=radius))


def sample_uniform_ball(
    number_of_samples: int,
    dimension: int,
    radius: float = 1.0,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample uniformly with respect to volume from a Euclidean ball.

    The radial distribution of a uniform point in a ``dimension``-dimensional
    ball is obtained by raising a Uniform(0, 1) draw to ``1 / dimension``.
    The supplied generator must be compatible with ``device``.
    """
    _validate_number_of_samples(number_of_samples)
    _validate_dimension(dimension)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive.")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")

    device = torch.device(device)
    directions = torch.randn(
        number_of_samples,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(dtype).eps
    )
    radial_fractions = torch.rand(
        number_of_samples,
        1,
        generator=generator,
        device=device,
        dtype=dtype,
    ).pow(1.0 / dimension)
    return radius * radial_fractions * directions


def gaussian_covariance_diagonal(
    dimension: int,
    k: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return ``diag(Sigma)`` for the experiment's anisotropic Gaussian.

    The covariance is

        diag(k ** (d - 1/d), k ** (-1/d), ..., k ** (-1/d)).
    """
    _validate_dimension(dimension)
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive.")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")

    leading_variance = k**(dimension - 1.0 / dimension)
    remaining_variance = k**(-1.0 / dimension)
    covariance_diagonal = torch.full(
        (dimension, ),
        remaining_variance,
        device=torch.device(device),
        dtype=dtype,
    )
    covariance_diagonal[0] = leading_variance
    return covariance_diagonal


def gaussian_hdr_log_volume(
    dimension: int,
    k: float,
    significance_level: float,
) -> float:
    """Return the exact log-volume of the Gaussian HDR at significance alpha.

    The region with probability ``1 - alpha`` is

        Sigma ** (1/2) B_d(0, F_Chi(d) ** (-1)(1 - alpha)).

    For the experiment covariance, ``det(Sigma) = k ** (d - 1)`` and hence
    ``abs(det(Sigma ** (1/2))) = k ** ((d - 1) / 2)``.
    """
    _validate_dimension(dimension)
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive.")
    if (not math.isfinite(significance_level) or not 0.0 < significance_level < 1.0):
        raise ValueError("significance_level must lie strictly between zero and one.")

    chi_squared_half = float(gammainccinv(0.5 * dimension, significance_level))
    log_chi_radius = 0.5 * (math.log(2.0) + math.log(chi_squared_half))
    log_unit_ball_volume = log_ball_volume(dimension=dimension, radius=1.0)
    log_covariance_square_root_determinant = (0.5 * (dimension - 1) * math.log(k))
    return (
        log_unit_ball_volume + dimension * log_chi_radius +
        log_covariance_square_root_determinant
    )


def gaussian_hdr_volume(
    dimension: int,
    k: float,
    significance_level: float,
) -> float:
    """Exponentiate :func:`gaussian_hdr_log_volume` when volume is required."""
    return math.exp(
        gaussian_hdr_log_volume(
            dimension=dimension,
            k=k,
            significance_level=significance_level,
        )
    )


def sample_gaussian(
    number_of_samples: int,
    dimension: int,
    k: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample the centered Gaussian used in the theorem experiment."""
    _validate_number_of_samples(number_of_samples)
    covariance_diagonal = gaussian_covariance_diagonal(
        dimension=dimension,
        k=k,
        device=device,
        dtype=dtype,
    )
    standard_normal = torch.randn(
        number_of_samples,
        dimension,
        generator=generator,
        device=torch.device(device),
        dtype=dtype,
    )
    return standard_normal * covariance_diagonal.sqrt()


def transport_uniform_ball_to_gaussian(
    points: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Push points in the unit ball to the experiment's Gaussian.

    If ``r = ||u||`` for a point sampled uniformly in the unit ball, then
    ``r ** d`` is uniform on ``[0, 1]``. Applying the inverse CDF of a
    Chi(d) random variable to this probability produces the radius of a
    standard Gaussian. The result is then multiplied by the square root of
    the experiment covariance matrix.

    The inverse CDF is evaluated with SciPy and is therefore intended as a
    sampling/set-construction utility rather than a differentiable map.
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor.")
    if points.ndim < 2:
        raise ValueError("points must have shape (..., dimension).")
    if not points.dtype.is_floating_point:
        raise TypeError("points must have a floating-point dtype.")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must contain only finite values.")

    dimension = points.shape[-1]
    _validate_dimension(dimension)
    covariance_diagonal = gaussian_covariance_diagonal(
        dimension=dimension,
        k=k,
        device=points.device,
        dtype=points.dtype,
    )

    source_radii = points.norm(dim=-1, keepdim=True)
    tolerance = 32.0 * torch.finfo(points.dtype).eps
    if bool((source_radii > 1.0 + tolerance).any()):
        raise ValueError("All points must lie in the unit ball.")

    source_radii = source_radii.clamp(max=1.0)
    radial_probabilities = source_radii.pow(dimension)
    radial_probabilities = radial_probabilities.clamp(
        max=1.0 - torch.finfo(points.dtype).eps
    )
    probabilities_numpy = (
        radial_probabilities.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    target_radii_numpy = np.sqrt(
        2.0 * gammaincinv(0.5 * dimension, probabilities_numpy)
    )
    target_radii = torch.as_tensor(
        target_radii_numpy,
        device=points.device,
        dtype=points.dtype,
    )

    directions = points / source_radii.clamp_min(torch.finfo(points.dtype).eps)
    standard_gaussian_points = directions * target_radii
    covariance_square_root = torch.diag(covariance_diagonal.sqrt())
    return standard_gaussian_points @ covariance_square_root.T


def sample_gaussian_hdr(
    number_of_samples: int,
    dimension: int,
    k: float,
    coverage_mass: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample the Gaussian HDR containing ``coverage_mass`` probability.

    The centered source ball of radius ``coverage_mass ** (1 / dimension)``
    contains the requested probability under the uniform source measure. Its
    image under :func:`transport_uniform_ball_to_gaussian` is the corresponding
    Gaussian highest-density region.
    """
    _validate_number_of_samples(number_of_samples)
    _validate_dimension(dimension)
    if not math.isfinite(coverage_mass) or not 0.0 < coverage_mass < 1.0:
        raise ValueError("coverage_mass must lie strictly between zero and one.")

    source_radius = math.exp(math.log(coverage_mass) / dimension)
    source_points = sample_uniform_ball(
        number_of_samples=number_of_samples,
        dimension=dimension,
        radius=source_radius,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return transport_uniform_ball_to_gaussian(source_points, k=k)


def solve_discrete_exact_ot(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    maximum_iterations: int = 1_000_000,
) -> DiscreteOTMatching:
    """Solve uniform-weight empirical OT and retain its one-to-one matching.

    The exact Earth Mover's Distance network-simplex solver from POT minimizes
    squared Euclidean cost. Equal-size empirical measures with uniform weights
    yield a permutation coupling, which is stored compactly as target indices.
    """
    if not isinstance(source_points,
                      torch.Tensor) or not isinstance(target_points, torch.Tensor):
        raise TypeError("source_points and target_points must be torch tensors.")
    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("Point clouds must have shape (number_of_points, dimension).")
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target point clouds must have the same shape.")
    if source_points.shape[0] < 1:
        raise ValueError("Point clouds must not be empty.")
    if not source_points.dtype.is_floating_point or not target_points.dtype.is_floating_point:
        raise TypeError("Point clouds must have floating-point dtypes.")
    if source_points.dtype != target_points.dtype:
        raise ValueError("Source and target point clouds must have the same dtype.")
    if source_points.device != target_points.device:
        raise ValueError("Source and target point clouds must be on the same device.")
    if not bool(torch.isfinite(source_points).all()
                ) or not bool(torch.isfinite(target_points).all()):
        raise ValueError("Point clouds must contain only finite values.")
    if not isinstance(maximum_iterations, int) or isinstance(maximum_iterations, bool):
        raise TypeError("maximum_iterations must be an integer.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")

    source_numpy = source_points.detach().to(device="cpu", dtype=torch.float64).numpy()
    target_numpy = target_points.detach().to(device="cpu", dtype=torch.float64).numpy()
    squared_cost_matrix = ot.dist(
        source_numpy,
        target_numpy,
        metric="sqeuclidean",
    )
    number_of_points = source_points.shape[0]
    uniform_weights = np.full(number_of_points, 1.0 / number_of_points)
    coupling = ot.emd(
        uniform_weights,
        uniform_weights,
        squared_cost_matrix,
        numItermax=maximum_iterations,
        check_marginals=True,
    )

    source_indices = np.arange(number_of_points)
    source_to_target_numpy = coupling.argmax(axis=1)
    matched_masses = coupling[source_indices, source_to_target_numpy]
    expected_mass = 1.0 / number_of_points
    is_permutation = np.unique(source_to_target_numpy).size == number_of_points
    has_uniform_matched_mass = np.allclose(
        matched_masses,
        expected_mass,
        rtol=1e-7,
        atol=1e-12,
    )
    if not is_permutation or not has_uniform_matched_mass:
        raise RuntimeError(
            "POT returned a non-permutation coupling; the discrete matching "
            "cannot be stored as one target index per source point."
        )

    mean_squared_cost = float(np.sum(coupling * squared_cost_matrix))
    source_to_target = torch.as_tensor(
        source_to_target_numpy,
        device=target_points.device,
        dtype=torch.long,
    )
    return DiscreteOTMatching(
        source_points=source_points.detach().clone(),
        target_points=target_points.detach().clone(),
        source_to_target=source_to_target,
        mean_squared_cost=mean_squared_cost,
    )


def _validate_ot_point_clouds(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
) -> None:
    """Apply the dense reference solver's input contract to new solvers."""
    if not isinstance(source_points,
                      torch.Tensor) or not isinstance(target_points, torch.Tensor):
        raise TypeError("source_points and target_points must be torch tensors.")
    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("Point clouds must have shape (number_of_points, dimension).")
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target point clouds must have the same shape.")
    if source_points.shape[0] < 1:
        raise ValueError("Point clouds must not be empty.")
    if (
        not source_points.dtype.is_floating_point
        or not target_points.dtype.is_floating_point
    ):
        raise TypeError("Point clouds must have floating-point dtypes.")
    if source_points.dtype != target_points.dtype:
        raise ValueError("Source and target point clouds must have the same dtype.")
    if source_points.device != target_points.device:
        raise ValueError("Source and target point clouds must be on the same device.")
    if not bool(torch.isfinite(source_points).all()
                ) or not bool(torch.isfinite(target_points).all()):
        raise ValueError("Point clouds must contain only finite values.")


def _cpu_float64_numpy(points: torch.Tensor) -> np.ndarray:
    """Expose a contiguous CPU float64 array, copying only when necessary."""
    points_cpu = points.detach().to(device="cpu", dtype=torch.float64)
    return np.ascontiguousarray(points_cpu.numpy())


def solve_discrete_exact_ot_lazy(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    maximum_iterations: int = 1_000_000,
) -> LazyExactOTResult:
    """Solve exact empirical OT without materializing the dense cost matrix.

    POT's coordinate-based lazy network simplex computes squared-Euclidean
    costs on demand in CPU float64. Its sparse coupling is retained directly.
    When the equal-weight coupling is a permutation, the result also exposes
    the same ``source_to_target`` representation as :class:`DiscreteOTMatching`.
    """
    _validate_ot_point_clouds(source_points, target_points)
    if not isinstance(maximum_iterations, int) or isinstance(maximum_iterations, bool):
        raise TypeError("maximum_iterations must be an integer.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")

    source_numpy = _cpu_float64_numpy(source_points)
    target_numpy = _cpu_float64_numpy(target_points)
    number_of_points = source_points.shape[0]
    uniform_weights = np.full(
        number_of_points,
        1.0 / number_of_points,
        dtype=np.float64,
    )
    pot_result = ot.solve_sample(
        source_numpy,
        target_numpy,
        a=uniform_weights,
        b=uniform_weights,
        metric="sqeuclidean",
        lazy=True,
        max_iter=maximum_iterations,
    )
    sparse_plan = pot_result.sparse_plan
    if sparse_plan is None:
        raise RuntimeError("POT's lazy exact solver did not return a sparse plan.")

    sparse_plan = sparse_plan.tocoo()
    sparse_sources = np.asarray(sparse_plan.row, dtype=np.int64)
    sparse_targets = np.asarray(sparse_plan.col, dtype=np.int64)
    sparse_masses = np.asarray(sparse_plan.data, dtype=np.float64)
    positive = sparse_masses > 0.0
    sparse_sources = sparse_sources[positive]
    sparse_targets = sparse_targets[positive]
    sparse_masses = sparse_masses[positive]

    source_marginals = np.bincount(
        sparse_sources,
        weights=sparse_masses,
        minlength=number_of_points,
    )
    target_marginals = np.bincount(
        sparse_targets,
        weights=sparse_masses,
        minlength=number_of_points,
    )
    if not np.allclose(
        source_marginals,
        uniform_weights,
        rtol=1e-7,
        atol=1e-12,
    ) or not np.allclose(
        target_marginals,
        uniform_weights,
        rtol=1e-7,
        atol=1e-12,
    ):
        raise RuntimeError("POT returned a lazy exact plan with invalid marginals.")

    expected_mass = 1.0 / number_of_points
    is_permutation = (
        sparse_masses.size == number_of_points
        and np.unique(sparse_sources).size == number_of_points
        and np.unique(sparse_targets).size == number_of_points and np.allclose(
            sparse_masses,
            expected_mass,
            rtol=1e-7,
            atol=1e-12,
        )
    )
    source_to_target = None
    if is_permutation:
        source_to_target_numpy = np.empty(number_of_points, dtype=np.int64)
        source_to_target_numpy[sparse_sources] = sparse_targets
        source_to_target = torch.as_tensor(
            source_to_target_numpy,
            device=target_points.device,
            dtype=torch.long,
        )
        if source_to_target.unique().numel() != number_of_points:
            raise RuntimeError("The lazy exact source-to-target map is not one-to-one.")

    optimal_transport_cost = float(pot_result.value_linear)
    if not math.isfinite(optimal_transport_cost):
        raise RuntimeError("POT returned a non-finite lazy exact transport cost.")
    return LazyExactOTResult(
        source_points=source_points.detach(),
        target_points=target_points.detach(),
        optimal_transport_cost=optimal_transport_cost,
        sparse_source_indices=torch.as_tensor(sparse_sources, dtype=torch.long),
        sparse_target_indices=torch.as_tensor(sparse_targets, dtype=torch.long),
        sparse_masses=torch.as_tensor(sparse_masses, dtype=torch.float64),
        source_marginals=torch.as_tensor(source_marginals, dtype=torch.float64),
        target_marginals=torch.as_tensor(target_marginals, dtype=torch.float64),
        source_to_target=source_to_target,
        status=None if pot_result.status is None else str(pot_result.status),
    )


def _squared_euclidean_cost_block(
    source_block: torch.Tensor,
    target_block: torch.Tensor,
) -> torch.Tensor:
    """Compute one squared-Euclidean cost block without a 3-D difference."""
    source_norms = source_block.square().sum(dim=1, keepdim=True)
    target_norms = target_block.square().sum(dim=1).unsqueeze(0)
    return (
        source_norms + target_norms - 2.0 * source_block @ target_block.T
    ).clamp_min(0.0)


def _solve_discrete_sinkhorn_ot_geomloss(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    epsilon: float,
    *,
    maximum_iterations: int,
    tolerance: float,
    batch_size: int,
    compute_device: torch.device,
    compute_backend: str,
) -> SinkhornOTResult:
    """Run POT's online GeomLoss/PyKeOps Sinkhorn integration."""
    source_compute = source_points.detach().to(
        device=compute_device,
        dtype=torch.float64,
    )
    target_compute = target_points.detach().to(
        device=compute_device,
        dtype=torch.float64,
    )
    number_of_points = source_compute.shape[0]
    uniform_weights = torch.full(
        (number_of_points, ),
        1.0 / number_of_points,
        device=compute_device,
        dtype=torch.float64,
    )
    try:
        geomloss_result = ot.solve_sample(
            source_compute,
            target_compute,
            metric="sqeuclidean",
            reg=epsilon,
            method="geomloss",
            lazy=True,
            max_iter=maximum_iterations,
            # GeomLoss 0.3.1 accepts ``tol`` in its signature but explicitly
            # rejects it because rigorous stopping criteria are not implemented.
            tol=None,
            a=uniform_weights,
            b=uniform_weights,
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "POT's GeomLoss Sinkhorn backend requires both geomloss and "
            "pykeops to be installed and importable."
        ) from error
    if geomloss_result.lazy_plan is None:
        raise RuntimeError(
            "GeomLoss did not expose a lazy KeOps plan. Refusing to "
            "materialize a dense Sinkhorn coupling."
        )

    source_marginals_compute = geomloss_result.marginal_a
    target_marginals_compute = geomloss_result.marginal_b
    barycentric_numerator = geomloss_result.plan_operator @ target_compute
    transported_source_compute = (
        barycentric_numerator / source_marginals_compute.unsqueeze(1)
    )

    source_potential = geomloss_result.potential_a
    target_potential = geomloss_result.potential_b
    log_uniform_weight = -math.log(number_of_points)
    transport_cost_compute = torch.zeros(
        (),
        device=compute_device,
        dtype=torch.float64,
    )
    entropy_compute = torch.zeros_like(transport_cost_compute)
    for start in range(0, number_of_points, batch_size):
        stop = min(start + batch_size, number_of_points)
        cost_block = _squared_euclidean_cost_block(
            source_compute[start:stop],
            target_compute,
        )
        log_plan_block = (
            2.0 * log_uniform_weight
            + (
                source_potential[start:stop].unsqueeze(1)
                + target_potential.unsqueeze(0)
                - cost_block
            ) / epsilon
        )
        plan_block = torch.exp(log_plan_block)
        transport_cost_compute += (plan_block * cost_block).sum()
        entropy_compute += (plan_block * log_plan_block).sum()

    regularized_cost_compute = (
        transport_cost_compute + epsilon * entropy_compute
    )
    if not bool(torch.isfinite(transported_source_compute).all()) or not bool(
        torch.isfinite(regularized_cost_compute)
    ):
        raise RuntimeError("GeomLoss returned non-finite Sinkhorn outputs.")

    uniform_weight = 1.0 / number_of_points
    max_source_marginal_error = float(
        (source_marginals_compute - uniform_weight).abs().max()
    )
    max_target_marginal_error = float(
        (target_marginals_compute - uniform_weight).abs().max()
    )
    final_error = max(
        max_source_marginal_error,
        max_target_marginal_error,
    )
    converged = math.isfinite(final_error) and final_error <= tolerance
    return SinkhornOTResult(
        source_points=source_points.detach(),
        target_points=target_points.detach(),
        transported_source_points=transported_source_compute.to(
            device=source_points.device,
            dtype=source_points.dtype,
        ),
        transport_cost=float(transport_cost_compute),
        regularized_transport_cost=float(regularized_cost_compute),
        regularization=epsilon,
        source_marginals=source_marginals_compute.to(
            device=source_points.device,
            dtype=source_points.dtype,
        ),
        target_marginals=target_marginals_compute.to(
            device=target_points.device,
            dtype=target_points.dtype,
        ),
        max_source_marginal_error=max_source_marginal_error,
        max_target_marginal_error=max_target_marginal_error,
        # GeomLoss 0.3.1 does not expose the number of completed iterations.
        number_of_iterations=None,
        final_error=final_error,
        converged=converged,
        compute_backend=compute_backend,
        compute_device=str(compute_device),
    )


def solve_discrete_sinkhorn_ot(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    epsilon: float,
    *,
    compute_backend: SinkhornComputeBackend = "cpu",
    maximum_iterations: int = 10_000,
    tolerance: float = 1e-9,
    batch_size: int = 256,
    verbose: bool = False,
) -> SinkhornOTResult:
    """Solve scalable entropic empirical OT and return its barycentric map.

    All backends avoid a persistent dense ``N x N`` matrix and preserve
    float64 computation. ``cpu`` uses POT's lazy empirical Sinkhorn solver;
    ``cuda`` uses POT's GeomLoss/PyKeOps integration on a required CUDA device;
    ``geomloss`` uses the same online POT integration on CUDA when available
    and CPU otherwise. The returned map is

    ``T_epsilon(x_i) = sum_j gamma_ij y_j / sum_j gamma_ij``.

    Unlike exact OT, the entropic coupling is generally not a permutation, so
    this function returns :class:`SinkhornOTResult` rather than forcing the
    solution into :class:`DiscreteOTMatching`. GeomLoss 0.3.1 does not expose
    rigorous tolerance stopping or its completed iteration count; for that
    backend, ``tolerance`` is applied to the returned marginal diagnostics and
    ``number_of_iterations`` is ``None``.
    """
    _validate_ot_point_clouds(source_points, target_points)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    if not isinstance(maximum_iterations, int) or isinstance(maximum_iterations, bool):
        raise TypeError("maximum_iterations must be an integer.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if compute_backend not in ("cpu", "cuda", "geomloss"):
        raise ValueError(
            "compute_backend must be 'cpu', 'cuda', or 'geomloss'."
        )
    if compute_backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The CUDA Sinkhorn backend was requested, but CUDA is not "
                "available to PyTorch."
            )
        return _solve_discrete_sinkhorn_ot_geomloss(
            source_points,
            target_points,
            epsilon,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            batch_size=batch_size,
            compute_device=torch.device("cuda"),
            compute_backend="cuda",
        )
    if compute_backend == "geomloss":
        geomloss_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return _solve_discrete_sinkhorn_ot_geomloss(
            source_points,
            target_points,
            epsilon,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            batch_size=batch_size,
            compute_device=geomloss_device,
            compute_backend="geomloss",
        )

    source_numpy = _cpu_float64_numpy(source_points)
    target_numpy = _cpu_float64_numpy(target_points)
    number_of_points = source_points.shape[0]
    uniform_weights = np.full(
        number_of_points,
        1.0 / number_of_points,
        dtype=np.float64,
    )
    pot_result = ot.solve_sample(
        source_numpy,
        target_numpy,
        a=uniform_weights,
        b=uniform_weights,
        metric="sqeuclidean",
        reg=epsilon,
        reg_type="entropy",
        lazy=True,
        batch_size=batch_size,
        max_iter=maximum_iterations,
        tol=tolerance,
        verbose=verbose,
        grad="envelope",
    )
    lazy_plan = pot_result.lazy_plan
    if lazy_plan is None:
        raise RuntimeError("POT's lazy Sinkhorn solver did not return a lazy plan.")

    transported_source_numpy = np.empty_like(source_numpy)
    source_marginals = np.empty(number_of_points, dtype=np.float64)
    target_marginals = np.zeros(number_of_points, dtype=np.float64)
    entropy = 0.0
    for start in range(0, number_of_points, batch_size):
        stop = min(start + batch_size, number_of_points)
        plan_block = np.asarray(lazy_plan[start:stop, :], dtype=np.float64)
        if not np.isfinite(plan_block).all():
            raise RuntimeError("POT returned non-finite lazy Sinkhorn masses.")
        row_masses = plan_block.sum(axis=1)
        if np.any(row_masses <= 0.0):
            raise RuntimeError("POT returned a zero-mass Sinkhorn source row.")
        source_marginals[start:stop] = row_masses
        target_marginals += plan_block.sum(axis=0)
        transported_source_numpy[start:stop] = (plan_block
                                                @ target_numpy) / row_masses[:, None]
        positive_masses = plan_block[plan_block > 0.0]
        entropy += float(np.sum(positive_masses * np.log(positive_masses)))

    transport_cost = float(pot_result.value_linear)
    regularized_transport_cost = transport_cost + epsilon * entropy
    if not math.isfinite(transport_cost
                         ) or not math.isfinite(regularized_transport_cost):
        raise RuntimeError("POT returned a non-finite Sinkhorn transport cost.")

    max_source_marginal_error = float(
        np.max(np.abs(source_marginals - uniform_weights))
    )
    max_target_marginal_error = float(
        np.max(np.abs(target_marginals - uniform_weights))
    )
    solver_log = pot_result.log or {}
    errors = solver_log.get("err", [])
    final_error = float(errors[-1]) if errors else max(
        max_source_marginal_error,
        max_target_marginal_error,
    )
    number_of_iterations = int(solver_log.get("niter", 0))
    converged = (
        math.isfinite(final_error) and final_error <= tolerance
        and max_source_marginal_error <= tolerance
        and max_target_marginal_error <= tolerance
    )

    transported_source_points = torch.as_tensor(
        transported_source_numpy,
        device=source_points.device,
        dtype=source_points.dtype,
    )
    return SinkhornOTResult(
        source_points=source_points.detach(),
        target_points=target_points.detach(),
        transported_source_points=transported_source_points,
        transport_cost=transport_cost,
        regularized_transport_cost=regularized_transport_cost,
        regularization=epsilon,
        source_marginals=torch.as_tensor(
            source_marginals,
            device=source_points.device,
            dtype=source_points.dtype,
        ),
        target_marginals=torch.as_tensor(
            target_marginals,
            device=target_points.device,
            dtype=target_points.dtype,
        ),
        max_source_marginal_error=max_source_marginal_error,
        max_target_marginal_error=max_target_marginal_error,
        number_of_iterations=number_of_iterations,
        final_error=final_error,
        converged=converged,
        compute_backend="cpu",
        compute_device="cpu",
    )


def solve_empirical_ot(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    backend: OTSolverBackend = "dense_exact",
    epsilon: float = 0.1,
    sinkhorn_compute_backend: SinkhornComputeBackend = "cpu",
    maximum_iterations: int = 1_000_000,
    tolerance: float = 1e-9,
    batch_size: int = 256,
    verbose: bool = False,
) -> EmpiricalOTResult:
    """Solve empirical OT using one of the supported solver backends.

    ``dense_exact`` retains the original dense POT reference solver,
    ``lazy_exact`` uses POT's coordinate-based exact solver without a dense
    cost matrix, and ``sinkhorn`` computes a lazy entropic coupling and its
    barycentric transport map. ``sinkhorn_compute_backend`` then selects POT's
    default lazy CPU solver or its online GeomLoss/PyKeOps integration with a
    forced or automatically selected device.
    Sinkhorn-specific arguments are ignored by the exact backends, apart from
    ``maximum_iterations`` which all solvers use.

    The result type reflects the selected algorithm: Sinkhorn is deliberately
    kept separate from :class:`DiscreteOTMatching` because its coupling is not
    generally a permutation.
    """
    if backend == "dense_exact":
        return solve_discrete_exact_ot(
            source_points,
            target_points,
            maximum_iterations=maximum_iterations,
        )
    if backend == "lazy_exact":
        return solve_discrete_exact_ot_lazy(
            source_points,
            target_points,
            maximum_iterations=maximum_iterations,
        )
    if backend == "sinkhorn":
        return solve_discrete_sinkhorn_ot(
            source_points,
            target_points,
            epsilon=epsilon,
            compute_backend=sinkhorn_compute_backend,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            batch_size=batch_size,
            verbose=verbose,
        )
    supported_backends = ", ".join(
        ("dense_exact", "lazy_exact", "sinkhorn")
    )
    raise ValueError(
        f"Unknown OT backend {backend!r}. Choose one of: {supported_backends}."
    )


def empirical_ot_transport_pairs(
    result: EmpiricalOTResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source points and their transported points in matching order.

    Exact results return their one-to-one matched targets. Sinkhorn results
    return the barycentric image associated with each source point.
    """
    if isinstance(result, DiscreteOTMatching):
        return result.source_points, result.matched_target_points
    if isinstance(result, LazyExactOTResult):
        matched_target_points = result.matched_target_points
        if matched_target_points is None:
            raise RuntimeError(
                "The lazy exact coupling does not expose a one-to-one matching."
            )
        return result.source_points, matched_target_points
    if isinstance(result, SinkhornOTResult):
        return result.source_points, result.transported_source_points
    raise TypeError(
        "result must be a DiscreteOTMatching, LazyExactOTResult, or "
        "SinkhornOTResult."
    )


def empirical_ot_transport_cost(result: EmpiricalOTResult) -> float:
    """Return the unregularized squared-Euclidean cost of an OT result."""
    if isinstance(result, DiscreteOTMatching):
        return result.mean_squared_cost
    if isinstance(result, LazyExactOTResult):
        return result.optimal_transport_cost
    if isinstance(result, SinkhornOTResult):
        return result.transport_cost
    raise TypeError(
        "result must be a DiscreteOTMatching, LazyExactOTResult, or "
        "SinkhornOTResult."
    )


def sample_empirical_ot(
    number_of_points: int,
    dimension: int,
    k: float,
    *,
    backend: OTSolverBackend = "dense_exact",
    epsilon: float = 0.1,
    sinkhorn_compute_backend: SinkhornComputeBackend = "cpu",
    source_generator: torch.Generator | None = None,
    target_generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    maximum_iterations: int = 1_000_000,
    tolerance: float = 1e-9,
    batch_size: int = 256,
    verbose: bool = False,
) -> EmpiricalOTResult:
    """Sample the two empirical measures and solve with ``backend``."""
    source_points = sample_uniform_ball(
        number_of_samples=number_of_points,
        dimension=dimension,
        radius=1.0,
        generator=source_generator,
        device=device,
        dtype=dtype,
    )
    target_points = sample_gaussian(
        number_of_samples=number_of_points,
        dimension=dimension,
        k=k,
        generator=target_generator,
        device=device,
        dtype=dtype,
    )
    return solve_empirical_ot(
        source_points,
        target_points,
        backend=backend,
        epsilon=epsilon,
        sinkhorn_compute_backend=sinkhorn_compute_backend,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        batch_size=batch_size,
        verbose=verbose,
    )


def sample_discrete_exact_ot_matching(
    number_of_points: int,
    dimension: int,
    k: float,
    *,
    source_generator: torch.Generator | None = None,
    target_generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    maximum_iterations: int = 1_000_000,
) -> DiscreteOTMatching:
    """Sample both empirical measures and solve their exact discrete OT."""
    source_points = sample_uniform_ball(
        number_of_samples=number_of_points,
        dimension=dimension,
        radius=1.0,
        generator=source_generator,
        device=device,
        dtype=dtype,
    )
    target_points = sample_gaussian(
        number_of_samples=number_of_points,
        dimension=dimension,
        k=k,
        generator=target_generator,
        device=device,
        dtype=dtype,
    )
    return solve_discrete_exact_ot(
        source_points=source_points,
        target_points=target_points,
        maximum_iterations=maximum_iterations,
    )


def estimate_ot_induced_region_volume(
    matching: EmpiricalOTResult,
    significance_level: float,
    number_of_monte_carlo_samples: int,
    *,
    generator: torch.Generator | None = None,
    batch_size: int = 10_000,
) -> MonteCarloOTRegionVolume:
    """Estimate the volume of an empirical-OT-induced region.

    Transported points associated with the source ball of mass ``1 - alpha``
    define the coordinate-wise bounding box. Exact backends use the matched
    target atoms; Sinkhorn uses the barycentric image of every source atom.
    Uniform samples from the box are pulled back through their nearest
    transported atom. The box log-volume plus the log inclusion fraction gives
    the induced-region log-volume.

    Nearest-point lookup extends the inverse empirical transport to the target
    space as a Voronoi-constant map.
    """
    source_points, transported_source_points = empirical_ot_transport_pairs(
        matching
    )
    if (not math.isfinite(significance_level) or not 0.0 < significance_level < 1.0):
        raise ValueError("significance_level must lie strictly between zero and one.")
    _validate_number_of_samples(number_of_monte_carlo_samples)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    dimension = source_points.shape[-1]
    log_coverage_mass = math.log1p(-significance_level)
    source_radius = math.exp(log_coverage_mass / dimension)
    source_inside = source_points.norm(dim=-1) <= source_radius
    if not bool(source_inside.any()):
        raise RuntimeError(
            "No discrete source points lie in the requested reference ball."
        )

    region_target_points = transported_source_points[source_inside]
    lower_bounds = region_target_points.amin(dim=0)
    upper_bounds = region_target_points.amax(dim=0)
    side_lengths = upper_bounds - lower_bounds
    if bool((side_lengths <= 0).any()):
        raise RuntimeError(
            "The induced target points do not define a positive-volume "
            "coordinate-wise bounding box."
        )

    log_bounding_box_volume = float(
        side_lengths.detach().to(device="cpu", dtype=torch.float64).log().sum()
    )
    target_tree = cKDTree(
        transported_source_points.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).numpy()
    )
    source_inside_numpy = source_inside.detach().to(device="cpu").numpy()

    number_inside = 0
    number_remaining = number_of_monte_carlo_samples
    while number_remaining:
        current_batch_size = min(batch_size, number_remaining)
        unit_samples = torch.rand(
            current_batch_size,
            dimension,
            generator=generator,
            device=transported_source_points.device,
            dtype=transported_source_points.dtype,
        )
        cube_samples = lower_bounds + unit_samples * side_lengths
        cube_samples_numpy = (
            cube_samples.detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        _, nearest_target_indices = target_tree.query(cube_samples_numpy, k=1)
        number_inside += int(
            np.count_nonzero(source_inside_numpy[nearest_target_indices])
        )
        number_remaining -= current_batch_size

    if number_inside == 0:
        log_inclusion_fraction = -math.inf
        log_volume = -math.inf
        log_volume_mc_standard_error = math.inf
    else:
        log_inclusion_fraction = (
            math.log(number_inside) - math.log(number_of_monte_carlo_samples)
        )
        log_volume = log_bounding_box_volume + log_inclusion_fraction
        log_volume_mc_standard_error = math.sqrt(
            (number_of_monte_carlo_samples - number_inside) /
            (number_of_monte_carlo_samples * number_inside)
        )
    return MonteCarloOTRegionVolume(
        log_volume=log_volume,
        log_inclusion_fraction=log_inclusion_fraction,
        log_bounding_box_volume=log_bounding_box_volume,
        log_volume_mc_standard_error=log_volume_mc_standard_error,
        number_inside=number_inside,
        number_of_samples=number_of_monte_carlo_samples,
        source_radius=source_radius,
        lower_bounds=lower_bounds.detach().clone(),
        upper_bounds=upper_bounds.detach().clone(),
    )


def run_log_volume_ratio_experiment(
    dimension: int,
    k: float,
    significance_level: float,
    number_of_ot_points: int,
    number_of_monte_carlo_samples: int,
    seed: int,
    *,
    ot_backend: OTSolverBackend = "dense_exact",
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_compute_backend: SinkhornComputeBackend = "cpu",
    ot_maximum_iterations: int = 1_000_000,
    ot_tolerance: float = 1e-9,
    ot_batch_size: int = 256,
    ot_verbose: bool = False,
    monte_carlo_batch_size: int = 10_000,
) -> LogVolumeRatioExperiment:
    """Run one reproducible log-volume-ratio experiment with an OT backend."""
    _validate_dimension(dimension)
    _validate_number_of_samples(number_of_ot_points)
    _validate_number_of_samples(number_of_monte_carlo_samples)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    source_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(seed + 10_000)
    monte_carlo_generator = torch.Generator().manual_seed(seed + 20_000)
    ot_result = sample_empirical_ot(
        number_of_points=number_of_ot_points,
        dimension=dimension,
        k=k,
        backend=ot_backend,
        epsilon=sinkhorn_epsilon,
        sinkhorn_compute_backend=sinkhorn_compute_backend,
        source_generator=source_generator,
        target_generator=target_generator,
        maximum_iterations=ot_maximum_iterations,
        tolerance=ot_tolerance,
        batch_size=ot_batch_size,
        verbose=ot_verbose,
    )
    ot_volume = estimate_ot_induced_region_volume(
        matching=ot_result,
        significance_level=significance_level,
        number_of_monte_carlo_samples=number_of_monte_carlo_samples,
        generator=monte_carlo_generator,
        batch_size=monte_carlo_batch_size,
    )
    hdr_log_volume = gaussian_hdr_log_volume(
        dimension=dimension,
        k=k,
        significance_level=significance_level,
    )
    source_points, _ = empirical_ot_transport_pairs(ot_result)
    number_of_source_region_points = int(
        (source_points.norm(dim=-1) <= ot_volume.source_radius).sum()
    )
    solver_converged = None
    if isinstance(ot_result, LazyExactOTResult):
        solver_converged = ot_result.status == "Converged"
    elif isinstance(ot_result, SinkhornOTResult):
        solver_converged = ot_result.converged
    return LogVolumeRatioExperiment(
        dimension=dimension,
        k=float(k),
        seed=seed,
        significance_level=significance_level,
        number_of_ot_points=number_of_ot_points,
        number_of_monte_carlo_samples=number_of_monte_carlo_samples,
        hdr_log_volume=hdr_log_volume,
        ot_log_volume=ot_volume.log_volume,
        log_volume_ratio=hdr_log_volume - ot_volume.log_volume,
        log_volume_mc_standard_error=ot_volume.log_volume_mc_standard_error,
        log_inclusion_fraction=ot_volume.log_inclusion_fraction,
        log_bounding_box_volume=ot_volume.log_bounding_box_volume,
        number_inside=ot_volume.number_inside,
        number_of_source_region_points=number_of_source_region_points,
        mean_squared_transport_cost=empirical_ot_transport_cost(ot_result),
        ot_backend=ot_backend,
        regularization=(
            sinkhorn_epsilon if ot_backend == "sinkhorn" else None
        ),
        solver_converged=solver_converged,
        sinkhorn_compute_backend=(
            sinkhorn_compute_backend if ot_backend == "sinkhorn" else None
        ),
    )


__all__ = [
    "DiscreteOTMatching",
    "EmpiricalOTResult",
    "LazyExactOTResult",
    "LogVolumeRatioExperiment",
    "MonteCarloOTRegionVolume",
    "OTSolverBackend",
    "SinkhornOTResult",
    "SinkhornComputeBackend",
    "ball_volume",
    "empirical_ot_transport_cost",
    "empirical_ot_transport_pairs",
    "estimate_ot_induced_region_volume",
    "gaussian_covariance_diagonal",
    "gaussian_hdr_log_volume",
    "gaussian_hdr_volume",
    "log_ball_volume",
    "run_log_volume_ratio_experiment",
    "sample_discrete_exact_ot_matching",
    "sample_empirical_ot",
    "sample_gaussian",
    "sample_gaussian_hdr",
    "sample_uniform_ball",
    "solve_discrete_exact_ot",
    "solve_discrete_exact_ot_lazy",
    "solve_discrete_sinkhorn_ot",
    "solve_empirical_ot",
    "transport_uniform_ball_to_gaussian",
]
