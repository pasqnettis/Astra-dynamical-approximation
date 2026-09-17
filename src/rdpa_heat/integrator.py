"""Implicit heat steps using the modified iteration in equation (4.4).

Only the *parameter Jacobians* are frozen at the beginning of a physical
step. Nonlinear bulk and boundary residuals are evaluated at every iterate.
This module does not implement the artificial-time RK4 initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from time import perf_counter
from typing import Callable, NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp

from . import operators
from .linalg import LeastSquaresFactor, factorize, relative_stationarity, solve_factored
from .quadrature import QuadratureRule

if TYPE_CHECKING:
    from .config import RunConfig


class StepDiagnostics(NamedTuple):
    """Per-iterate diagnostics; components sum to the squared defect.

    The four columns contain the original, unscaled objective terms: bulk,
    boundary, accumulated displacement regularization, and increment
    regularization. Thus each squared augmented residual is divided by h².
    """

    defects: jax.Array
    components: jax.Array
    increment_norms: jax.Array
    stationarity: jax.Array
    defect_ratios: jax.Array


class StepSystem(NamedTuple):
    anchor: jax.Array
    h: jax.Array
    epsilon: jax.Array
    alpha: jax.Array
    quadrature: QuadratureRule
    previous_values: jax.Array
    bulk_matrix: jax.Array
    boundary_matrix: jax.Array
    factor: LeastSquaresFactor
    method_code: jax.Array


class StepResult(NamedTuple):
    parameters: jax.Array
    diagnostics: StepDiagnostics


@dataclass(frozen=True)
class SimulationResult:
    times: jax.Array
    parameters: jax.Array
    diagnostics: StepDiagnostics
    elapsed_seconds: float


class SimulationFailure(RuntimeError):
    """An aborted trajectory with all successfully completed steps retained."""

    def __init__(
        self, message: str, partial_result: SimulationResult, failed_step: int
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result
        self.failed_step = failed_step


def modified_matrix(
    bulk_matrix: jax.Array, boundary_matrix: jax.Array, epsilon: float
) -> jax.Array:
    """Build the two distinct regularization blocks printed in (4.4)."""
    identity = jnp.eye(bulk_matrix.shape[1], dtype=bulk_matrix.dtype)
    return jnp.concatenate(
        (
            bulk_matrix,
            boundary_matrix,
            epsilon / jnp.sqrt(2.0) * identity,
            epsilon * identity,
        ),
        axis=0,
    )


def modified_rhs(
    bulk_residual: jax.Array,
    boundary_residual: jax.Array,
    displacement: jax.Array,
    epsilon: float,
) -> jax.Array:
    """Return b for the equation-(4.4) objective ||M increment + b||².

    The least-squares solver must receive ``-b``. The displacement term is
    essential after the first iteration and cannot be folded into damping.
    """
    return jnp.concatenate(
        (
            bulk_residual,
            boundary_residual,
            epsilon / jnp.sqrt(2.0) * displacement,
            jnp.zeros_like(displacement),
        )
    )


@partial(jax.jit, static_argnames=("method",))
def _assemble_step_kernel(
    theta_n: jax.Array,
    h: float,
    epsilon: float,
    alpha: float,
    quadrature: QuadratureRule,
    *,
    method: str,
) -> StepSystem:
    root_bulk = jnp.sqrt(quadrature.weights)
    root_boundary = jnp.sqrt(quadrature.boundary_weights)
    bulk = root_bulk[:, None] * operators.bulk_jacobian(
        theta_n, quadrature.points, h
    )
    boundary = jnp.concatenate(
        (
            root_boundary[:, None]
            * operators.value_jacobian(theta_n, quadrature.boundary_points),
            jnp.sqrt(alpha)
            * root_boundary[:, None]
            * operators.tangent_jacobian(
                theta_n, quadrature.boundary_points, quadrature.tangents
            ),
        ),
        axis=0,
    )
    matrix = modified_matrix(bulk, boundary, epsilon)
    return StepSystem(
        theta_n,
        jnp.asarray(h, dtype=theta_n.dtype),
        jnp.asarray(epsilon, dtype=theta_n.dtype),
        jnp.asarray(alpha, dtype=theta_n.dtype),
        quadrature,
        operators.values(theta_n, quadrature.points),
        bulk,
        boundary,
        factorize(matrix, method=method),
        jnp.asarray(0 if method == "qr" else 1, dtype=jnp.int32),
    )


def assemble_step(
    theta_n: jax.Array,
    h: float,
    epsilon: float,
    alpha: float,
    quadrature: QuadratureRule,
    method: str = "qr",
) -> StepSystem:
    """Freeze all Jacobians and factorize once for one physical time step."""
    if not all(math.isfinite(value) and value > 0 for value in (h, epsilon, alpha)):
        raise ValueError("h, epsilon, and alpha must be finite and strictly positive")
    if method not in ("qr", "normal"):
        raise ValueError("method must be 'qr' or 'normal'")
    return _assemble_step_kernel(
        jnp.asarray(theta_n, dtype=jnp.float64),
        h,
        epsilon,
        alpha,
        quadrature,
        method=method,
    )


@partial(jax.jit, static_argnames=("iterations", "method"))
def _advance_step_kernel(
    theta_n: jax.Array,
    system: StepSystem,
    *,
    iterations: int,
    method: str,
) -> StepResult:
    rule = system.quadrature
    root_bulk = jnp.sqrt(rule.weights)
    root_boundary = jnp.sqrt(rule.boundary_weights)
    root_tangent = jnp.sqrt(system.alpha) * root_boundary
    n_bulk = system.bulk_matrix.shape[0]
    n_boundary = system.boundary_matrix.shape[0]
    n_params = theta_n.shape[0]

    def update(theta: jax.Array, _: None):
        bulk_residual = root_bulk * (
            operators.values(theta, rule.points)
            - system.previous_values
            - system.h * operators.laplacians(theta, rule.points)
        )
        boundary_residual = jnp.concatenate(
            (
                root_boundary * operators.values(theta, rule.boundary_points),
                root_tangent
                * operators.tangential_values(
                    theta, rule.boundary_points, rule.tangents
                ),
            )
        )
        rhs = modified_rhs(
            bulk_residual, boundary_residual, theta - theta_n, system.epsilon
        )
        increment = solve_factored(system.factor, -rhs, method=method)
        residual = system.factor.matrix @ increment + rhs
        boundary_end = n_bulk + n_boundary
        displacement_end = boundary_end + n_params
        components = jnp.stack(
            (
                jnp.vdot(residual[:n_bulk], residual[:n_bulk]),
                jnp.vdot(
                    residual[n_bulk:boundary_end], residual[n_bulk:boundary_end]
                ),
                jnp.vdot(
                    residual[boundary_end:displacement_end],
                    residual[boundary_end:displacement_end],
                ),
                jnp.vdot(residual[displacement_end:], residual[displacement_end:]),
            )
        ) / system.h**2
        defect = jnp.sqrt(jnp.sum(components))
        diagnostics = StepDiagnostics(
            defect,
            components,
            jnp.linalg.norm(increment),
            relative_stationarity(system.factor.matrix, increment, -rhs),
            system.h * defect / system.epsilon**2,
        )
        # increment is already Delta theta, not a physical-time velocity.
        return theta + increment, diagnostics

    parameters, diagnostics = jax.lax.scan(
        update, theta_n, xs=None, length=iterations
    )
    return StepResult(parameters, diagnostics)


def advance_step(
    theta_n: jax.Array, step_system: StepSystem, config: RunConfig
) -> StepResult:
    """Perform exactly config.iterations modified Gauss–Newton updates."""
    if config.iterations < 1:
        raise ValueError("iterations must be positive")
    if config.linear_solver not in ("qr", "normal"):
        raise ValueError("linear_solver must be 'qr' or 'normal'")
    if int(step_system.method_code) != (0 if config.linear_solver == "qr" else 1):
        raise ValueError("The requested solver does not match the step factorization")
    if not bool(jnp.array_equal(theta_n, step_system.anchor)):
        raise ValueError("theta_n must match the anchor used to assemble the step")
    return _advance_step_kernel(
        jnp.asarray(theta_n, dtype=jnp.float64),
        step_system,
        iterations=config.iterations,
        method=config.linear_solver,
    )


def _empty_diagnostics(iterations: int) -> StepDiagnostics:
    empty = jnp.empty((0, iterations), dtype=jnp.float64)
    return StepDiagnostics(
        empty,
        jnp.empty((0, iterations, 4), dtype=jnp.float64),
        empty,
        empty,
        empty,
    )


def _is_finite(tree) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree_util.tree_leaves(tree)
    )


def simulate(
    theta0: jax.Array,
    config: RunConfig,
    quadrature: QuadratureRule,
    progress: Callable[[int, int, StepResult], None] | None = None,
) -> SimulationResult:
    """Advance a trajectory, retaining the last valid state on failure.

    ``progress(completed_step, total_steps, step_result)`` runs on the host
    after each successful step. A SimulationFailure contains a trajectory
    ending immediately before its one-based ``failed_step``.
    """
    if config.steps < 1 or config.final_time <= 0:
        raise ValueError("steps and final_time must be positive")
    if config.iterations < 1:
        raise ValueError("iterations must be positive")
    started = perf_counter()
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    if theta.ndim != 1:
        raise ValueError("theta0 must be a flat parameter vector")
    parameters = []
    history: list[StepDiagnostics] = []
    h = config.final_time / config.steps

    def snapshot() -> SimulationResult:
        diagnostics = (
            jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *history)
            if history
            else _empty_diagnostics(config.iterations)
        )
        return SimulationResult(
            jnp.arange(len(parameters), dtype=jnp.float64) * h,
            jnp.stack(parameters)
            if parameters
            else jnp.empty((0, theta.size), dtype=jnp.float64),
            diagnostics,
            perf_counter() - started,
        )

    if not _is_finite(theta):
        raise SimulationFailure("Initial parameters are nonfinite", snapshot(), 0)
    parameters.append(theta)
    for step in range(1, config.steps + 1):
        try:
            system = assemble_step(
                theta,
                h,
                config.epsilon,
                config.alpha,
                quadrature,
                method=config.linear_solver,
            )
            if not _is_finite(system):
                raise FloatingPointError("nonfinite assembled system or factorization")
            result = advance_step(theta, system, config)
            if not _is_finite(result):
                raise FloatingPointError("nonfinite iterate, solve, or diagnostic")
        except Exception as error:
            raise SimulationFailure(
                f"Physical step {step} failed: {error}", snapshot(), step
            ) from error
        theta = result.parameters
        parameters.append(theta)
        history.append(result.diagnostics)
        if progress is not None:
            progress(step, config.steps, result)
    return snapshot()
