"""Initial fitting and frozen-correction RK4 in artificial time.

The correction is frozen for a whole pass, while the parameter Jacobian is
recomputed at every RK4 stage. No physical time step, heat operator, or boundary
penalty enters either the fit or the refinement velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import RunConfig
from .linalg import factorize, solve_factored
from .model import initialize_parameters
from .operators import tangential_values, value_jacobian, values
from .problem import initial_condition
from .quadrature import QuadratureRule, build_quadrature


@dataclass
class InitializationResult:
    theta: jax.Array
    history: list[dict]
    initial_error: float
    final_error: float


class InitializationFailure(FloatingPointError):
    """Numerical failure carrying the last valid parameters and diagnostics."""

    def __init__(self, message: str, theta: jax.Array, history: list[dict]):
        super().__init__(message)
        self.theta = theta
        self.history = history


class RefinementIntegration(NamedTuple):
    theta: jax.Array
    finite: jax.Array
    completed_steps: jax.Array


def rk4_step(velocity: Callable, q: jax.Array, step: float) -> jax.Array:
    """A classical RK4 step; all four stage velocities are freshly evaluated."""
    k1 = velocity(q)
    k2 = velocity(q + 0.5 * step * k1)
    k3 = velocity(q + 0.5 * step * k2)
    k4 = velocity(q + step * k3)
    return q + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ridge_velocity(
    q: jax.Array,
    correction: jax.Array,
    weights: jax.Array,
    epsilon: float,
    jacobian: Callable,
    method: str = "qr",
) -> jax.Array:
    """Solve the bulk-only regularized tangent fit at the current stage q."""
    sqrt_weights = jnp.sqrt(weights)
    matrix = jnp.concatenate(
        (sqrt_weights[:, None] * jacobian(q), epsilon * jnp.eye(q.size)), axis=0
    )
    rhs = jnp.concatenate((sqrt_weights * correction, jnp.zeros_like(q)))
    return solve_factored(factorize(matrix, method), rhs, method)


def _checked_rk4_step(velocity: Callable, q: jax.Array, step: float):
    # Check the stages as well as the endpoint: a nonfinite internal solve may
    # never be hidden by retaining only the endpoint of a scan chunk.
    k1 = velocity(q)
    q2 = q + 0.5 * step * k1
    k2 = velocity(q2)
    q3 = q + 0.5 * step * k2
    k3 = velocity(q3)
    q4 = q + step * k3
    k4 = velocity(q4)
    endpoint = q + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    finite = jnp.all(jnp.isfinite(jnp.stack((q, k1, q2, k2, q3, k3, q4, k4, endpoint))))
    return jnp.where(finite, endpoint, q), finite


@partial(jax.jit, static_argnames=("jacobian", "steps", "method"))
def integrate_refinement_pass(
    theta: jax.Array,
    correction: jax.Array,
    weights: jax.Array,
    epsilon: float,
    jacobian: Callable,
    steps: int,
    method: str = "qr",
    step_size: float | None = None,
) -> RefinementIntegration:
    """Integrate a supplied *constant* correction over a pass or a scan chunk.

    A custom step_size lets the production caller split a pass into progress
    chunks without changing its forcing or artificial-time discretization.
    The callable jacobian takes only the current flat parameter vector.
    """
    step = 1.0 / steps if step_size is None else step_size

    def velocity(q):
        return ridge_velocity(q, correction, weights, epsilon, jacobian, method)

    def advance(carry, _):
        q, active, count = carry
        endpoint, finite = jax.lax.cond(
            active,
            lambda current: _checked_rk4_step(velocity, current, step),
            lambda current: (current, jnp.array(False)),
            q,
        )
        return (endpoint, active & finite, count + (active & finite)), None

    result, _ = jax.lax.scan(
        advance, (theta, jnp.array(True), jnp.array(0, dtype=jnp.int32)), None, length=steps
    )
    return RefinementIntegration(*result)


# Passing points as numerical arguments to this kernel avoids making a new
# compiled closure for each pass or physical configuration.
@partial(jax.jit, static_argnames=("steps", "method"))
def _network_refinement_chunk(theta, correction, points, weights, epsilon, steps, step_size, method):
    return integrate_refinement_pass(
        theta, correction, weights, epsilon,
        lambda q: value_jacobian(q, points), steps, method, step_size
    )


_adam_transform = optax.scale_by_adam()


def _fit_loss(theta, points, weights, target):
    return jnp.sum(weights * jnp.square(values(theta, points) - target))


_fit_value_and_grad = jax.value_and_grad(_fit_loss)


@partial(jax.jit, static_argnames=("steps",))
def _adam_chunk(theta, optimizer_state, start, steps, points, weights, target,
                learning_rate, learning_rate_late, decay_step):
    def update(carry, offset):
        q, opt_state, active, count = carry

        def take_step(args):
            current, state = args
            loss, gradient = _fit_value_and_grad(current, points, weights, target)
            direction, next_state = _adam_transform.update(gradient, state, current)
            rate = jnp.where(start + offset < decay_step, learning_rate, learning_rate_late)
            candidate = optax.apply_updates(current, -rate * direction)
            leaves = jax.tree_util.tree_leaves((loss, gradient, candidate, next_state))
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
            next_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(finite, new, old), next_state, state
            )
            return jnp.where(finite, candidate, current), next_state, finite

        candidate, next_state, finite = jax.lax.cond(
            active, take_step, lambda args: (*args, jnp.array(False)), (q, opt_state)
        )
        return (candidate, next_state, active & finite, count + (active & finite)), None

    result, _ = jax.lax.scan(
        update,
        (theta, optimizer_state, jnp.array(True), jnp.array(0, dtype=jnp.int32)),
        jnp.arange(steps),
    )
    return result


def _bulk_error(theta, quadrature):
    residual = values(theta, quadrature.points) - jax.vmap(initial_condition)(quadrature.points)
    return float(jnp.sqrt(jnp.sum(quadrature.weights * residual**2)))


def _metrics(theta, quadrature, validation):
    boundary = values(theta, validation.boundary_points)
    tangent = tangential_values(theta, validation.boundary_points, validation.tangents)
    return {
        "bulk_error": _bulk_error(theta, validation),
        "training_bulk_error": _bulk_error(theta, quadrature),
        "boundary_l2": float(jnp.sqrt(jnp.sum(validation.boundary_weights * boundary**2))),
        "tangential_seminorm": float(jnp.sqrt(jnp.sum(validation.boundary_weights * tangent**2))),
    }


def _record(history, record, theta, progress):
    numbers = [v for v in record.values() if isinstance(v, (float, np.floating))]
    if not np.all(np.isfinite(numbers)):
        # Keep failure histories valid JSON so reporting the numerical failure
        # cannot itself fail while trying to serialize NaN or infinity.
        history.append({
            key: (str(value) if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
            for key, value in record.items()
        })
        raise InitializationFailure("Nonfinite initialization diagnostics", theta, history)
    history.append(record)
    if progress is not None:
        progress(record)


def refine_initial_condition(theta_anchor, config: RunConfig, quadrature: QuadratureRule,
                             progress=None) -> InitializationResult:
    """Run all configured correction passes, accepting only bulk improvement."""
    validation = build_quadrature(config.validation_points)
    theta = jnp.asarray(theta_anchor, dtype=jnp.float64)
    history: list[dict] = []
    metrics = _metrics(theta, quadrature, validation)
    _record(history, {"stage": "refinement_start", **metrics}, theta, progress)
    initial_error = metrics["bulk_error"]

    for pass_index, spec in enumerate(config.initialization.passes, start=1):
        anchor = theta
        before = metrics["bulk_error"]
        # Freeze this vector once, before every step/stage of the whole pass.
        correction = jax.vmap(initial_condition)(quadrature.points) - values(anchor, quadrature.points)
        completed = 0
        while completed < spec.steps:
            count = min(10, spec.steps - completed)
            result = _network_refinement_chunk(
                theta, correction, quadrature.points, quadrature.weights, spec.epsilon,
                count, 1.0 / spec.steps, config.linear_solver
            )
            theta = result.theta
            completed += int(result.completed_steps)
            if not bool(result.finite):
                history.append({"stage": "refinement_failure", "pass": pass_index,
                                "completed_steps": completed})
                raise InitializationFailure(
                    f"Nonfinite RK4 stage in refinement pass {pass_index}", theta, history
                )
            if progress is not None:
                progress({"stage": "refinement_progress", "pass": pass_index,
                          "completed_steps": completed, "steps": spec.steps})
        candidate_metrics = _metrics(theta, quadrature, validation)
        accepted = candidate_metrics["bulk_error"] < before
        record = {"stage": "refinement", "pass": pass_index, "steps": spec.steps,
                  "epsilon": spec.epsilon, "accepted": accepted, "error_before": before,
                  **candidate_metrics}
        # If validation itself overflows, the accepted anchor is the last
        # state whose function-space diagnostics are known to be finite.
        _record(history, record, anchor, progress)
        if accepted:
            metrics = candidate_metrics
        else:
            theta = anchor
    return InitializationResult(theta, history, initial_error, metrics["bulk_error"])


def fit_initial_condition(config: RunConfig, quadrature: QuadratureRule,
                          progress=None) -> InitializationResult:
    """Adam warm start selected on independent quadrature, followed by RK4."""
    spec = config.initialization
    validation = build_quadrature(config.validation_points)
    theta = initialize_parameters(config.seed)
    optimizer_state = _adam_transform.init(theta)
    target = jax.vmap(initial_condition)(quadrature.points)
    history: list[dict] = []
    metrics = _metrics(theta, quadrature, validation)
    _record(history, {"stage": "adam", "update": 0, **metrics}, theta, progress)
    initial_error = metrics["bulk_error"]
    best_error, best_theta = initial_error, theta

    completed = 0
    while completed < spec.adam_steps:
        count = min(spec.validation_every, spec.adam_steps - completed)
        theta, optimizer_state, finite, steps_done = _adam_chunk(
            theta, optimizer_state, completed, count, quadrature.points, quadrature.weights,
            target, spec.learning_rate, spec.learning_rate_late, spec.decay_step
        )
        completed += int(steps_done)
        if not bool(finite):
            history.append({"stage": "adam_failure", "update": completed})
            raise InitializationFailure("Nonfinite Adam update", theta, history)
        metrics = _metrics(theta, quadrature, validation)
        improved = metrics["bulk_error"] < best_error
        _record(history, {"stage": "adam", "update": completed,
                          "best_so_far": improved, **metrics}, best_theta, progress)
        if improved:
            best_error, best_theta = metrics["bulk_error"], theta

    history.append({"stage": "adam_selected", "bulk_error": best_error})
    try:
        refined = refine_initial_condition(best_theta, config, quadrature, progress)
    except InitializationFailure as error:
        error.history = history + error.history
        raise
    return InitializationResult(
        refined.theta, history + refined.history, initial_error, refined.final_error
    )
