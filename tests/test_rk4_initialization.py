"""Analytic checks distinguish frozen-target RK4 from a relaxation flow."""

from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rdpa_heat.config import InitializationConfig, RefinementPass, RunConfig
from rdpa_heat import initialization as init
from rdpa_heat.model import initialize_parameters
from rdpa_heat.quadrature import build_quadrature


def test_classical_rk4_uses_four_distinct_stage_states():
    observed = []

    def velocity(q):
        observed.append(float(q[0]))
        return q

    result = init.rk4_step(velocity, jnp.array([1.0]), 0.2)
    np.testing.assert_allclose(observed, [1.0, 1.1, 1.11, 1.222], atol=1e-14)
    np.testing.assert_allclose(result, [1 + .2 + .2**2 / 2 + .2**3 / 6 + .2**4 / 24])


@pytest.mark.parametrize("method", ["qr", "normal"])
def test_affine_refinement_has_exact_constant_velocity_endpoint(method):
    matrix = jnp.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.3]])
    anchor = jnp.array([0.2, -0.4])
    weights = jnp.array([0.5, 1.0, 2.0])
    correction = jnp.array([0.6, -0.2, 1.5])
    epsilon = 0.15
    expected_velocity = np.linalg.solve(
        np.asarray(matrix.T @ (weights[:, None] * matrix)) + epsilon**2 * np.eye(2),
        np.asarray(matrix.T @ (weights * correction)),
    )
    result = init.integrate_refinement_pass(
        anchor, correction, weights, epsilon, lambda q: matrix, 7, method
    )
    assert bool(result.finite)
    assert int(result.completed_steps) == 7
    np.testing.assert_allclose(result.theta, anchor + expected_velocity, atol=1e-13)


def test_nonlinear_frozen_correction_flow_has_fourth_order_convergence():
    anchor, epsilon, correction = 0.5, 0.2, 1.0

    # Independent scalar root of the integrated ODE, using only Python math.
    lower, upper = anchor, 2.0
    for _ in range(100):
        middle = (lower + upper) / 2
        integral = middle**2 - anchor**2 + epsilon**2 / 2 * math.log(middle / anchor)
        if integral < correction:
            lower = middle
        else:
            upper = middle
    exact = (lower + upper) / 2

    def jacobian(q):
        return 2 * q.reshape(1, 1)

    errors = []
    for steps in (8, 16, 32):
        result = init.integrate_refinement_pass(
            jnp.array([anchor]), jnp.array([correction]), jnp.ones(1), epsilon,
            jacobian, steps
        )
        errors.append(abs(float(result.theta[0]) - exact))
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders > 3.6) & (orders < 4.5)), (errors, orders)


def test_jacobian_is_recomputed_for_every_stage():
    observed = []

    def jacobian(q):
        observed.append(float(q[0]))
        return 2 * q.reshape(1, 1)

    with jax.disable_jit():
        init.integrate_refinement_pass(
            jnp.array([0.5]), jnp.ones(1), jnp.ones(1), 0.2, jacobian, 2
        )
    assert len(observed) == 8
    assert len(set(observed)) == 8


def test_chunking_preserves_frozen_forcing_over_whole_pass():
    def jacobian(q):
        return 2 * q.reshape(1, 1)

    args = (jnp.array([0.8]), jnp.ones(1), 0.2, jacobian)
    full = init.integrate_refinement_pass(jnp.array([0.5]), *args, 12)
    first = init.integrate_refinement_pass(jnp.array([0.5]), *args, 5, step_size=1 / 12)
    second = init.integrate_refinement_pass(first.theta, *args, 7, step_size=1 / 12)
    np.testing.assert_allclose(second.theta, full.theta, atol=1e-14)


def test_nonfinite_stage_returns_last_valid_parameter_state():
    def broken_jacobian(q):
        return jnp.array([[jnp.nan]])

    anchor = jnp.array([0.4])
    result = init.integrate_refinement_pass(
        anchor, jnp.ones(1), jnp.ones(1), 0.2, broken_jacobian, 3
    )
    assert not bool(result.finite)
    assert int(result.completed_steps) == 0
    np.testing.assert_array_equal(result.theta, anchor)


def test_rejected_refinement_pass_preserves_anchor(monkeypatch):
    theta = initialize_parameters(2)
    quadrature = build_quadrature(3)
    cfg = RunConfig(validation_points=3, initialization=InitializationConfig(
        adam_steps=0, passes=(RefinementPass(1, 1e-2),)
    ))
    monkeypatch.setattr(init, "_metrics", lambda q, *args: {
        "bulk_error": float(jnp.linalg.norm(q)), "training_bulk_error": 0.0,
        "boundary_l2": 0.0, "tangential_seminorm": 0.0,
    })
    monkeypatch.setattr(init, "_network_refinement_chunk", lambda q, *args:
                        init.RefinementIntegration(2 * q, jnp.array(True), jnp.array(1)))
    result = init.refine_initial_condition(theta, cfg, quadrature)
    np.testing.assert_array_equal(result.theta, theta)
    assert result.history[-1]["accepted"] is False
    assert result.final_error == result.initial_error


def test_refinement_is_independent_of_physical_h_alpha_and_epsilon():
    theta = initialize_parameters(1)
    quadrature = build_quadrature(3)
    base = RunConfig(validation_points=3, initialization=InitializationConfig(
        adam_steps=0, passes=(RefinementPass(1, 0.1),)
    ))
    changed = replace(base, final_time=2.0, steps=2, alpha=4.0, epsilon=0.8, iterations=3)
    first = init.refine_initial_condition(theta, base, quadrature)
    second = init.refine_initial_condition(theta, changed, quadrature)
    np.testing.assert_array_equal(first.theta, second.theta)
    assert first.history == second.history


def test_warmstart_preserves_best_independent_validation_checkpoint(monkeypatch):
    # An artificial validation minimum at the first update ensures a final
    # optimizer iterate cannot silently replace the best saved checkpoint.
    theta = initialize_parameters(0)
    cfg = RunConfig(validation_points=3, initialization=InitializationConfig(
        adam_steps=3, validation_every=1, passes=()
    ))
    monkeypatch.setattr(init, "initialize_parameters", lambda seed: theta)
    monkeypatch.setattr(init, "_adam_chunk", lambda q, state, *args:
                        (q + 1, state, jnp.array(True), jnp.array(1)))
    monkeypatch.setattr(init, "_metrics", lambda q, *args: {
        "bulk_error": float((q[0] - 1)**2), "training_bulk_error": 0.0,
        "boundary_l2": 0.0, "tangential_seminorm": 0.0,
    })
    result = init.fit_initial_condition(cfg, build_quadrature(3))
    np.testing.assert_array_equal(result.theta, theta + 1)
    assert result.final_error == 0.0


def test_production_warmstart_smoke_is_finite_and_reduces_validation_error():
    cfg = RunConfig(validation_points=5, initialization=InitializationConfig(
        adam_steps=20, validation_every=10, passes=()
    ))
    result = init.fit_initial_condition(cfg, build_quadrature(3))
    assert np.isfinite(result.theta).all()
    assert result.final_error < result.initial_error
    assert [r["update"] for r in result.history if r["stage"] == "adam"] == [0, 10, 20]
