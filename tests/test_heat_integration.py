"""Independent algebraic and control-flow checks of physical heat evolution."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rdpa_heat import integrator
from rdpa_heat.linalg import factorize, solve_factored
from rdpa_heat.quadrature import QuadratureRule


def _config(**overrides):
    defaults = dict(
        final_time=0.4,
        steps=2,
        epsilon=0.1,
        alpha=0.2,
        iterations=3,
        linear_solver="qr",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _rule():
    return QuadratureRule(
        points=jnp.array([[0.0, 0.0]]),
        weights=jnp.ones(1),
        boundary_points=jnp.array([[jnp.pi, 0.0]]),
        boundary_weights=jnp.ones(1),
        tangents=jnp.array([[0.0, 1.0]]),
    )


def _scalar_operators(monkeypatch, *, constant=False):
    """Replace only the field model, leaving actual step assembly and solve."""
    counts = {"bulk_jacobian": [], "value_jacobian": [], "tangent_jacobian": []}

    def basis(points):
        if constant:
            return jnp.ones(points.shape[0])
        return jnp.cos(points[:, 0] / 2) * jnp.cos(points[:, 1] / 2)

    decay = 0.0 if constant else 0.5
    monkeypatch.setattr(integrator.operators, "values", lambda q, x: q[0] * basis(x))
    monkeypatch.setattr(
        integrator.operators, "laplacians", lambda q, x: -decay * q[0] * basis(x)
    )
    monkeypatch.setattr(
        integrator.operators,
        "tangential_values",
        lambda q, x, t: jnp.zeros(x.shape[0]),
    )

    def bulk_jacobian(q, x, h):
        counts["bulk_jacobian"].append(np.asarray(q).copy())
        return ((1 + h * decay) * basis(x))[:, None]

    def value_jacobian(q, x):
        counts["value_jacobian"].append(np.asarray(q).copy())
        return basis(x)[:, None]

    def tangent_jacobian(q, x, t):
        counts["tangent_jacobian"].append(np.asarray(q).copy())
        return jnp.zeros((x.shape[0], q.size))

    monkeypatch.setattr(integrator.operators, "bulk_jacobian", bulk_jacobian)
    monkeypatch.setattr(integrator.operators, "value_jacobian", value_jacobian)
    monkeypatch.setattr(integrator.operators, "tangent_jacobian", tangent_jacobian)
    return counts


@pytest.mark.parametrize("method", ["qr", "normal"])
def test_augmented_system_matches_explicit_modified_objective(method):
    bulk = jnp.array([[1.0, 0.3], [0.2, 1.4], [-0.4, 0.1]])
    boundary = jnp.array([[0.7, -0.2], [0.0, 0.5]])
    rb = jnp.array([0.1, -0.3, 0.2])
    rc = jnp.array([0.5, -0.1])
    displacement = jnp.array([0.4, -0.7])
    epsilon = 0.3
    matrix = integrator.modified_matrix(bulk, boundary, epsilon)
    rhs = integrator.modified_rhs(rb, rc, displacement, epsilon)
    increment = solve_factored(factorize(matrix, method=method), -rhs, method=method)
    expected_matrix = bulk.T @ bulk + boundary.T @ boundary + 1.5 * epsilon**2 * jnp.eye(2)
    expected_rhs = -bulk.T @ rb - boundary.T @ rc - 0.5 * epsilon**2 * displacement
    np.testing.assert_allclose(matrix.T @ matrix, expected_matrix, atol=2e-15)
    np.testing.assert_allclose(-matrix.T @ rhs, expected_rhs, atol=2e-15)
    np.testing.assert_allclose(increment, np.linalg.solve(expected_matrix, expected_rhs), atol=2e-14)

    def objective(delta):
        return (
            jnp.sum((bulk @ delta + rb) ** 2)
            + jnp.sum((boundary @ delta + rc) ** 2)
            + epsilon**2 / 2 * jnp.sum((delta + displacement) ** 2)
            + epsilon**2 * jnp.sum(delta**2)
        )

    np.testing.assert_allclose(jax.grad(objective)(increment), 0.0, atol=2e-14)
    # Rescaling the full objective by h² must not rescale its minimizer.
    h = 0.13
    scaled = solve_factored(
        factorize(matrix / h, method=method), -rhs / h, method=method
    )
    np.testing.assert_allclose(scaled, increment, atol=2e-14)


def test_rank_deficient_data_matrix_is_regularized():
    bulk = jnp.array([[1.0, 1.0], [2.0, 2.0]])
    boundary = jnp.zeros((1, 2))
    matrix = integrator.modified_matrix(bulk, boundary, 1e-3)
    rhs = integrator.modified_rhs(jnp.ones(2), jnp.zeros(1), jnp.zeros(2), 1e-3)
    result = solve_factored(factorize(matrix, method="qr"), -rhs, method="qr")
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[0], result[1], atol=1e-10)


def test_physical_iteration_matches_scalar_recurrence_and_rebuilds(monkeypatch):
    config = _config()
    with jax.disable_jit():
        counts = _scalar_operators(monkeypatch)
        events = []
        result = integrator.simulate(
            jnp.array([1.0]), config, _rule(),
            progress=lambda n, total, step: events.append((n, total)),
        )
    h = config.final_time / config.steps
    coefficient = 1 + h / 2
    expected = [1.0]
    for _ in range(config.steps):
        anchor = expected[-1]
        q = anchor
        for _ in range(config.iterations):
            delta = -(
                coefficient * (coefficient * q - anchor)
                + 0.5 * config.epsilon**2 * (q - anchor)
            ) / (coefficient**2 + 1.5 * config.epsilon**2)
            q += delta
        expected.append(q)
    np.testing.assert_allclose(result.parameters[:, 0], expected, atol=3e-15)
    assert result.diagnostics.defects.shape == (config.steps, config.iterations)
    np.testing.assert_allclose(
        jnp.sum(result.diagnostics.components, axis=-1),
        result.diagnostics.defects**2,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        result.diagnostics.defect_ratios,
        h * result.diagnostics.defects / config.epsilon**2,
    )
    for recorded_anchors in counts.values():
        assert len(recorded_anchors) == config.steps
        np.testing.assert_allclose(np.array(recorded_anchors)[:, 0], expected[:-1])
    assert events == [(1, 2), (2, 2)]


def test_nonzero_previous_boundary_trace_is_penalized(monkeypatch):
    config = _config(iterations=1)
    theta = jnp.array([1.0])
    with jax.disable_jit():
        _scalar_operators(monkeypatch, constant=True)
        system = integrator.assemble_step(theta, 0.2, config.epsilon, config.alpha, _rule())
        result = integrator.advance_step(theta, system, config)
    expected = 1 - 1 / (2 + 1.5 * config.epsilon**2)
    np.testing.assert_allclose(result.parameters, [expected], atol=2e-15)
    assert result.parameters[0] < 0.6


def test_qr_and_normal_trajectories_agree(monkeypatch):
    with jax.disable_jit():
        _scalar_operators(monkeypatch)
        qr = integrator.simulate(jnp.array([0.8]), _config(linear_solver="qr"), _rule())
        normal = integrator.simulate(
            jnp.array([0.8]), _config(linear_solver="normal"), _rule()
        )
    np.testing.assert_allclose(qr.parameters, normal.parameters, atol=2e-14)


def test_failure_preserves_last_completed_step(monkeypatch):
    config = _config(steps=3)
    with jax.disable_jit():
        _scalar_operators(monkeypatch)
        original = integrator.advance_step
        calls = 0

        def fail_second(theta, system, cfg):
            nonlocal calls
            calls += 1
            result = original(theta, system, cfg)
            if calls == 2:
                return result._replace(parameters=jnp.full_like(result.parameters, jnp.nan))
            return result

        monkeypatch.setattr(integrator, "advance_step", fail_second)
        with pytest.raises(integrator.SimulationFailure) as caught:
            integrator.simulate(jnp.array([1.0]), config, _rule())
    failure = caught.value
    assert failure.failed_step == 2
    assert failure.partial_result.parameters.shape == (2, 1)
    assert failure.partial_result.diagnostics.defects.shape == (1, config.iterations)
    assert np.all(np.isfinite(failure.partial_result.parameters))
    np.testing.assert_allclose(failure.partial_result.times, [0.0, config.final_time / 3])


def test_production_jit_assembly_and_scan():
    from rdpa_heat.model import initialize_parameters
    from rdpa_heat.quadrature import build_quadrature

    theta = initialize_parameters(seed=2)
    config = _config(final_time=0.01, steps=1, iterations=2)
    rule = build_quadrature(19)
    system = integrator.assemble_step(
        theta, config.final_time, config.epsilon, config.alpha, rule
    )
    assert system.factor.matrix.shape == (819, 153)
    result = integrator.advance_step(theta, system, config)
    assert result.parameters.shape == (153,)
    assert result.diagnostics.defects.shape == (2,)
    assert np.all(np.isfinite(result.parameters))
    assert np.all(np.isfinite(result.diagnostics.components))
    assert float(jnp.max(result.diagnostics.stationarity)) < 1e-11


def test_alpha_weights_squared_tangent_term_and_system_contract():
    from rdpa_heat.model import initialize_parameters
    from rdpa_heat.quadrature import build_quadrature

    theta = initialize_parameters(0)
    rule = build_quadrature(3)
    first = integrator.assemble_step(theta, 0.1, 0.1, 0.2, rule)
    second = integrator.assemble_step(theta, 0.1, 0.1, 0.8, rule)
    n_boundary = len(rule.boundary_points)
    np.testing.assert_allclose(first.boundary_matrix[:n_boundary], second.boundary_matrix[:n_boundary])
    np.testing.assert_allclose(2 * first.boundary_matrix[n_boundary:], second.boundary_matrix[n_boundary:], atol=1e-14)
    with pytest.raises(ValueError, match="factorization"):
        integrator.advance_step(theta, first, _config(linear_solver="normal"))
    with pytest.raises(ValueError, match="anchor"):
        integrator.advance_step(theta + 1, first, _config())
    with pytest.raises(ValueError, match="finite"):
        integrator.assemble_step(theta, float("nan"), 0.1, 0.2, rule)
