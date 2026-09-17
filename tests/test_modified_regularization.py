import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rdpa_heat.linalg import factorize, relative_stationarity, solve_factored


@pytest.mark.parametrize("method", ["qr", "normal"])
def test_augmented_solver_matches_independent_objective(method):
    rng = np.random.default_rng(31)
    b_bulk, c_boundary = jnp.asarray(rng.normal(size=(8, 4))), jnp.asarray(rng.normal(size=(6, 4)))
    residual_bulk, residual_boundary = jnp.asarray(rng.normal(size=8)), jnp.asarray(rng.normal(size=6))
    displacement = jnp.asarray(rng.normal(size=4))
    epsilon = 0.7
    matrix = jnp.concatenate((b_bulk, c_boundary, epsilon / jnp.sqrt(2) * jnp.eye(4), epsilon * jnp.eye(4)))
    residual = jnp.concatenate((residual_bulk, residual_boundary, epsilon / jnp.sqrt(2) * displacement, jnp.zeros(4)))
    solution = solve_factored(factorize(matrix, method), -residual, method)

    def objective(delta):
        return (
            jnp.sum((b_bulk @ delta + residual_bulk)**2)
            + jnp.sum((c_boundary @ delta + residual_boundary)**2)
            + epsilon**2 / 2 * jnp.sum((delta + displacement)**2)
            + epsilon**2 * jnp.sum(delta**2)
        )

    np.testing.assert_allclose(jax.grad(objective)(solution), 0, atol=1e-12)
    expected = np.linalg.solve(
        b_bulk.T @ b_bulk + c_boundary.T @ c_boundary + 1.5 * epsilon**2 * np.eye(4),
        -b_bulk.T @ residual_bulk - c_boundary.T @ residual_boundary - 0.5 * epsilon**2 * displacement,
    )
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-13)
    assert float(relative_stationarity(matrix, solution, -residual)) < 1e-14
    # Scaling by h converts between the printed and h²-scaled objectives.
    h = 0.03125
    scaled_solution = solve_factored(factorize(matrix / h, method), -residual / h, method)
    np.testing.assert_allclose(scaled_solution, solution, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("method", ["qr", "normal"])
def test_rank_deficient_data_stabilized_by_positive_regularization(method):
    data = jnp.array([[1., 2., 1.], [2., 4., 2.], [0., 0., 0.]])
    epsilon = 1e-2
    matrix = jnp.concatenate((data, epsilon * jnp.eye(3)))
    rhs = jnp.array([1., 2., 3., 0., 0., 0.])
    actual = solve_factored(factorize(matrix, method), rhs, method)
    expected = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    assert np.isfinite(actual).all()


def test_factorization_reused_for_multiple_rhs_and_zero_stationarity():
    matrix = jnp.array([[1., 2.], [2., 1.], [1., 1.]])
    rhs = jnp.array([[1., 2.], [3., 1.], [-1., 4.]])
    factor = factorize(matrix)
    result = solve_factored(factor, rhs)
    np.testing.assert_allclose(result, np.linalg.lstsq(matrix, rhs, rcond=None)[0], rtol=1e-12, atol=1e-13)
    assert relative_stationarity(matrix, jnp.zeros(2), jnp.zeros(3)) == 0


def test_unknown_solver_rejected():
    with pytest.raises(ValueError, match="method"):
        factorize(jnp.eye(2), "unknown")
