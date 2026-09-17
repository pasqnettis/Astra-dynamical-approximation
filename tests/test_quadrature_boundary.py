import jax.numpy as jnp
import numpy as np
import pytest

from rdpa_heat.problem import initial_condition
from rdpa_heat.quadrature import build_quadrature, simpson_rule


@pytest.mark.parametrize("n", [3, 19, 65])
def test_simpson_exact_cubic_moments(n):
    nodes, weights = simpson_rule(n)
    for power in range(4):
        exact = 0.0 if power % 2 else 2 * np.pi ** (power + 1) / (power + 1)
        np.testing.assert_allclose(weights @ nodes**power, exact, atol=3e-13)


@pytest.mark.parametrize("n", [0, 1, 2, 18, -3, 3.5, True])
def test_invalid_quadrature_size(n):
    with pytest.raises(ValueError, match="odd integer"):
        build_quadrature(n)


def test_physical_measures_and_initial_norm():
    quad = build_quadrature(19)
    assert quad.points.shape == (361, 2)
    assert quad.boundary_points.shape == (76, 2)
    np.testing.assert_allclose(quad.weights.sum(), 4 * np.pi**2, atol=1e-13)
    np.testing.assert_allclose(quad.boundary_weights.sum(), 8 * np.pi, atol=1e-13)
    np.testing.assert_allclose(jnp.sqrt(quad.weights @ initial_condition(quad.points)**2), np.pi, atol=1e-13)
    # Tensor Simpson must also integrate coordinate-mixed cubics exactly.
    integral = quad.weights @ (quad.points[:, 0]**2 * quad.points[:, 1]**2)
    np.testing.assert_allclose(integral, (2 * np.pi**3 / 3)**2, rtol=1e-14)


def test_boundary_tangents_and_corner_edge_contributions():
    n = 19
    quad = build_quadrature(n)
    # v(x)=x_1 has zero tangential derivative on vertical edges and one
    # on horizontal edges, unlike its full spatial gradient norm.
    derivative = quad.tangents @ jnp.array([1.0, 0.0])
    np.testing.assert_array_equal(derivative[:2*n], 0)
    np.testing.assert_array_equal(derivative[2*n:], 1)
    np.testing.assert_allclose(quad.boundary_weights @ derivative**2, 4*np.pi)
    corner_mask = np.all(np.isclose(np.abs(quad.boundary_points), np.pi), axis=1)
    assert corner_mask.sum() == 8
    for corner in ((-np.pi, -np.pi), (-np.pi, np.pi), (np.pi, -np.pi), (np.pi, np.pi)):
        matches = np.all(np.isclose(quad.boundary_points, corner), axis=1)
        assert matches.sum() == 2
        np.testing.assert_array_equal(np.sort(np.asarray(quad.tangents)[matches], axis=0), [[0, 0], [1, 1]])
