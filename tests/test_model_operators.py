import json

import jax
import jax.numpy as jnp
import numpy as np

from rdpa_heat.model import (
    PARAMETER_COUNT, initialize_parameters, parameter_layout, phi, unravel_parameters,
)
from rdpa_heat.operators import (
    bulk_jacobian, laplacian, laplacians, tangent_jacobian,
    tangential_values, value_jacobian, values,
)
from rdpa_heat.problem import exact_solution, initial_condition


def test_model_shape_initialization_and_layout():
    theta = initialize_parameters(0)
    assert theta.shape == (PARAMETER_COUNT,) == (153,)
    assert theta.dtype == jnp.float64
    assert phi(theta, jnp.array([0.2, -0.4])).shape == ()
    np.testing.assert_array_equal(theta, initialize_parameters(0))
    assert not np.array_equal(theta, initialize_parameters(1))
    tree = unravel_parameters(theta)
    np.testing.assert_array_equal(tree.input_translation, 0)
    for _, bias in tree.layers:
        np.testing.assert_array_equal(bias, 0)
    assert tree.output_bias == 0
    layout = parameter_layout()
    json.dumps(layout)
    assert sum(leaf["size"] for leaf in layout) == 153
    for record, leaf in zip(layout, jax.tree_util.tree_leaves(tree), strict=True):
        assert record["shape"] == list(leaf.shape)
        offset = record["offset"]
        np.testing.assert_array_equal(theta[offset:offset + record["size"]], leaf.ravel())


def test_heat_analytic_data():
    point = jnp.array([0.37, -0.82])
    lap = jnp.trace(jax.hessian(initial_condition)(point))
    np.testing.assert_allclose(lap, -initial_condition(point) / 2, atol=1e-14)
    np.testing.assert_allclose(exact_solution(0.6, point), jnp.exp(-0.3) * initial_condition(point))
    np.testing.assert_allclose(initial_condition(jnp.array([[jnp.pi, 0], [0, -jnp.pi]])), 0, atol=1e-15)


def test_spatial_and_parameter_derivatives_match_finite_differences():
    theta = initialize_parameters(3)
    points = jnp.array([[0.13, -0.47], [0.74, 0.26]])
    tangents = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    direction = jax.random.normal(jax.random.PRNGKey(5), theta.shape, dtype=jnp.float64)
    direction /= jnp.linalg.norm(direction)
    eps = 1e-5
    for function, jacobian in (
        (lambda q: values(q, points), value_jacobian(theta, points)),
        (lambda q: values(q, points) - 0.03 * laplacians(q, points),
         bulk_jacobian(theta, points, 0.03)),
        (lambda q: tangential_values(q, points, tangents),
         tangent_jacobian(theta, points, tangents)),
    ):
        finite_difference = (function(theta + eps * direction) - function(theta - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(jacobian @ direction, finite_difference, rtol=2e-7, atol=1e-9)
    spatial_eps = 1e-4
    point = points[0]
    finite_lap = sum(
        (phi(theta, point + spatial_eps * d) - 2 * phi(theta, point)
         + phi(theta, point - spatial_eps * d)) / spatial_eps**2
        for d in jnp.eye(2)
    )
    np.testing.assert_allclose(laplacian(theta, point), finite_lap, rtol=1e-6, atol=1e-7)
    finite_tangent = (values(theta, points + eps * tangents) - values(theta, points - eps * tangents)) / (2 * eps)
    np.testing.assert_allclose(tangential_values(theta, points, tangents), finite_tangent, rtol=1e-7, atol=1e-9)
