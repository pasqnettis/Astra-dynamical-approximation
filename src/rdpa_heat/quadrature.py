"""Physical Simpson quadrature on the square and its four separate edges."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class QuadratureRule(NamedTuple):
    points: jax.Array
    weights: jax.Array
    boundary_points: jax.Array
    boundary_weights: jax.Array
    tangents: jax.Array


def simpson_rule(n):
    """Return n nodes and physical weights on [-pi, pi], n odd and >= 3."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 3 or n % 2 != 1:
        raise ValueError("Simpson quadrature needs an odd integer n >= 3")
    points = jnp.linspace(-jnp.pi, jnp.pi, n, dtype=jnp.float64)
    coefficients = jnp.where(jnp.arange(n) % 2 == 1, 4.0, 2.0)
    coefficients = coefficients.at[0].set(1.0).at[-1].set(1.0)
    return points, coefficients * (2 * jnp.pi / (n - 1)) / 3


def build_quadrature(config=19):
    """Build tensor-product bulk and edge-specific boundary quadrature.

    Corners intentionally appear twice, once for each incident edge integral.
    Accept either a RunConfig or an odd point count. The same point count is
    used in each coordinate and on every edge.
    """
    from .config import RunConfig

    n = config.quadrature_points if isinstance(config, RunConfig) else config
    axis, weights = simpson_rule(n)
    x, y = jnp.meshgrid(axis, axis, indexing="ij")
    points = jnp.stack((x.ravel(), y.ravel()), axis=-1)
    bulk_weights = (weights[:, None] * weights[None, :]).ravel()
    const = jnp.full(n, jnp.pi, dtype=jnp.float64)
    boundary = jnp.concatenate((
        jnp.stack((-const, axis), axis=-1),
        jnp.stack((const, axis), axis=-1),
        jnp.stack((axis, -const), axis=-1),
        jnp.stack((axis, const), axis=-1),
    ))
    tangents = jnp.concatenate((
        jnp.tile(jnp.array([0.0, 1.0]), (2 * n, 1)),
        jnp.tile(jnp.array([1.0, 0.0]), (2 * n, 1)),
    ))
    return QuadratureRule(points, bulk_weights, boundary, jnp.tile(weights, 4), tangents)
