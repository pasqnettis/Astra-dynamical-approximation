"""Analytic data for the square-domain experiment in Section 5.2."""

import jax.numpy as jnp


def initial_condition(x):
    """Cosine-product initial condition; coordinates occupy the final axis."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.cos(x[..., 0] / 2) * jnp.cos(x[..., 1] / 2)


def exact_solution(t, x):
    """Exact heat solution, used only for initialization and validation."""
    return jnp.exp(-jnp.asarray(t, dtype=jnp.float64) / 2) * initial_condition(x)
