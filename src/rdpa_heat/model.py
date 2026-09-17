"""The paper's 153-parameter, four-hidden-layer tanh network."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class Parameters(NamedTuple):
    input_translation: jax.Array
    layers: tuple
    output_weights: jax.Array
    output_bias: jax.Array


def _template():
    return Parameters(
        jnp.zeros(2, dtype=jnp.float64),
        tuple(
            (jnp.zeros((6, width), dtype=jnp.float64), jnp.zeros(6, dtype=jnp.float64))
            for width in (2, 6, 6, 6)
        ),
        jnp.zeros(6, dtype=jnp.float64),
        jnp.zeros((), dtype=jnp.float64),
    )


_TEMPLATE = _template()
_, unravel_parameters = ravel_pytree(_TEMPLATE)
PARAMETER_COUNT = 153


def initialize_parameter_tree(seed=0):
    """Glorot-uniform weights and zero biases, including input translation."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)

    def glorot(key, n_out, n_in):
        limit = jnp.sqrt(6.0 / (n_in + n_out))
        return jax.random.uniform(
            key, (n_out, n_in), dtype=jnp.float64, minval=-limit, maxval=limit
        )

    layers = tuple(
        (glorot(keys[i], 6, width), jnp.zeros(6, dtype=jnp.float64))
        for i, width in enumerate((2, 6, 6, 6))
    )
    return Parameters(
        jnp.zeros(2, dtype=jnp.float64),
        layers,
        glorot(keys[4], 1, 6).reshape(6),
        jnp.zeros((), dtype=jnp.float64),
    )


def initialize_parameters(seed=0):
    """Return the flat parameter vector used by all integrators."""
    return ravel_pytree(initialize_parameter_tree(seed))[0]


def phi(theta, x):
    """Evaluate the scalar network at a single physical coordinate pair."""
    params = unravel_parameters(theta)
    value = x + params.input_translation
    for weights, bias in params.layers:
        value = jnp.tanh(weights @ value + bias)
    return params.output_weights @ value + params.output_bias


def parameter_layout():
    """JSON-safe leaf names, shapes and offsets in ravel_pytree order."""
    leaves = [("input_translation", _TEMPLATE.input_translation)]
    for i, (weights, bias) in enumerate(_TEMPLATE.layers):
        leaves.extend([(f"layers.{i}.weights", weights), (f"layers.{i}.bias", bias)])
    leaves.extend([
        ("output_weights", _TEMPLATE.output_weights),
        ("output_bias", _TEMPLATE.output_bias),
    ])
    result, offset = [], 0
    for order, (name, value) in enumerate(leaves):
        size = value.size
        result.append({
            "name": name, "shape": list(value.shape), "order": order,
            "offset": offset, "size": size, "dtype": "float64",
        })
        offset += size
    return result
