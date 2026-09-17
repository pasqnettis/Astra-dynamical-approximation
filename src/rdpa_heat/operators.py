"""Batched spatial derivatives and parameter Jacobians, compiled by shape."""

import jax
import jax.numpy as jnp

from .model import phi


_spatial_hessian = jax.jacfwd(jax.grad(phi, argnums=1), argnums=1)


def laplacian(theta, x):
    """Trace of the forward-over-reverse spatial Hessian."""
    return jnp.trace(_spatial_hessian(theta, x))


def tangential_value(theta, x, tangent):
    return jax.jvp(lambda point: phi(theta, point), (x,), (tangent,))[1]


def _implicit_value(theta, x, h):
    return phi(theta, x) - h * laplacian(theta, x)


values = jax.jit(jax.vmap(phi, in_axes=(None, 0)))
laplacians = jax.jit(jax.vmap(laplacian, in_axes=(None, 0)))
tangential_values = jax.jit(jax.vmap(tangential_value, in_axes=(None, 0, 0)))
value_jacobian = jax.jit(jax.vmap(jax.grad(phi, argnums=0), in_axes=(None, 0)))
boundary_jacobian = value_jacobian
bulk_jacobian = jax.jit(jax.vmap(
    jax.grad(_implicit_value, argnums=0), in_axes=(None, 0, None)
))
tangent_jacobian = jax.jit(jax.vmap(
    jax.grad(tangential_value, argnums=0), in_axes=(None, 0, 0)
))
