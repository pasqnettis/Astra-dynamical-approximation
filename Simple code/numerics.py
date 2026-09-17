"""Flat neural representation, physical quadrature, and differentiation."""
import jax
jax.config.update('jax_enable_x64', True)  # Before any numerical arrays.
import jax.numpy as jnp
import numpy as np


def phi(theta, x):
    """153 parameters, in the original checkpoint's ravel_pytree order."""
    z = x + theta[0:2]                         # input translation (2,)
    z = jnp.tanh(theta[2:14].reshape(6, 2) @ z + theta[14:20])
    z = jnp.tanh(theta[20:56].reshape(6, 6) @ z + theta[56:62])
    z = jnp.tanh(theta[62:98].reshape(6, 6) @ z + theta[98:104])
    z = jnp.tanh(theta[104:140].reshape(6, 6) @ z + theta[140:146])
    return theta[146:152] @ z + theta[152]      # scalar output


def initial_parameters(seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    parts = [jnp.zeros(2)]
    for key, width in zip(keys[:4], (2, 6, 6, 6)):
        limit = jnp.sqrt(6.0 / (width + 6))
        parts.extend([jax.random.uniform(key, (6 * width,), dtype=jnp.float64,
                                        minval=-limit, maxval=limit), jnp.zeros(6)])
    limit = jnp.sqrt(6.0 / 7)
    parts.extend([jax.random.uniform(keys[4], (6,), dtype=jnp.float64,
                                    minval=-limit, maxval=limit), jnp.zeros(1)])
    return jnp.concatenate(parts)


def initial_condition(x):
    return jnp.cos(x[..., 0] / 2) * jnp.cos(x[..., 1] / 2)


def build_quadrature(n=19):
    if type(n) is not int or n < 3 or n % 2 != 1:
        raise ValueError('Simpson point count must be an odd integer >= 3')
    axis = jnp.linspace(-jnp.pi, jnp.pi, n)
    coefficients = jnp.where(jnp.arange(n) % 2, 4., 2.).at[0].set(1.).at[-1].set(1.)
    weights = coefficients * (2 * jnp.pi / (n - 1)) / 3
    x, y = jnp.meshgrid(axis, axis, indexing='ij')
    edge = jnp.full(n, jnp.pi)
    # Corners occur twice: once in each incident edge integral.
    boundary = jnp.concatenate([jnp.stack(pair, axis=1) for pair in
                                ((-edge, axis), (edge, axis), (axis, -edge), (axis, edge))])
    return dict(points=jnp.stack((x.ravel(), y.ravel()), axis=1),
                weights=jnp.outer(weights, weights).ravel(), boundary_points=boundary,
                boundary_weights=jnp.tile(weights, 4),
                tangents=jnp.repeat(jnp.array([[0., 1.], [1., 0.]]), 2*n, axis=0))


spatial_hessian = jax.jacfwd(jax.grad(phi, argnums=1), argnums=1)

def laplacian(theta, x):
    return jnp.trace(spatial_hessian(theta, x))

def tangential_derivative(theta, x, tangent):
    return jax.jvp(lambda point: phi(theta, point), (x,), (tangent,))[1]

def implicit_value(theta, x, h):
    return phi(theta, x) - h * laplacian(theta, x)

values = jax.jit(jax.vmap(phi, (None, 0)))
laplacians = jax.jit(jax.vmap(laplacian, (None, 0)))
tangential_values = jax.jit(jax.vmap(tangential_derivative, (None, 0, 0)))
value_jacobian = jax.jit(jax.vmap(jax.grad(phi), (None, 0)))
bulk_jacobian = jax.jit(jax.vmap(jax.grad(implicit_value), (None, 0, None)))
tangent_jacobian = jax.jit(jax.vmap(jax.grad(tangential_derivative), (None, 0, 0)))


def check_finite(*arrays):
    if not all(np.isfinite(np.asarray(a)).all() for a in arrays):
        raise FloatingPointError('Nonfinite numerical result; computation stopped')


def measure(theta, time, quadrature, alpha=1., exact=None):
    q = quadrature
    if exact is None:
        exact = jnp.exp(-time/2) * initial_condition(q['points'])
    if exact.shape != q['weights'].shape:
        raise ValueError('Exact samples must match quadrature weights')
    error = values(theta, q['points']) - exact
    bulk = jnp.sqrt(jnp.sum(q['weights'] * error**2))
    boundary = jnp.sum(q['boundary_weights'] * values(theta, q['boundary_points'])**2)
    tangent = jnp.sum(q['boundary_weights'] * tangential_values(
        theta, q['boundary_points'], q['tangents'])**2)
    result = dict(bulk_l2=bulk, relative_l2=bulk/jnp.sqrt(jnp.sum(q['weights']*exact**2)),
                  boundary_l2=jnp.sqrt(boundary), boundary_tangential=jnp.sqrt(tangent),
                  boundary_weighted=jnp.sqrt(boundary+alpha*tangent))
    check_finite(*result.values())
    return {key: float(value) for key, value in result.items()}
