"""Reusable QR and normal-equation factorizations for regularized systems.

These kernels are pure and JIT-compatible. Callers check factors and solutions
for finite values at host boundaries and stop rather than change the method.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


class LeastSquaresFactor(NamedTuple):
    matrix: jax.Array
    projector: jax.Array
    triangular: jax.Array


@partial(jax.jit, static_argnames=("method",))
def factorize(matrix, method="qr"):
    """Factor a tall full-column-rank matrix once for many right-hand sides."""
    if method == "qr":
        q, r = jnp.linalg.qr(matrix, mode="reduced")
        return LeastSquaresFactor(matrix, q.T, r)
    if method == "normal":
        return LeastSquaresFactor(matrix, matrix.T, jnp.linalg.cholesky(matrix.T @ matrix))
    raise ValueError("method must be 'qr' or 'normal'")


@partial(jax.jit, static_argnames=("method",))
def solve_factored(factor, rhs, method="qr"):
    """Solve min_x ||M x - rhs|| using an existing factorization of M."""
    projected = factor.projector @ rhs
    if method == "qr":
        return solve_triangular(factor.triangular, projected, lower=False)
    if method == "normal":
        intermediate = solve_triangular(factor.triangular, projected, lower=True)
        return solve_triangular(factor.triangular.T, intermediate, lower=False)
    raise ValueError("method must be 'qr' or 'normal'")


@jax.jit
def relative_stationarity(matrix, solution, rhs):
    """Scaled gradient norm for min ||M x-rhs||², with Frobenius matrix norm."""
    matrix_norm = jnp.linalg.norm(matrix)
    scale = matrix_norm * (matrix_norm * jnp.linalg.norm(solution) + jnp.linalg.norm(rhs))
    scale = jnp.maximum(scale, jnp.finfo(matrix.dtype).tiny)
    return jnp.linalg.norm(matrix.T @ (matrix @ solution - rhs)) / scale
