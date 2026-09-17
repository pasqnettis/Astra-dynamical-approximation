"""Equation (4.4): frozen Jacobians, current nonlinear residuals, two penalties."""
from functools import partial
from time import perf_counter
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np
from numerics import (values, laplacians, tangential_values, value_jacobian,
                      bulk_jacobian, tangent_jacobian, check_finite)


@jax.jit
def build_matrices(theta_n, h, epsilon, alpha, quadrature):
    q = quadrature
    # sqrt(weights) converts Euclidean squares into physical quadrature integrals.
    wo = jnp.sqrt(q['weights'])
    wg = jnp.sqrt(q['boundary_weights'])
    B = wo[:, None] * bulk_jacobian(theta_n, q['points'], h)  # (n_bulk,153): square 361; L-shape 225
    # sqrt(alpha), not alpha: the squared tangent norm has coefficient alpha.
    C = jnp.concatenate((
        wg[:, None] * value_jacobian(theta_n, q['boundary_points']),
        jnp.sqrt(alpha) * wg[:, None] * tangent_jacobian(
            theta_n, q['boundary_points'], q['tangents'])))   # (2*n_boundary,153): square 152; L-shape 140
    I = jnp.eye(theta_n.size)
    # Multiply ALL of (4.4) by h²: same minimizer, simpler scaling.
    # Distinct penalties: accumulated displacement AND current increment.
    M = jnp.concatenate((B, C, epsilon/jnp.sqrt(2.)*I, epsilon*I))  # (n_bulk+2*n_boundary+306,153): L-shape 671
    return B, C, M


@jax.jit
def residual_vector(theta, theta_n, previous_values, h, epsilon, alpha, q):
    """b_k in min ||M Delta_theta + b_k||², evaluated at CURRENT theta."""
    return jnp.concatenate((
        jnp.sqrt(q['weights']) * (values(theta, q['points']) - previous_values
                                 - h * laplacians(theta, q['points'])),
        # Boundary residual is u_k, NEVER u_k - u_n.
        jnp.sqrt(q['boundary_weights']) * values(theta, q['boundary_points']),
        jnp.sqrt(alpha) * jnp.sqrt(q['boundary_weights']) * tangential_values(
            theta, q['boundary_points'], q['tangents']),
        epsilon/jnp.sqrt(2.) * (theta - theta_n),
        jnp.zeros_like(theta)))


@partial(jax.jit, static_argnames=('gn_steps', 'solver'))
def heat_step(theta_n, h, epsilon, alpha, gn_steps, quadrature, solver='qr'):
    """Exactly gn_steps updates; all matrices/factors frozen at theta_n."""
    B, C, M = build_matrices(theta_n, h, epsilon, alpha, quadrature)
    previous = values(theta_n, quadrature['points'])
    if solver == 'qr':
        Q, R = jnp.linalg.qr(M, mode='reduced')  # ONCE per physical step
    elif solver == 'normal':
        G = B.T @ B + C.T @ C + 1.5 * epsilon**2 * jnp.eye(theta_n.size)
        L = jnp.linalg.cholesky(G)
    else:
        raise ValueError("solver must be 'qr' or 'normal'")

    def gauss_newton_iteration(theta, unused):
        b = residual_vector(theta, theta_n, previous, h, epsilon, alpha, quadrature)
        if solver == 'qr':
            increment = solve_triangular(R, -Q.T @ b, lower=False)
        else:
            no, nc = B.shape[0], C.shape[0]
            # The accumulated displacement has coefficient 1/2, not 3/2.
            rhs = (-B.T @ b[:no] - C.T @ b[no:no+nc]
                   - 0.5 * epsilon**2 * (theta - theta_n))
            increment = solve_triangular(L.T, solve_triangular(L, rhs, lower=True))
        residual = M @ increment + b
        defect = jnp.linalg.norm(residual) / h
        stationarity = jnp.linalg.norm(M.T @ residual)
        diagnostics = jnp.array([defect, jnp.linalg.norm(increment), stationarity,
                                 h*defect/epsilon**2])
        # This is already Delta_theta: DO NOT multiply by h.
        return theta + increment, diagnostics

    return jax.lax.scan(gauss_newton_iteration, theta_n, None, length=gn_steps)


@partial(jax.jit, static_argnames=('gn_steps', 'solver', 'count'))
def physical_chunk(theta, h, epsilon, alpha, gn_steps, quadrature, solver, count):
    def physical_step(theta_n, unused):
        theta_next, diagnostics = heat_step(theta_n, h, epsilon, alpha,
                                             gn_steps, quadrature, solver)
        return theta_next, (theta_next, diagnostics)
    return jax.lax.scan(physical_step, theta, None, length=count)


def solve_heat(theta0, T, n_steps, epsilon, alpha, gn_steps, quadrature,
               solver='qr', chunk_size=32, progress=True):
    """Host loop around compiled chunks; return trajectory and per-iterate data.

    Columns: defect, increment norm, absolute stationarity, h*defect/epsilon².
    An interrupted trajectory restarts; completed sweep cases persist in run.py.
    """
    if any(type(x) is not int or x < 1 for x in (n_steps, gn_steps, chunk_size)):
        raise ValueError('Step counts must be positive integers')
    if not all(np.isfinite(x) and x > 0 for x in (T, epsilon, alpha)):
        raise ValueError('T, epsilon, alpha must be finite and positive')
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    if theta.shape != (153,):
        raise ValueError('Expected 153 parameters')
    check_finite(theta)
    history, diagnostics = [np.asarray(theta)[None, :]], []
    started = perf_counter()
    for start in range(0, n_steps, chunk_size):
        count = min(chunk_size, n_steps-start)
        theta, (states, data) = physical_chunk(theta, T/n_steps, epsilon, alpha,
                                              gn_steps, quadrature, solver, count)
        check_finite(states, data)
        history.append(np.asarray(states))
        diagnostics.append(np.asarray(data))
        if progress:
            print(f'  {start+count}/{n_steps} steps, {perf_counter()-started:.1f}s', flush=True)
    return dict(parameters=np.concatenate(history), diagnostics=np.concatenate(diagnostics),
                times=np.linspace(0, T, n_steps+1), seconds=perf_counter()-started)
