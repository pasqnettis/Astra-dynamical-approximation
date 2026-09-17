"""Bulk-only Adam fit and frozen-forcing artificial-time RK4 refinement."""
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import optax
from numerics import values, value_jacobian, initial_condition, initial_parameters, check_finite


@jax.jit
def bulk_error_squared(theta, q, target=None):
    if target is None:
        target = initial_condition(q['points'])
    return jnp.sum(q['weights'] * (values(theta, q['points'])-target)**2)


def rk4_step(q, step, velocity):
    """Each call to velocity evaluates its Jacobian at that stage's parameters."""
    k1 = velocity(q)
    k2 = velocity(q + step*k1/2)
    k3 = velocity(q + step*k2/2)
    k4 = velocity(q + step*k3)
    return q + step*(k1 + 2*k2 + 2*k3 + k4)/6


@partial(jax.jit, static_argnames=('count',))
def refinement_chunk(theta, correction, epsilon, step, quadrature, count):
    root_weights = jnp.sqrt(quadrature['weights'])
    def velocity(q):
        # Fresh J(q), QR and solve at EVERY stage. No heat/boundary terms.
        J = root_weights[:, None] * value_jacobian(q, quadrature['points'])
        A = jnp.concatenate((J, epsilon*jnp.eye(q.size)))
        rhs = jnp.concatenate((root_weights*correction, jnp.zeros_like(q)))
        Q, R = jnp.linalg.qr(A, mode='reduced')
        return solve_triangular(R, Q.T@rhs)
    def artificial_step(q, unused):
        new = rk4_step(q, step, velocity)
        return new, jnp.all(jnp.isfinite(new))
    return jax.lax.scan(artificial_step, theta, None, length=count)


def refine(theta, quadrature, validation, passes=((100, 1e-4), (200, 1e-5)),
           target=None, validation_target=None):
    target, validation_target = prepare_targets(quadrature, validation, target, validation_target)
    history = []
    for steps, epsilon in passes:
        if type(steps) is not int or steps < 1 or not (0 < epsilon < float('inf')):
            raise ValueError('RK4 passes require positive integer steps and finite epsilon > 0')
        anchor = theta
        before = float(bulk_error_squared(anchor, validation, validation_target))
        # Freeze this ONCE for the ENTIRE pass, including all chunks/stages.
        correction = target - values(anchor, quadrature['points'])
        for start in range(0, steps, 10):
            theta, finite = refinement_chunk(theta, correction, epsilon, 1/steps,
                                              quadrature, min(10, steps-start))
            check_finite(theta)
            if not bool(jnp.all(finite)):
                raise FloatingPointError('Nonfinite RK4 stage result')
        after = float(bulk_error_squared(theta, validation, validation_target))
        check_finite(after)
        accepted = after < before
        if not accepted:
            theta = anchor
        history.append(dict(steps=steps, epsilon=epsilon, accepted=accepted,
                            error_before=before**0.5, error_after=after**0.5))
        print(f'RK4 {steps} steps: error {after**0.5:.6g}, accepted={accepted}', flush=True)
    return theta, history


def initialize(quadrature, validation, seed=0, adam_steps=20000,
               learning_rates=(1e-3, 1e-4), switch_step=10000, check_every=100,
               passes=((100, 1e-4), (200, 1e-5)), target=None, validation_target=None):
    target, validation_target = prepare_targets(quadrature, validation, target, validation_target)
    if type(adam_steps) is not int or adam_steps < 0 or type(check_every) is not int or check_every < 1:
        raise ValueError('Invalid Adam step/check counts')
    if switch_step < 0 or any(not (0 < lr < float('inf')) for lr in learning_rates):
        raise ValueError('Invalid Adam schedule')
    theta = initial_parameters(seed)
    optimizer = optax.adam(lambda step: jnp.where(step < switch_step, *learning_rates))
    state = optimizer.init(theta)
    best, best_loss = theta, float(bulk_error_squared(theta, validation, validation_target))

    @partial(jax.jit, static_argnames=('count',))
    def adam_chunk(theta, state, count):
        def update(carry, unused):
            q, opt_state = carry
            loss, gradient = jax.value_and_grad(bulk_error_squared)(q, quadrature, target)
            updates, opt_state = optimizer.update(gradient, opt_state, q)
            q = optax.apply_updates(q, updates)
            return (q, opt_state), jnp.isfinite(loss) & jnp.all(jnp.isfinite(q))
        return jax.lax.scan(update, (theta, state), None, length=count)

    history = []
    for start in range(0, adam_steps, check_every):
        (theta, state), finite = adam_chunk(theta, state, min(check_every, adam_steps-start))
        if not bool(jnp.all(finite)):
            raise FloatingPointError('Nonfinite Adam computation')
        loss = float(bulk_error_squared(theta, validation, validation_target))
        check_finite(loss)
        if loss < best_loss:
            best, best_loss = theta, loss
        history.append(dict(step=min(start+check_every, adam_steps), bulk_l2=loss**0.5))
    theta, refinement = refine(best, quadrature, validation, passes, target, validation_target)
    return theta, dict(adam=history, refinement=refinement)


def prepare_targets(quadrature, validation, target, validation_target):
    if (target is None) != (validation_target is None):
        raise ValueError('Supply both solver and validation targets')
    if target is None:
        target = initial_condition(quadrature['points'])
        validation_target = initial_condition(validation['points'])
    target, validation_target = jnp.asarray(target), jnp.asarray(validation_target)
    if target.shape != quadrature['weights'].shape or validation_target.shape != validation['weights'].shape:
        raise ValueError('Target shape must match its quadrature weights')
    check_finite(target, validation_target)
    return target, validation_target
