"""Short numerical comparisons. The ONLY file importing the original package."""
from pathlib import Path
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from scipy.optimize import brentq
import numerics as simple
from integrator import build_matrices, residual_vector, heat_step, solve_heat
from initialization import rk4_step, refinement_chunk, initialize, refine
from run import load_initial, run_cases

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rdpa_heat import model, operators
from rdpa_heat.quadrature import build_quadrature as original_quadrature
from rdpa_heat.integrator import assemble_step, advance_step, modified_rhs
from rdpa_heat.config import RunConfig
from rdpa_heat.diagnostics import evaluate_state


def close(a, b, atol=1e-10, rtol=1e-9):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def main():
    theta = simple.initial_parameters(0)
    close(theta, model.initialize_parameters(0), atol=0, rtol=0)
    q, oldq = simple.build_quadrature(5), original_quadrature(5)
    for key in q:
        close(q[key], getattr(oldq, key), atol=0, rtol=0)
    close(q['weights'].sum(), 4*np.pi**2)
    close(q['boundary_weights'].sum(), 8*np.pi)
    for name, args in [('values',(q['points'],)), ('laplacians',(q['points'],)),
                       ('value_jacobian',(q['points'],)), ('bulk_jacobian',(q['points'], .02)),
                       ('tangential_values',(q['boundary_points'],q['tangents'])),
                       ('tangent_jacobian',(q['boundary_points'],q['tangents']))]:
        close(getattr(simple,name)(theta,*args), getattr(operators,name)(theta,*args))
    h, eps, alpha = .02, .1, 1.
    B,C,M = build_matrices(theta,h,eps,alpha,q)
    system = assemble_step(theta,h,eps,alpha,oldq)
    close(B,system.bulk_matrix); close(C,system.boundary_matrix); close(M,system.factor.matrix)
    close(M.T@M, B.T@B+C.T@C+1.5*eps**2*jnp.eye(153))
    current = theta+jnp.linspace(-.001,.001,153)
    b = residual_vector(current,theta,simple.values(theta,q['points']),h,eps,alpha,q)
    no, nc = B.shape[0], C.shape[0]
    close(b,modified_rhs(b[:no],b[no:no+nc],current-theta,eps))
    Q,R = jnp.linalg.qr(M,mode='reduced')
    increment = solve_triangular(R,-Q.T@b)
    close((M.T@M)@increment, -B.T@b[:no]-C.T@b[no:no+nc]-.5*eps**2*(current-theta))
    def objective(v):
        return (jnp.sum((B@v+b[:no])**2)+jnp.sum((C@v+b[no:no+nc])**2)
                +.5*eps**2*jnp.sum((v+current-theta)**2)+eps**2*jnp.sum(v**2))
    close(jax.grad(objective)(increment), jnp.zeros(153), atol=1e-9)
    config = RunConfig(steps=1, final_time=h, epsilon=eps, alpha=alpha, iterations=1,
                       quadrature_points=5)
    one, _ = heat_step(theta,h,eps,alpha,1,q,'qr')
    close(one,advance_step(theta,system,config).parameters)
    print('PASS: parameter ordering, derivatives, quadrature, matrices, residuals, both coefficients', flush=True)

    checkpoint = ROOT/'runs/default_initialization/initial.npz'
    theta0 = load_initial(checkpoint) if checkpoint.exists() else theta
    # Finite fitted initialization keeps this short trajectory well conditioned.
    result = solve_heat(theta0,.02,2,eps,alpha,3,q,'qr',chunk_size=2,progress=False)
    normal = solve_heat(theta0,.02,2,eps,alpha,3,q,'normal',chunk_size=2,progress=False)
    close(result['parameters'],normal['parameters'],atol=2e-9)
    old = theta0
    config = RunConfig(steps=2,final_time=.02,epsilon=eps,alpha=alpha,iterations=3,quadrature_points=5)
    for index in range(2):
        old = advance_step(old,assemble_step(old,.01,eps,alpha,oldq),config).parameters
        close(old,result['parameters'][index+1])
    validation = simple.build_quadrature(9)
    metrics = simple.measure(old,.02,validation,alpha)
    original = evaluate_state(old,.02,original_quadrature(9),alpha)
    for key in metrics:
        close(metrics[key],original[key])
    print('PASS: checkpoint, short trajectories, QR/normal agreement, independent errors', flush=True)

    # Default matrix dimensions and full 20-update physical step.
    baseline_q = simple.build_quadrature(19)
    baseline_B, baseline_C, baseline_M = build_matrices(theta0,h,eps,alpha,baseline_q)
    assert baseline_B.shape == (361,153) and baseline_C.shape == (152,153)
    assert baseline_M.shape == (819,153)
    baseline_theta, baseline_data = heat_step(theta0,h,eps,alpha,20,baseline_q,'qr')
    baseline_system = assemble_step(theta0,h,eps,alpha,original_quadrature(19))
    baseline_config = RunConfig(steps=1,final_time=h,epsilon=eps,alpha=alpha,iterations=20)
    close(baseline_theta,advance_step(theta0,baseline_system,baseline_config).parameters)
    assert baseline_data.shape == (20,4)
    print('PASS: default 19-point matrices and exactly 20 Gauss–Newton updates', flush=True)

    # Scalar nonlinear Phi(q)=q², fixed forcing: independent implicit exact endpoint.
    anchor, forcing, regularization = 1., .7, .2
    def velocity(q):
        return 2*q*forcing/(4*q*q+regularization**2)
    endpoint = brentq(lambda q:q*q-anchor**2+regularization**2/2*np.log(q/anchor)-forcing,1.,2.,xtol=1e-14)
    errors = []
    for n in (4,8,16):
        x = anchor
        for _ in range(n):
            x = rk4_step(x,1/n,velocity)
        errors.append(abs(x-endpoint))
    assert errors[0]/errors[1] > 12 and errors[1]/errors[2] > 12, errors
    close(rk4_step(jnp.array([1.,2.]),1.,lambda q:jnp.array([.2,.3])),[1.2,2.3])
    # Compare actual compiled refinement against explicit stagewise solves, frozen c.
    correction = simple.initial_condition(q['points'])-simple.values(theta0,q['points'])
    def network_velocity(t):
        A = jnp.concatenate((jnp.sqrt(q['weights'])[:,None]*simple.value_jacobian(t,q['points']),
                             .1*jnp.eye(153)))
        Q,R = jnp.linalg.qr(A,mode='reduced')
        return solve_triangular(R,Q.T@jnp.concatenate((jnp.sqrt(q['weights'])*correction,jnp.zeros(153))))
    expected = theta0
    for _ in range(2):
        expected = rk4_step(expected,.5,network_velocity)
    refined, finite = refinement_chunk(theta0,correction,.1,.5,q,2)
    close(refined,expected); assert np.all(finite)
    # Zero steps is an explicit supported way to skip Adam; exercise Adam separately.
    fitted, history = initialize(q,validation,adam_steps=2,check_every=1,passes=())
    assert len(history['adam']) == 2 and fitted.shape == (153,)
    # Force a finite bad candidate to verify rejection retains the anchor.
    import initialization
    saved = initialization.refinement_chunk
    try:
        initialization.refinement_chunk = lambda *args: (jnp.zeros(153),jnp.array([True]))
        retained, records = refine(theta0,q,validation,((1,.1),))
        close(retained,theta0,atol=0,rtol=0); assert not records[0]['accepted']
    finally:
        initialization.refinement_chunk = saved
    print('PASS: Adam, stagewise frozen-target RK4, fourth order, pass rejection', flush=True)

    common = dict(T=.02,alpha=1.,gn_steps=2,solver='qr',quad_points=5,validation_points=9,
                  precision='float64',rule='composite Simpson')
    # Persistent small report, separate from all publication results.
    output = Path(__file__).resolve().parent/'results'/'verification'
    rows = run_cases(theta0,common,[.1],[1,2],output,chunk_size=2)
    archives = {p:p.stat().st_mtime_ns for p in output.glob('case_*.npz')}
    resumed = run_cases(theta0,common,[.1],[2],output,chunk_size=2)
    assert len(resumed)==len(rows)==2
    assert archives == {p:p.stat().st_mtime_ns for p in archives}
    from plotting import convergence_plot
    convergence_plot(rows,output,common,[.1,.01],[1,2],simple.measure(theta0,0,validation)['bulk_l2'])
    for ext in ('pdf','svg','png'):
        assert (output/f'convergence.{ext}').stat().st_size > 1000
    assert 'partial sweep' in (output/'convergence.svg').read_text()
    try:
        run_cases(theta0,dict(common,alpha=.5),[.1],[1],output)
    except ValueError as error:
        assert 'different OUTPUT' in str(error)
    else:
        raise AssertionError('Resume mismatch was not rejected')
    (output/'verification.json').write_text(json.dumps(dict(passed=True,rk4_errors=errors,
                                            checkpoint=str(checkpoint),metrics=metrics),indent=2))
    print('PASS: convergence smoke test, resume/row preservation, mismatch rejection, partial exports', flush=True)


if __name__ == '__main__':
    main()
