"""Short L-domain checks; no full convergence sweep or expensive initialization."""
from pathlib import Path
import json
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from geometry_l import PROBLEM, build_quadrature, initial_condition, exact_solution
from numerics import initial_parameters, values, measure
from initialization import initialize, refine, bulk_error_squared
from integrator import build_matrices, solve_heat
from run import save_npz, load_initial, run_cases
from experiments_l import conditioning_experiment, error_maps
from plotting import convergence_plot


def close(a,b,atol=1e-10):
    np.testing.assert_allclose(a,b,atol=atol,rtol=1e-9)


def main():
    q, validation = build_quadrature(5), build_quadrature(17)
    for rule, bulk_count, boundary_count in ((q,225,70),(validation,3201,262)):
        assert rule['points'].shape == (bulk_count,2)
        assert rule['boundary_points'].shape == (boundary_count,2)
        assert len(np.unique(rule['points'],axis=0)) == bulk_count
        assert np.all(rule['weights'] > 0) and np.all(rule['boundary_weights'] > 0)
        close(rule['weights'].sum(),3*np.pi**2)
        close(rule['boundary_weights'].sum(),8*np.pi)
    assert len(np.unique(q['boundary_points'],axis=0)) == 64
    p, w = np.asarray(q['points']), np.asarray(q['weights'])
    assert not np.any((p[:,0] > 0)&(p[:,1] > 0))
    for a in range(4):
        for b in range(4):
            def moment(lo,hi,k):
                return (hi**(k+1)-lo**(k+1))/(k+1)
            analytic = sum(moment(x0,x1,a)*moment(y0,y1,b) for x0,x1,y0,y1 in
                ((-np.pi,0,-np.pi,0),(-np.pi,0,0,np.pi),(0,np.pi,-np.pi,0)))
            close(np.sum(w*p[:,0]**a*p[:,1]**b),analytic,atol=1e-8)
    # Independent unmerged construction, checking weights on a non-polynomial field.
    axis = np.linspace(0,np.pi,9)
    weights = np.array([1,4,2,4,2,4,2,4,1])*np.pi/24
    direct = 0.
    for ox,oy in ((-np.pi,-np.pi),(-np.pi,0),(0,-np.pi)):
        x,y = np.meshgrid(axis+ox,axis+oy,indexing='ij')
        direct += np.sum(np.outer(weights,weights)*np.exp(.1*x+.2*y))
    close(np.sum(w*np.exp(.1*p[:,0]+.2*p[:,1])),direct)
    boundary = np.asarray(q['boundary_points'])
    x,y = boundary.T
    # No negative-axis internal seams are treated as boundary.
    assert not np.any((np.abs(x)<1e-14)&(y<0)&(y>-np.pi+1e-14))
    assert not np.any((np.abs(y)<1e-14)&(x<0)&(x>-np.pi+1e-14))
    for corner in ((-np.pi,-np.pi),(np.pi,-np.pi),(np.pi,0),(0,0),(0,np.pi),(-np.pi,np.pi)):
        entries = np.all(np.isclose(boundary,corner),axis=1)
        assert entries.sum()==2
        close(np.sum(np.asarray(q['tangents'])[entries],axis=0),[1,1])
    close(initial_condition(q['boundary_points']),np.zeros(70))
    close(jnp.sum(validation['weights']*initial_condition(validation['points'])**2),3*np.pi**2/4)
    lap = jax.vmap(lambda x:jnp.trace(jax.hessian(initial_condition)(x)))(q['points'])
    close(lap,-2*initial_condition(q['points']))
    print('PASS: counts, merged weights, polynomials, boundary topology, eigenfunction',flush=True)

    small, fine = build_quadrature(3), build_quadrature(5)
    theta, history = initialize(small,fine,adam_steps=20,check_every=10,passes=(),
                                target=initial_condition(small['points']),
                                validation_target=initial_condition(fine['points']))
    best_error = min([float(bulk_error_squared(initial_parameters(),fine,
                     initial_condition(fine['points'])))**.5]
                     + [row['bulk_l2'] for row in history['adam']])
    close(float(bulk_error_squared(theta,fine,initial_condition(fine['points'])))**.5,best_error)
    # Instrument actual refinement chunks: all see exactly the SAME forcing.
    import initialization
    original = initialization.refinement_chunk
    seen = []
    def record(*args):
        seen.append(np.array(args[1]))
        return original(*args)
    initialization.refinement_chunk = record
    try:
        refined, _ = refine(theta,small,fine,((11,.5),),initial_condition(small['points']),initial_condition(fine['points']))
    finally:
        initialization.refinement_chunk = original
    assert len(seen)==2
    close(seen[0],seen[1],atol=0)
    close(seen[0],initial_condition(small['points'])-values(theta,small['points']))
    assert float(bulk_error_squared(refined,fine,initial_condition(fine['points']))) <= float(bulk_error_squared(theta,fine,initial_condition(fine['points'])))
    B,C,M = build_matrices(refined,.001,1.,.1,q)
    assert B.shape==(225,153) and C.shape==(140,153) and M.shape==(671,153)
    qr = solve_heat(refined,.002,2,1.,.1,2,small,'qr',progress=False)
    normal = solve_heat(refined,.002,2,1.,.1,2,small,'normal',progress=False)
    close(qr['parameters'],normal['parameters'],atol=2e-8)
    print('PASS: sine initialization, frozen RK4 target, dimensions, QR/normal trajectory',flush=True)

    output = Path(__file__).resolve().parent/'results'/'verification_l'
    output.mkdir(parents=True,exist_ok=True)
    save_npz(output/'initial.npz',theta=np.asarray(refined),problem=PROBLEM,metadata=json.dumps(dict(time=0)))
    close(load_initial(output/'initial.npz',PROBLEM),refined)
    with tempfile.TemporaryDirectory() as directory:
        legacy = Path(directory)/'legacy.npz'
        save_npz(legacy,theta=np.asarray(refined))
        close(load_initial(legacy),refined)
        for path,problem in ((legacy,PROBLEM),(output/'initial.npz','heat-square-cosine-v1')):
            try:
                load_initial(path,problem)
            except ValueError:
                pass
            else:
                raise AssertionError('Cross-domain checkpoint accepted')
    common = dict(problem=PROBLEM,T=.002,alpha=.1,gn_steps=2,solver='qr',quad_points=5,
                  validation_points=9,precision='float64',rule='composite Simpson',
                  decay_rate=2.,initial_norm=float(np.sqrt(3)*np.pi/2),title='L-shaped test')
    rows = run_cases(refined,common,[1.],[1,2],output,quadrature=small,validation=fine,exact_solution=exact_solution)
    stamps = {p:p.stat().st_mtime_ns for p in output.glob('case_*.npz')}
    resumed = run_cases(refined,common,[1.],[2],output,quadrature=small,validation=fine,exact_solution=exact_solution)
    assert len(rows)==len(resumed)==2
    assert stamps=={p:p.stat().st_mtime_ns for p in stamps}
    convergence_plot(rows,output,common,[1.,.1],[1,2],measure(refined,0,fine,.1,exact_solution(0,fine['points']))['bulk_l2'])
    assert 'partial sweep' in (output/'convergence.svg').read_text()
    error_maps(qr['parameters'],qr['times'],output,33)
    with np.load(output/'display_maps.npz') as data:
        assert np.all(np.isnan(data['solutions'][:,data['mask']]))
        assert np.all(np.isfinite(data['solutions'][:,~data['mask']]))
    report = conditioning_experiment(refined,small,fine,output,T=.002,n_steps=2,epsilon=1.,gn_steps=2)
    assert report['qr']['status']==report['normal']['status']=='complete'
    close(report['condition_G'],report['condition_M_squared'],atol=1e-6)
    with np.load(output/'conditioning.npz') as data:
        close(data['caption'],data['B'].T@data['B']+1e6*np.eye(153),atol=1e-8)
    for name in ('convergence','solution_error_maps','conditioning'):
        for suffix in ('pdf','svg','png'):
            assert (output/f'{name}.{suffix}').stat().st_size>1000
    (output/'verification.json').write_text(json.dumps(dict(passed=True,metrics=measure(refined,0,fine,.1,exact_solution(0,fine['points']))),indent=2))
    print('PASS: checkpoints, sweep/resume, masked maps, conditioning, all exports',flush=True)


if __name__ == '__main__':
    main()
