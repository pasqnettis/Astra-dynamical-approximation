"""L-shaped display maps and first-step conditioning measurements."""
import json
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from numerics import values, measure, check_finite
from geometry_l import exact_solution
from integrator import build_matrices, residual_vector, heat_step
from run import save_npz
from plotting import plt
from matplotlib.patches import Polygon


def export_figure(fig, output, name):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(output/f'{name}.{extension}', dpi=600, bbox_inches='tight')
    plt.close(fig)


def error_maps(parameters, times, output, display_points=257):
    """Display grid only: never use these samples to measure L² errors."""
    if type(display_points) is not int or display_points < 3:
        raise ValueError('display_points must be an integer >= 3')
    axis = np.linspace(-np.pi, np.pi, display_points)
    x, y = np.meshgrid(axis, axis, indexing='xy')
    points = jnp.asarray(np.stack((x.ravel(), y.ravel()), axis=1))
    mask = (x > 0) & (y > 0)
    solutions, errors = [], []
    for theta, time in zip((parameters[0], parameters[-1]), (times[0], times[-1])):
        prediction = np.asarray(values(theta, points)).reshape(x.shape)
        exact = np.asarray(exact_solution(time, points)).reshape(x.shape)
        check_finite(prediction)
        solutions.append(np.where(mask, np.nan, prediction))
        errors.append(np.where(mask, np.nan, np.abs(prediction-exact)))
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for column, time in enumerate((times[0], times[-1])):
        for row, (data, title, cmap) in enumerate(((solutions, 'Solution', 'RdBu_r'),
                                                  (errors, 'Absolute error', 'viridis'))):
            ax = axes[row,column]
            options = {}
            if row == 0:
                limit = max(float(np.nanmax(np.abs(data[column]))), 1e-15)
                options = dict(vmin=-limit, vmax=limit)
            else:
                options = dict(vmin=0)
            artist = ax.pcolormesh(x, y, np.ma.masked_invalid(data[column]),
                                   shading='auto', cmap=cmap, rasterized=True, **options)
            # Clip the half-pixel cells at the exact polygon boundary as well.
            outline = Polygon([(-np.pi,-np.pi),(np.pi,-np.pi),(np.pi,0),
                               (0,0),(0,np.pi),(-np.pi,np.pi)], transform=ax.transData)
            artist.set_clip_path(outline)
            ax.set(title=f'{title}, t={time:g}', xlabel='$x_1$', ylabel='$x_2$', aspect='equal')
            ax.set_xlim(-np.pi,np.pi)
            ax.set_ylim(-np.pi,np.pi)
            fig.colorbar(artist, ax=ax, label='$u$' if row == 0 else '$|u-y|$')
    fig.suptitle('L-shaped heat equation')
    export_figure(fig, output, 'solution_error_maps')
    save_npz(output/'display_maps.npz', axis=axis, mask=mask,
             solutions=np.array(solutions), absolute_errors=np.array(errors),
             times=np.array([times[0],times[-1]]))


def conditioning_experiment(theta0, q, validation, output, T=1., n_steps=2048,
                            epsilon=1e-4, alpha=.1, gn_steps=20):
    if type(n_steps) is not int or n_steps < 1 or type(gn_steps) is not int or gn_steps < 1:
        raise ValueError('Step counts must be positive integers')
    if any(not np.isfinite(v) or v <= 0 for v in (T,epsilon,alpha)):
        raise ValueError('T, epsilon, alpha must be positive and finite')
    h = T/n_steps
    B, C, M = build_matrices(theta0,h,epsilon,alpha,q)
    G = M.T@M  # Actual equation-(4.4) normal matrix, scaled by h².
    caption = B.T@B + epsilon**2/h**2*jnp.eye(theta0.size)
    check_finite(B,C,M,G,caption)
    sm = np.linalg.svd(np.asarray(M), compute_uv=False)
    sg = np.linalg.svd(np.asarray(G), compute_uv=False)
    sc = np.linalg.svd(np.asarray(caption), compute_uv=False)
    eigen_g = np.linalg.eigvalsh(np.asarray(G))
    eigen_caption = np.linalg.eigvalsh(np.asarray(caption))
    def condition(s):
        return float(s[0]/s[-1]) if s[-1] > 0 else None
    report = dict(h=h, n_steps=n_steps, epsilon=epsilon, alpha=alpha, gn_steps=gn_steps,
                  condition_M=condition(sm), condition_G=condition(sg),
                  condition_M_squared=condition(sm)**2 if condition(sm) is not None else None,
                  condition_caption=condition(sc),
                  min_eigenvalue_G=float(eigen_g[0]),
                  matrix_definitions=dict(M='[B; C; epsilon/sqrt(2) I; epsilon I]',
                      G='M.T @ M (actual normal matrix, h^2-scaled)',
                      caption='B.T @ B + epsilon^2/h^2 I (literal Figure 5.5 caption; bulk only)'))
    previous = values(theta0,q['points'])
    b = residual_vector(theta0,theta0,previous,h,epsilon,alpha,q)
    increments, endpoints = {}, {}
    for solver in ('qr','normal'):
        try:
            if solver == 'qr':
                Q,R = jnp.linalg.qr(M,mode='reduced')
                increment = solve_triangular(R,-Q.T@b)
            else:
                # Same explicit normal matrix as the time integrator.
                normal = B.T@B+C.T@C+1.5*epsilon**2*jnp.eye(theta0.size)
                L = jnp.linalg.cholesky(normal)
                increment = solve_triangular(L.T,solve_triangular(L,-M.T@b,lower=True))
            check_finite(increment)
            endpoint, diagnostics = heat_step(theta0,h,epsilon,alpha,gn_steps,q,solver)
            check_finite(endpoint,diagnostics)
            increments[solver], endpoints[solver] = np.asarray(increment), np.asarray(endpoint)
            report[solver] = dict(status='complete', first_increment_norm=float(jnp.linalg.norm(increment)),
                first_increment_stationarity=float(jnp.linalg.norm(M.T@(M@increment+b))),
                max_stationarity=float(diagnostics[:,2].max()),
                **measure(endpoint,h,validation,alpha,exact_solution(h,validation['points'])))
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            report[solver] = dict(status='failed', reason=str(error))
    if len(endpoints) == 2:
        field_difference = values(endpoints['qr'],validation['points'])-values(endpoints['normal'],validation['points'])
        report['comparison'] = dict(increment_difference=float(np.linalg.norm(increments['qr']-increments['normal'])),
            endpoint_parameter_difference=float(np.linalg.norm(endpoints['qr']-endpoints['normal'])),
            endpoint_field_l2=float(jnp.sqrt(jnp.sum(validation['weights']*field_difference**2))))
    # A singular matrix has condition None, not an invalid JSON Infinity.
    (output/'conditioning.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    save_npz(output/'conditioning.npz', theta0=np.asarray(theta0), B=np.asarray(B), C=np.asarray(C),
             M=np.asarray(M), G=np.asarray(G), caption=np.asarray(caption),
             singular_M=sm, singular_G=sg, singular_caption=sc,
             eigen_G=eigen_g, eigen_caption=eigen_caption,
             **{f'increment_{k}':v for k,v in increments.items()},
             **{f'endpoint_{k}':v for k,v in endpoints.items()})
    np.savetxt(output/'conditioning.csv', np.column_stack((np.arange(153),sm,sg,sc,eigen_g,eigen_caption)),
               delimiter=',',header='index,singular_M,singular_G,singular_caption,eigen_G,eigen_caption',comments='')
    fig, axes = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax, spectrum, label, cond in zip(axes,(sm,sg,sc),
            ('Augmented M','Actual normal G = MᵀM','Figure 5.5 caption (bulk only)'),
            (condition(sm),condition(sg),condition(sc))):
        ax.semilogy(np.sort(spectrum),'.',markersize=3)
        ax.set(xlabel='Sorted index',ylabel='Singular value',title=label+'\nκ='+('singular' if cond is None else f'{cond:.3g}'))
        ax.grid(alpha=.2)
    fig.suptitle(f'First-step conditioning: ε={epsilon:g}, N={n_steps}, α={alpha:g}, float64')
    export_figure(fig,output,'conditioning')
    return report
