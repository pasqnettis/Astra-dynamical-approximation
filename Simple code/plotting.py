"""Publication exports; all plotting remains outside JAX."""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'rdpa-simple-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', os.path.join(tempfile.gettempdir(), 'rdpa-simple-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def convergence_plot(rows, output, settings, epsilons, steps, initial_error):
    if not rows:
        return
    selected = [r for r in rows if r['epsilon'] in epsilons and r['n_steps'] in steps]
    if not selected:
        return
    partial = len(selected) < len(set(epsilons))*len(set(steps))
    with plt.rc_context({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 11, 'axes.labelsize': 12, 'savefig.bbox': 'tight'}):
        fig, ax = plt.subplots(figsize=(7.2, 5.3))
        for epsilon, color, marker in zip(epsilons,
                ['#0072B2', '#D55E00', '#009E73', '#CC79A7']*10, ['o', 's', '^', 'D']*10):
            group = sorted((r for r in selected if r['epsilon'] == epsilon), key=lambda r:r['h'])
            if group:
                ax.loglog([r['h'] for r in group], [r['bulk_l2'] for r in group],
                          color=color, marker=marker, markersize=4, label=fr'$\varepsilon={epsilon:g}$')
        h = np.array(sorted(set(r['h'] for r in selected)))
        rate = settings.get('decay_rate', .5)
        norm0 = settings.get('initial_norm', np.pi)
        ax.loglog(h, norm0*np.exp(-rate*settings['T'])*settings['T']*rate**2/2*h,
                  'k--', linewidth=1, label=r'$O(h)$')
        ax.axhline(initial_error, color='0.5', linestyle=':', label='Initial approximation error')
        ax.set(xlabel=r'Time step $h=T/N$', ylabel=r'$\|u_N-y(T)\|_{L^2(\Omega)}$',
               title=settings.get('title', 'Heat equation convergence') + (' — partial sweep' if partial else ''))
        ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=9)
        n, v = settings['quad_points'], settings['validation_points']
        annotation = settings.get('quadrature_description',
            f'Composite Simpson: {n}×{n} bulk, {n}/edge; validation: {v}×{v}, {v}/edge')
        fig.text(.5, .015, f"α={settings['alpha']:g}; K={settings['gn_steps']}; float64; {settings['solver']}; T={settings['T']:g}\n"
                 + annotation,
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .14, 1, 1))
        # Same filenames are refreshed: title identifies partial exports.
        for extension in ('pdf', 'svg', 'png'):
            fig.savefig(output / f'convergence.{extension}', dpi=600)
        plt.close(fig)
