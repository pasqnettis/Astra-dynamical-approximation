"""Edit the variables below, then: python 'Simple code/run.py'."""
from pathlib import Path

# ------------------------ EDIT THESE PARAMETERS ------------------------
MODE = 'convergence'                 # 'initialize', 'single', 'convergence', 'plot'
T = 1.0
ALPHA = 1.0
EPSILON = 0.001                      # single-run epsilon
N_STEPS = 256                       # single-run physical steps
GN_STEPS = 20
EPSILONS = [0.1, 0.01, 0.001, 0.0001]  # execution follows this order
STEP_COUNTS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
QUAD_POINTS = 19                     # composite Simpson, per axis AND per edge
VALIDATION_POINTS = 65               # independent error quadrature
SEED = 0
ADAM_STEPS = 20000
LEARNING_RATES = (1e-3, 1e-4)
SWITCH_STEP = 10000
CHECK_EVERY = 100
RK4_PASSES = ((100, 1e-4), (200, 1e-5))
INPUT_CHECKPOINT = None              # e.g. 'runs/default_initialization/initial.npz'
SOLVER = 'qr'                        # 'qr' or 'normal'
USE_JIT = True
CHUNK_SIZE = 32                      # physical steps between host checks
RESUME = True
OUTPUT = Path(__file__).resolve().parent / 'results'
# ----------------------------------------------------------------------

import csv
import json
import importlib.metadata
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
from numerics import build_quadrature, measure, check_finite
from initialization import initialize
from integrator import solve_heat
from plotting import convergence_plot


def save_npz(path, **arrays):
    """Replace only after writing the complete archive (also safe on Ctrl-C)."""
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_initial(path, problem='heat-square-cosine-v1'):
    with np.load(path, allow_pickle=False) as data:
        theta = data['theta'].copy()
        # Legacy unlabelled archives are square checkpoints only.
        stored_problem = str(data['problem']) if 'problem' in data else 'heat-square-cosine-v1'
        if stored_problem != problem:
            raise ValueError('Checkpoint belongs to a different problem')
        if 'format_version' in data and int(data['format_version']) != 1:
            raise ValueError('Unsupported checkpoint format')
        if 'parameter_layout' in data:
            layout = json.loads(str(data['parameter_layout']))
            expected = [(0,[2]), (2,[6,2]), (14,[6]), (20,[6,6]), (56,[6]),
                        (62,[6,6]), (98,[6]), (104,[6,6]), (140,[6]), (146,[6]), (152,[])]
            if [(p['offset'], p['shape']) for p in layout] != expected:
                raise ValueError('Incompatible checkpoint parameter ordering')
        if 'metadata' in data:
            metadata = json.loads(str(data['metadata']))
            if metadata.get('time', 0) != 0 or metadata.get('kind') == 'final':
                raise ValueError('Expected an initialization checkpoint, not a final state')
    if theta.shape != (153,) or theta.dtype != np.float64:
        raise ValueError('Checkpoint must contain 153 float64 parameters in documented order')
    check_finite(theta)
    return jnp.asarray(theta)


def completed_cases(output, common, theta0):
    """Read ALL completed cases so extending/reordering a sweep preserves CSV rows."""
    rows = []
    for path in sorted(output.glob('case_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            settings = json.loads(str(data['settings']))
            settings.setdefault('problem', 'heat-square-cosine-v1')
            if any(settings.get(k) != v for k, v in common.items()) or not np.array_equal(data['theta0'], theta0):
                raise ValueError(f'{path.name}: settings or initial parameters differ; choose a different OUTPUT directory')
            rows.append(json.loads(str(data['metrics'])))
    return rows


def write_csv(output, rows):
    temporary = output / 'convergence.tmp'
    with temporary.open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (-r['epsilon'], r['n_steps'])))
    temporary.replace(output / 'convergence.csv')


def run_cases(theta0, common, epsilons, step_counts, output, resume=True, chunk_size=32,
              quadrature=None, validation=None, exact_solution=None):
    """One shared initialization; save each completed trajectory independently."""
    if not epsilons or not step_counts:
        raise ValueError('Provide at least one epsilon and step count')
    if any(not np.isfinite(e) or e <= 0 for e in epsilons):
        raise ValueError('All epsilon values must be finite and positive')
    if any(type(n) is not int or n < 1 for n in step_counts):
        raise ValueError('All step counts must be positive integers')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    q = build_quadrature(common['quad_points']) if quadrature is None else quadrature
    if validation is None:
        validation = build_quadrature(common['validation_points'])
    def exact(time):
        return None if exact_solution is None else exact_solution(time, validation['points'])
    rows = completed_cases(output, common, theta0)
    initial_error = measure(theta0, 0., validation, common['alpha'], exact(0.))['bulk_l2']
    try:
        for epsilon in epsilons:
            for steps in step_counts:
                if resume and any(r['epsilon'] == epsilon and r['n_steps'] == steps for r in rows):
                    print(f'Skipping completed ε={epsilon:g}, N={steps}', flush=True)
                    continue
                print(f'Running ε={epsilon:g}, N={steps}', flush=True)
                result = solve_heat(theta0, common['T'], steps, epsilon, common['alpha'],
                                    common['gn_steps'], q, common['solver'], chunk_size)
                row = dict(epsilon=epsilon, n_steps=steps, h=common['T']/steps,
                           **measure(result['parameters'][-1], common['T'], validation, common['alpha'], exact(common['T'])),
                           seconds=result['seconds'], last_defect=float(result['diagnostics'][-1,-1,0]),
                           max_stationarity=float(result['diagnostics'][:,:,2].max()))
                settings = dict(common, epsilon=epsilon, n_steps=steps)
                settings.setdefault('problem', 'heat-square-cosine-v1')
                # repr retains distinct floating-point epsilon values in filenames.
                save_npz(output / f'case_eps_{epsilon!r}_N_{steps}.npz',
                         theta0=np.asarray(theta0), **result,
                         settings=json.dumps(settings), metrics=json.dumps(row, allow_nan=False))
                rows = [r for r in rows if (r['epsilon'],r['n_steps']) != (epsilon,steps)] + [row]
                write_csv(output, rows)
    finally:
        # Includes KeyboardInterrupt: completed cases and a partial figure survive.
        write_csv(output, rows)
        convergence_plot(rows, output, common, epsilons, step_counts, initial_error)
    return rows


def main():
    if MODE not in ('initialize', 'single', 'convergence', 'plot'):
        raise ValueError('Unknown MODE')
    jax.config.update('jax_disable_jit', not USE_JIT)
    output = Path(OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    common = dict(T=T, alpha=ALPHA, gn_steps=GN_STEPS, solver=SOLVER,
                  quad_points=QUAD_POINTS, validation_points=VALIDATION_POINTS,
                  precision='float64', rule='composite Simpson', problem='heat-square-cosine-v1')
    q, validation = build_quadrature(QUAD_POINTS), build_quadrature(VALIDATION_POINTS)
    local = output / 'initial.npz'
    if MODE == 'plot':
        theta0 = load_initial(local)
    elif INPUT_CHECKPOINT is not None:
        theta0 = load_initial(Path(INPUT_CHECKPOINT))
    elif RESUME and local.exists():
        theta0 = load_initial(local)
    else:
        started = perf_counter()
        theta0, history = initialize(q, validation, SEED, ADAM_STEPS, LEARNING_RATES,
                                    SWITCH_STEP, CHECK_EVERY, RK4_PASSES)
        history['seconds'] = perf_counter()-started
        (output/'initialization.json').write_text(json.dumps(history, indent=2))
    # Validate existing records BEFORE replacing the shared initialization.
    rows = completed_cases(output, common, theta0)
    if MODE != 'plot':
        save_npz(local, theta=np.asarray(theta0), problem='heat-square-cosine-v1', metadata=json.dumps(dict(time=0)),
                 layout=np.array('input2,W1(6,2),b1(6),W2/b2,W3/b3,W4/b4,wout6,bout1'))
        versions = {name: importlib.metadata.version(name) for name in
                    ('jax', 'jaxlib', 'optax', 'numpy', 'matplotlib')}
        (output/'settings.json').write_text(json.dumps(dict(common, mode=MODE,
            epsilons=EPSILONS, step_counts=STEP_COUNTS, epsilon=EPSILON, n_steps=N_STEPS,
            seed=SEED, adam_steps=ADAM_STEPS, learning_rates=LEARNING_RATES,
            switch_step=SWITCH_STEP, check_every=CHECK_EVERY, rk4_passes=RK4_PASSES,
            input_checkpoint=str(INPUT_CHECKPOINT), versions=versions,
            device=str(jax.devices()[0]), use_jit=USE_JIT), indent=2))
    if MODE == 'initialize':
        print(measure(theta0, 0., validation, ALPHA))
    elif MODE == 'plot':
        write_csv(output, rows)
        convergence_plot(rows, output, common, EPSILONS, STEP_COUNTS,
                         measure(theta0, 0., validation, ALPHA)['bulk_l2'])
    else:
        run_cases(theta0, common, [EPSILON] if MODE == 'single' else EPSILONS,
                  [N_STEPS] if MODE == 'single' else STEP_COUNTS, output, RESUME, CHUNK_SIZE)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Completed cases are saved; rerun to resume.')
