"""Edit these variables, then: python 'Simple code/run_l.py'."""
from pathlib import Path

MODE = 'convergence'  # initialize, single, convergence, conditioning, plot
T = 1.0
ALPHA = 0.1
EPSILON = 0.001
N_STEPS = 2048
GN_STEPS = 20
EPSILONS = [0.0001]
STEP_COUNTS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
MAIN_NODES = 5                  # per patch side: 5 endpoints + 4 midpoints
VALIDATION_MAIN_NODES = 17      # 33 evaluation points per patch side
SEED = 0
ADAM_STEPS = 20000
LEARNING_RATES = (1e-3, 1e-4)
SWITCH_STEP = 10000
CHECK_EVERY = 100
RK4_PASSES = ((100, 1e-4), (200, 1e-5))
INPUT_CHECKPOINT = None         # must be labelled as an L-shaped initialization
SOLVER = 'qr'                  # 'normal' for comparison; use another OUTPUT
USE_JIT = True
CHUNK_SIZE = 32
RESUME = True
DISPLAY_POINTS = 257
CONDITIONING_STEPS = 2048
CONDITIONING_EPSILON = 0.0001
OUTPUT = Path(__file__).resolve().parent/'results'/'l_shape'/'eps_small'

import json
import importlib.metadata
from time import perf_counter
import numpy as np
import jax
from geometry_l import PROBLEM, build_quadrature, initial_condition, exact_solution, quadrature_description
from initialization import initialize
from numerics import measure
from run import save_npz, load_initial, completed_cases, write_csv, run_cases
from plotting import convergence_plot
from experiments_l import error_maps, conditioning_experiment


def main():
    if MODE not in ('initialize','single','convergence','conditioning','plot'):
        raise ValueError('Unknown MODE')
    jax.config.update('jax_disable_jit',not USE_JIT)
    output = Path(OUTPUT)
    output.mkdir(parents=True,exist_ok=True)
    q, validation = build_quadrature(MAIN_NODES), build_quadrature(VALIDATION_MAIN_NODES)
    common = dict(problem=PROBLEM,T=T,alpha=ALPHA,gn_steps=GN_STEPS,solver=SOLVER,
                  quad_points=2*MAIN_NODES-1,validation_points=2*VALIDATION_MAIN_NODES-1,
                  main_nodes=MAIN_NODES,validation_main_nodes=VALIDATION_MAIN_NODES,
                  precision='float64',rule='composite Simpson',decay_rate=2.,initial_norm=float(np.sqrt(3)*np.pi/2),
                  title='L-shaped heat equation convergence',
                  quadrature_description=quadrature_description(MAIN_NODES,VALIDATION_MAIN_NODES))
    local = output/'initial.npz'
    history = None
    if MODE == 'plot':
        theta0 = load_initial(local,PROBLEM)
    elif INPUT_CHECKPOINT is not None:
        theta0 = load_initial(Path(INPUT_CHECKPOINT),PROBLEM)
    elif RESUME and local.exists():
        theta0 = load_initial(local,PROBLEM)
    else:
        started = perf_counter()
        theta0, history = initialize(q,validation,SEED,ADAM_STEPS,LEARNING_RATES,
            SWITCH_STEP,CHECK_EVERY,RK4_PASSES,target=initial_condition(q['points']),
            validation_target=initial_condition(validation['points']))
        history['seconds'] = perf_counter()-started
    rows = completed_cases(output,common,theta0)  # Check BEFORE replacing artifacts.
    metrics0 = measure(theta0,0.,validation,ALPHA,exact_solution(0.,validation['points']))
    if MODE != 'plot':
        save_npz(local,theta=np.asarray(theta0),problem=PROBLEM,format_version=1,
                 metadata=json.dumps(dict(time=0.,metrics=metrics0)))
        if history is not None:
            (output/'initialization.json').write_text(json.dumps(history,indent=2))
        metadata = dict(common, mode=MODE, epsilon=EPSILON,n_steps=N_STEPS,epsilons=EPSILONS,
            step_counts=STEP_COUNTS,seed=SEED,adam_steps=ADAM_STEPS,learning_rates=LEARNING_RATES,
            switch_step=SWITCH_STEP,check_every=CHECK_EVERY,rk4_passes=RK4_PASSES,
            input_checkpoint=str(INPUT_CHECKPOINT),use_jit=USE_JIT,chunk_size=CHUNK_SIZE,
            display_points=DISPLAY_POINTS,conditioning_steps=CONDITIONING_STEPS,
            conditioning_epsilon=CONDITIONING_EPSILON,initial_metrics=metrics0,
            device=str(jax.devices()[0]),versions={name:importlib.metadata.version(name)
                for name in ('jax','jaxlib','numpy','optax','matplotlib')})
        (output/'settings.json').write_text(json.dumps(metadata,indent=2))
    if MODE == 'initialize':
        print(metrics0)
    elif MODE == 'conditioning':
        report = conditioning_experiment(theta0,q,validation,output,T,CONDITIONING_STEPS,
                                         CONDITIONING_EPSILON,ALPHA,GN_STEPS)
        print(json.dumps(report,indent=2))
    elif MODE == 'plot':
        write_csv(output,rows)
        convergence_plot(rows,output,common,EPSILONS,STEP_COUNTS,metrics0['bulk_l2'])
    else:
        run_cases(theta0,common,[EPSILON] if MODE == 'single' else EPSILONS,
                  [N_STEPS] if MODE == 'single' else STEP_COUNTS,output,RESUME,CHUNK_SIZE,
                  quadrature=q,validation=validation,exact_solution=exact_solution)
    # Plotting mode also regenerates Figure 5.4 if its selected case exists.
    if MODE in ('single','plot'):
        path = output/f'case_eps_{EPSILON!r}_N_{N_STEPS}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                error_maps(data['parameters'],data['times'],output,DISPLAY_POINTS)
        elif MODE == 'plot':
            print(f'No saved map case for ε={EPSILON}, N={N_STEPS}; convergence plot only.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Completed cases remain saved; rerun to resume.')
