# Regularized parametric heat solver

JAX implementation of equation **(4.4)** in *Regularized dynamical parametric approximation of boundary value problems*. It evolves a 153-parameter neural representation of the heat equation on `[-pi, pi]^2` with zero Dirichlet data and initial condition `cos(x1/2) cos(x2/2)`.

The complete mathematical specification and file structure are in [project.md](project.md). [PLAN.md](PLAN.md) preserves the original plan.

The specification audit, resolved findings, and measured initialization/time-convergence results are in [AUDIT.md](AUDIT.md).

## Setup and first run

Use Python 3.11 or newer. The package pins JAX/jaxlib 0.4.38 and Optax 0.2.4, matching the tested CPU environment. Double precision is enabled when importing `rdpa_heat`.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m rdpa_heat initialize --config configs/baseline.toml --output runs/initial
python -m rdpa_heat run --config configs/baseline.toml --checkpoint runs/initial/initial.npz --output runs/baseline
```

With dependencies already installed, editable installation is optional: prefix the commands with `PYTHONPATH=src` when running from this directory. `pytest` is already configured to find `src`.

For a small wiring check (not an accuracy experiment):

```bash
PYTHONPATH=src python -m rdpa_heat run --adam-steps 100 --init-pass 10:0.01 --steps 4 --iterations 3 --quadrature-points 5 --validation-points 9 --output runs/smoke
```

Choose a new or empty output directory for every run. Existing results are never silently overwritten. `run` computes initialization automatically if `--checkpoint` is omitted. Use an `initial.npz` checkpoint, not a final physical-time state.

## Method and conventions

- Four width-six tanh layers, scalar affine output, and trainable input translation give 153 parameters. The redundant input translation is intentionally preserved.
- Physical Simpson quadrature uses 19 points per axis (361 bulk samples) and 19 points on each of four edges. The validation rule defaults to 65 points per axis/edge.
- The boundary norm is `integral(u²) + alpha * integral((tangential derivative u)²)`, with positive user input `alpha` and default `0.2`. This quadratic convention is explicit: it is not the square of the sum of norms printed on page 15.
- At every physical time step, Jacobians and a QR factorization are assembled once at the previous parameters, then reused for exactly 20 modified Gauss–Newton iterations by default. Nonlinear residuals are updated every iteration.
- The two printed regularization terms in (4.4) give `1.5 * epsilon² * I` in the normal matrix and `-0.5 * epsilon² * (theta_k - theta_n)` on the right-hand side. The unnumbered matrix remark in the manuscript has a different coefficient; this implementation follows (4.4).
- The solved quantity is the parameter increment, not a velocity. The boundary residual is the new solution itself, not the change from the previous boundary trace.
- QR solves the augmented system without squaring its condition number. `--linear-solver normal` selects the same objective via Cholesky normal equations for comparison. Neither backend adds damping or changes regularization on failure.

### Initial-condition computation

Adam first fits the interior target using integral weights. The best independently validated checkpoint is retained. Two artificial-time RK4 refinement passes then use `(steps, epsilon) = (100, 1e-4), (200, 1e-5)`.

Each pass freezes `correction = y0 - Phi(theta_anchor)` over its entire artificial-time interval `[0,1]`. Every RK4 stage recomputes the parameter Jacobian at that stage's parameters and solves an interior-only ridge least-squares problem. Initialization contains no heat Laplacian, physical step size, or boundary penalty. Updating the correction at each stage would instead solve the wrong relaxation equation.

Only passes that improve the independent bulk error are accepted. Boundary trace and tangential errors are also recorded. Finite neural expressivity and regularization mean that an exact fit is not guaranteed. Optimizer settings and the second-pass schedule are engineering defaults, not reported settings for the paper's two-dimensional experiment.

## Configuration and experiments

TOML configuration sections are `[simulation]`, `[initialization]`, and `[sweep]`. See the three files in `configs/`. CLI arguments override TOML; unprovided fields use the defaults in `config.py`. Unknown fields and invalid values fail explicitly.

```bash
# Change the squared tangential-seminorm weight and physical regularization.
python -m rdpa_heat run --config configs/baseline.toml --alpha 0.5 --epsilon 0.001 --checkpoint runs/initial/initial.npz --output runs/alpha_05

# Replace the entire artificial-time refinement schedule.
python -m rdpa_heat initialize --init-pass 100:1e-4 --init-pass 400:1e-6 --output runs/refined

# Sweep h and epsilon while reusing one exact initial parameter vector.
python -m rdpa_heat sweep --config configs/convergence.toml --checkpoint runs/initial/initial.npz --output runs/convergence

# Smaller exploratory sweep.
python -m rdpa_heat sweep --checkpoint runs/initial/initial.npz --sweep-steps 16 32 64 --sweep-epsilons 0.001 --output runs/small_sweep

# Quadrature checks at fixed physical configuration and initial parameters.
python -m rdpa_heat run --checkpoint runs/initial/initial.npz --quadrature-points 37 --output runs/quadrature_37
python -m rdpa_heat run --checkpoint runs/initial/initial.npz --quadrature-points 73 --output runs/quadrature_73
```

`configs/figure_5_1.toml` requests the expensive `N=16384`, `epsilon=1e-5` experiment. It is a reproduction target, not an accuracy guarantee. Use `--no-plots` to skip figures or `--quiet` to suppress progress messages. `--help` lists all optimizer and numerical overrides.

### Publication convergence figure with alpha=1

```bash
python make_convergence_plot.py
# Resume completed trajectories after an interruption:
python make_convergence_plot.py --resume
# Prioritize a particular epsilon while completing the same sweep:
python make_convergence_plot.py --resume --first-epsilon 0.0001
# Re-export the figure without recomputing trajectories:
python make_convergence_plot.py --plot-only
```

This standalone script runs epsilon = 0.1, 0.01, 0.001, 0.0001 at N = 32, 64, 128, 256, 512, 1024, 2048, 4096, using alpha=1, exactly 20 modified Gauss–Newton steps and float64. It reuses the existing initial checkpoint when available. The defaults are composite Simpson with 19 points per axis (361 bulk points), 19 points per boundary edge (76 edge-associated points), and independent 65×65 Simpson error quadrature. `--quad-points`, `--validation-points`, and `--steps` make those choices explicit inputs.

The figure itself specifies these numerical settings. `runs/publication_alpha1/` contains vector PDF/SVG, 600-dpi PNG, a caption, CSV data, provenance metadata, and endpoint checkpoints. The script uses the production solver kernels in compiled physical-step chunks and verifies agreement against the public integrator before running. It records endpoint diagnostics rather than exporting the full trajectory at every step.

Completed records remain in the CSV when the execution order changes. Ctrl+C saves progress and an explicitly labeled partial figure; an interrupted trajectory restarts on the next resume. The figure is also refreshed after each completed epsilon curve.

The default convergence sweep has 32 trajectories and can be expensive on CPU. Initialization and validation error can obscure time convergence; inspect the plotted initialization floor and quadrature sensitivity before interpreting observed slopes.

## Results and failure handling

Every run records the resolved configuration and numerical environment. Initialization saves `initial.npz`, JSON/CSV fitting history, and initial-error measurements. Heat runs add:

- `trajectory.npz`: all time-grid parameters and inner-iteration diagnostics, without sampled Jacobian histories.
- `metrics.csv`: independent bulk errors and boundary norms at every physical time.
- `inner_iterations.csv`: each iteration's defect, four squared contributions, increment norm, linear stationarity, and `h*delta/epsilon²`.
- `summary.json`, `report.md`, and `final.npz`.
- `snapshots.png/.pdf` and `diagnostics.png/.pdf`; snapshots are labeled with actual grid times.

Convergence sweeps add `convergence.csv` with adjacent observed orders and `convergence.png/.pdf`. The initialization checkpoint at the sweep root is shared by every trajectory.

Nonfinite initialization stages or physical solves stop the run. Failure artifacts preserve the last valid state and error context; the method is not silently changed. Defect ratios are diagnostic quantities, not certificates that unknown constants in the theoretical assumptions are satisfied.

## Validation coverage

Tests cover Simpson measures and polynomials, parameter count/layout, mixed derivatives, the modified-regularization algebra, rank-deficient least squares, RK4 stage updates and fourth-order convergence on an independently solvable nonlinear model, best-checkpoint selection, frozen physical Jacobians, boundary enforcement, backend agreement, and failure-state retention.

The exact heat solution `exp(-t/2) * cos(x1/2) * cos(x2/2)` is used only for initial fitting and diagnostics. Physical residuals use the actual previous network state.
