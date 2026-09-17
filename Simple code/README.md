# Readable JAX heat solver

This directory is independent of `src/rdpa_heat`. Only `check_reference.py`
imports the original implementation, for verification. Requires Python 3.10+
with JAX, jaxlib, Optax, NumPy, SciPy and Matplotlib. The tested environment is
JAX/jaxlib 0.4.38 and Optax 0.2.4; run metadata records installed versions.

## Run and configure

Edit the ordinary variables at the top of `run.py`, then from the project root:

```bash
python "Simple code/run.py"
```

The default mode is the **full, expensive convergence sweep**, with a fresh
initialization unless `results/initial.npz` already exists. No long sweep is
started by the verification script. For a first short run choose `MODE='single'`
and `N_STEPS=8`. To reuse the existing accurate initial fit, set:

```python
INPUT_CHECKPOINT = 'runs/default_initialization/initial.npz'
```

Input paths are relative to the current working directory. The default output
path is always this directory's `results`. Choose a separate `OUTPUT` for
experiments with different common settings or initialization.

Modes:

- `initialize`: Adam and RK4 only (or load the specified initial checkpoint).
- `single`: one trajectory with `EPSILON` and `N_STEPS`.
- `convergence`: all pairs in `EPSILONS` and `STEP_COUNTS`, in listed order.
- `plot`: regenerate exports from saved completed cases, without evolution.

Set `USE_JIT=False` for debugging. Both modes use the same equations and loop
bodies; disabling compilation can be much slower. `CHUNK_SIZE` controls how
many physical steps run between progress messages and finite-value checks.
An interrupt is handled when control returns from compiled code.

## Recommended reading order

1. `run.py`: parameter block.
2. `integrator.py`: `build_matrices` and `residual_vector`.
3. `integrator.py`: `heat_step` and its Gauss–Newton loop.
4. `integrator.py`: `physical_chunk` and `solve_heat`.
5. `initialization.py`: Adam, frozen correction, and the four RK4 stages.
6. `numerics.py`: parameter slices and automatic differentiation.

## Numerical method

Solve `u_t = Δu` on `[-π,π]²`, with zero Dirichlet data and
`u(0,x)=cos(x₁/2)cos(x₂/2)`. The exact solution `exp(-t/2)u(0,x)` is used only
for initialization targets and error measurement. Four width-six tanh layers,
an input translation and scalar output give 153 parameters. Flat slices match
the original checkpoint order exactly. All numerical computation uses float64.

The default physical quadrature is composite Simpson with 19 points per axis:
361 bulk nodes and 76 edge-associated nodes. Corners contribute to both incident
edge integrals. Weights integrate physical area and length, without independent
normalization. Errors use an independent 65-by-65 rule and 65 points per edge.

The boundary squared norm is `∫u² ds + alpha ∫(∂tan u)² ds`.
This agreed quadratic convention differs from squaring the printed sum of norms,
which would add a cross term. Default `alpha=1`; it must be positive.

At each physical step, modified equation (4.4) uses frozen Jacobians of
`Phi-h*Laplacian(Phi)` and the boundary traces. The augmented matrix contains
both `(epsilon/sqrt(2))*I` and `epsilon*I`. The former residual contains the
accumulated displacement `theta_k-theta_n`; the latter contains zero.
Consequently the normal matrix has `1.5*epsilon²*I`, and the right-hand side has
`-0.5*epsilon²*(theta_k-theta_n)`. These follow the printed equation, despite the
different coefficient in the nearby matrix remark. Exactly `GN_STEPS` updates
(default 20) are taken, with no damping or early exit. The default QR and optional
Cholesky normal-equation paths share the same nonlinear residuals.

Initialization first minimizes the bulk fitting error with Adam, keeping the
best independently evaluated parameters. Each RK4 pass freezes
`correction = y0-Phi(anchor)` for the entire artificial interval `[0,1]`.
Every stage solves a new regularized Jacobian least-squares problem at the stage
parameters. It contains neither Laplacians nor boundary penalties. The pass is
accepted only if independent bulk error improves. Default passes are
`(100,1e-4)` and `(200,1e-5)`. Adam and pass-two settings are engineering choices.
A resumed initial checkpoint is authoritative: changing Adam settings does not
refit it unless you select a new output directory or set `RESUME=False`.

## Results and resumption

- `initial.npz`: the shared starting parameter vector.
- `initialization.json`: fitting history when initialization was computed.
- `settings.json`: resolved settings, dependency versions and device.
- `case_eps_..._N_....npz`: each completed trajectory, settings, scalar errors,
  and per-iteration diagnostics, with the starting vector for comparison.
- `convergence.csv`: all completed cases, including cases outside a newly selected subset.
- `convergence.pdf`, `.svg`, `.png`: vector exports and a 600-dpi raster image.
  The title identifies incomplete requested sweeps as partial.

Diagnostics columns are defect `||M increment+b||/h`, parameter increment norm,
absolute stationarity `||M.T(M increment+b)||`, and `h*defect/epsilon²`.
These are diagnostics, not certificates of theoretical bounds. The endpoint
metrics include absolute/relative bulk errors and three boundary norms.

With `RESUME=True`, completed matching cases are skipped. Matching requires the
same time, alpha, iteration count, solver, quadrature, precision, and **identical
starting parameters**. Epsilon and physical step count identify the case.
Settings mismatches raise a clear error requesting a new output directory.
There is deliberately no source-hash system; after changing numerical formulas,
choose a new output directory yourself. Checkpoints store numeric arrays only,
never Python pickles. Raw compatible `theta` archives are also accepted; their
parameter ordering is the caller's responsibility.

Ctrl-C retains completed cases and refreshes a partial figure when at least one
requested case has completed. Resume restarts an unfinished trajectory. Finite
checks abort on numerical failure; no fallback silently changes the method.
The previous package and its 23 publication cases are untouched.

## Verification

```bash
python "Simple code/check_reference.py"
```

This compares original and simplified parameter initialization, derivatives,
quadrature, matrices, residuals, increments, and short trajectories. It checks
both regularization coefficients, well-conditioned QR/normal agreement,
independent field errors, checkpoint compatibility, frozen-target RK4 and fourth
order on a nonlinear scalar problem, Adam execution, and pass rejection.
A tiny shared-initialization sweep checks case skipping, CSV preservation,
mismatched settings, and partial PDF/SVG/PNG exports. It stores its small report
under `results/verification`, separate from publication runs. These comparisons
establish implementation agreement; they do not replace a full convergence study.

## L-shaped experiments (Section 5.2.2)

Edit `run_l.py`, then run:

```bash
python "Simple code/run_l.py"
```

This solves the heat equation on the square with the upper-right quadrant removed,
with initial data `sin(x₁) sin(x₂)` and exact solution `exp(-2t) sin(x₁) sin(x₂)`.
The default boundary weight is **0.1**, as in the L-shaped paper experiment.
The shared integrator, network, QR/normal choices and RK4 stage code are reused.
Only sampled initial/exact targets and quadrature change. The agreed quadratic
boundary norm and equation-(4.4) regularization coefficients remain unchanged.

`MAIN_NODES=5` means five main Simpson endpoints plus four midpoints per side of
**each of three squares**. Composite Simpson 1/3 is applied to each square, and
shared bulk nodes are merged by **summing** weights. There are 225 unique bulk
nodes and 70 boundary entries (64 unique boundary coordinates). Collinear boundary
nodes merge; corners keep both edge-specific tangential contributions. Internal
square interfaces are never penalized as boundary. The area is `3π²` and the
boundary length is `8π`.

This explicit five-main-node choice gives spacing `π/8`, slightly denser than the
square default `π/9`. `VALIDATION_MAIN_NODES=17` gives 33 points per patch side,
3,201 unique bulk nodes, 262 boundary entries, and spacing `π/32`, matching the
square validation spacing. Do not confuse main nodes with total evaluation nodes.

The default `MODE='convergence'` starts the full expensive sweep after computing
(or resuming) a **new L-shaped initialization**. Modes are:

- `initialize`: Adam plus frozen-target RK4, or load a labelled L-shaped checkpoint.
- `single`: one trajectory and initial/final solution/error maps. Defaults are
  `N_STEPS=2048`, `EPSILON=0.001`, `T=1`, matching Figure 5.4's parameters.
- `convergence`: the four epsilon values and eight step counts listed in the script,
  using one initial vector. This corresponds to the type of study in Figure 5.3.
- `conditioning`: first-step matrices and QR/normal comparison with
  `CONDITIONING_STEPS=2048`, `CONDITIONING_EPSILON=0.0001`. No full trajectory runs.
- `plot`: regenerate convergence plots and, if the selected `EPSILON,N_STEPS` case
  exists, its initial/final maps, without evolving the PDE.

Results default to `results/l_shape`; choose another `OUTPUT` for a normal-equation
sweep. Matching completed trajectories are reused. Square or unlabelled legacy
checkpoints are rejected by the L-shaped entrypoint, while the square entrypoint
continues to accept its legacy checkpoints. New checkpoints/results identify the
problem. No previous square results are modified.

The default display grid is 257×257, with the removed quadrant masked. Every panel
has its own labelled color scale. This grid is for visualization only; bulk and
boundary errors use independent quadrature. Map arrays are in `display_maps.npz`.

Conditioning outputs distinguish the actual augmented matrix `M`, actual normal
matrix `G=M.T@M`, and **literal Figure 5.5 caption matrix**
`B.T@B + epsilon²/h² I`. Here `B` is the physical weighted derivative of
`Phi-h*Laplacian(Phi)`. The caption matrix is bulk-only and is not the actual
modified equation-(4.4) system. `G` uses the solver's h²-scaled convention; uniform
scaling changes singular values but not condition numbers. The report includes
`κ(M)²`, directly measured `κ(G)`, eigenvalues, singular values, first-increment
stationarity, and the QR/normal first-step field difference. A failed solve is
recorded without a fallback. There is no assumption that the paper's reported
condition number or loss of accuracy will be reproduced for this initialization.
Figures are PDF/SVG and 600-dpi PNG; measurements are NPZ/CSV/JSON.

Run the short checks with:

```bash
python "Simple code/check_l.py"
python "Simple code/check_reference.py"
```

The first script checks L-shaped geometry, polynomial quadrature, duplicate-weight
aggregation, sine targets, frozen RK4 forcing, normal/QR agreement, domain checkpoint
separation, small convergence/resume runs, maps and spectra. It writes only a small
report to `results/verification_l`. Its deliberately short initialization is a
functional test, not a paper-quality fit. The second checks square compatibility.
Neither launches a full sweep.
