# JAX implementation plan for the regularized heat-equation solver

**Intended Markdown file:** `project.md`. This specification has not been saved because the session is in Plan mode.

## 1. Objective and mathematical decisions

Implement the modified regularized Gauss–Newton iteration in **equation (4.4)**, following your correction, for the square-domain heat equation in Section 5.2 of the [attached paper](</Users/pasquale/Downloads/Regularized_dynamical_parametric_approximation_of_boundary_value_problems.pdf>):

\[
\partial_t y=\Delta y,\qquad
\Omega=[-\pi,\pi]^2,\qquad
y|_{\partial\Omega}=0,
\]

\[
y_0(x_1,x_2)=\cos(x_1/2)\cos(x_2/2).
\]

Use the exact solution exclusively for initialization targets and validation:

\[
y(t,x)=e^{-t/2}y_0(x).
\]

The implementation has two separate stages:

1. Compute accurate initial neural-network parameters using an optimizer followed by regularized RK4 refinement in artificial time.
2. Advance the heat equation using implicit Euler and the modified iteration (4.4).

**Agreed boundary norm**

Expose `alpha` through configuration and the command line, with default `0.2`:

\[
\boxed{
\|v\|_{\Gamma,\alpha}^{2}
=
\int_\Gamma v^2\,ds
+
\alpha\int_\Gamma|\partial_{\mathrm{tan}}v|^2\,ds
}
\]

Require \(\alpha>0\). Here \(\partial_{\mathrm{tan}}\) denotes differentiation along each boundary edge.

This is the agreed quadratic interpretation. The printed sum of norms on page 15 would introduce a cross term when squared; the implementation must document this distinction.

**Other decisions**

- Follow the coefficients printed in (4.4). They produce a matrix regularization of **\(3\varepsilon^2/2\)**, despite the different coefficient in the nearby matrix remark.
- Freeze parameter Jacobians at the beginning of each physical time step, as prescribed by the paper.
- Perform exactly \(K=20\) modified Gauss–Newton updates by default.
- Use float64 throughout.
- Use augmented QR least squares by default; provide normal equations with Cholesky as a comparison backend.
- Limit the initial implementation to the square domain and its cosine initial condition. The L-shaped experiment and alternative formulations A1–A3 are outside this implementation.

## 2. Step-by-step numerical implementation

### Step 1 — Define the 153-parameter neural representation

Implement a pure JAX scalar function `phi(theta, x)` with:

\[
z_0=x+b_{\mathrm{in}},\qquad
z_1=\tanh(W_1z_0+b_1),
\]

\[
z_i=\tanh(W_i z_{i-1}+b_i),\quad i=2,3,4,
\]

\[
\Phi(\theta,x)=w_{\mathrm{out}}^\top z_4+b_{\mathrm{out}}.
\]

Use these parameter shapes:

| Parameter | Shape | Count |
|---|---:|---:|
| Input translation | `(2,)` | 2 |
| First-layer weights and bias | `(6,2)`, `(6,)` | 18 |
| Three further hidden layers | Three × `((6,6), (6,))` | 126 |
| Output weights and scalar bias | `(6,)`, `()` | 7 |
| **Total** | | **153** |

Resolve the paper’s inconsistent output-dimension notation in favor of a scalar output, consistent with the PDE and its stated parameter count.

Initialize weights with Glorot uniform initialization, biases and input translation with zeros, and a configurable random seed. Keep the input coordinates in their physical units.

Use a parameter PyTree for model construction and a flat vector for numerical integration. `ravel_pytree` supplies the flatten/unflatten pair. [JAX documentation](https://docs.jax.dev/en/latest/_autosummary/jax.flatten_util.ravel_pytree.html)

Preserve the input translation even though it creates parameter redundancy with the first-layer bias. Regularization must handle this redundancy.

### Step 2 — Construct domain and boundary quadrature

Interpret the paper’s “10 quadrature nodes per axis” as 10 coarse endpoints and nine inserted midpoints:

- 19 uniformly spaced evaluation points per coordinate.
- \(19^2=361\) bulk evaluations.
- 19 points per boundary edge, giving 76 edge-associated evaluations.

For spacing \(d=2\pi/18\), use composite Simpson weights:

\[
w=\frac d3(1,4,2,4,\ldots,2,4,1).
\]

Build bulk weights using the tensor product. Integrate the four boundary edges separately using the same one-dimensional rule.

Retain each corner’s contribution from its two incident edges, including its edge-specific tangent. This is the endpoint treatment of the separate edge integrals.

Check that:

\[
\sum_i w_{\Omega,i}=4\pi^2,
\qquad
\sum_j w_{\Gamma,j}=8\pi.
\]

Use physical integral weights throughout. Independently normalizing bulk and boundary losses would change the method.

For tangential differentiation:

- On \(x_1=\pm\pi\), differentiate with respect to \(x_2\).
- On \(x_2=\pm\pi\), differentiate with respect to \(x_1\).

Create a separate, finer validation rule with 65 points per coordinate and 65 points per edge.

### Step 3 — Implement JAX differential operators

Provide batched evaluations of:

\[
\Phi,\qquad
\Delta_x\Phi,\qquad
\partial_{\mathrm{tan}}\Phi,
\]

and their parameter Jacobians.

Compute the spatial Hessian using forward-over-reverse automatic differentiation and take its trace for the Laplacian. Differentiate the scalar quantity \(\Phi-h\Delta\Phi\) with respect to the flat parameter vector, then batch over quadrature points.

For boundary derivatives, use a spatial JVP in the edge-tangent direction, followed by parameter differentiation. These derivative compositions are supported by JAX’s higher-order autodiff. [JAX documentation](https://docs.jax.dev/en/latest/higher-order.html)

Implementation requirements:

- Enable `jax_enable_x64` before constructing numerical arrays. [JAX precision documentation](https://docs.jax.dev/en/latest/default_dtypes.html)
- Use `vmap` for pointwise batching and `jit` for reusable kernels.
- Keep NumPy, plotting, and file operations outside compiled functions.
- Pass changing numerical values as arguments; avoid rebuilding compiled functions every step.
- Keep matrix dimensions fixed for a given quadrature configuration.

An in-memory feasibility check already verified finite float64 derivative matrices of sizes `(361,153)`, `(76,153)`, and `(76,153)` on the installed JAX 0.4.38 CPU environment.

### Step 4 — Obtain a rough initial fit

Train the network against \(y_0\) using the bulk quadrature loss:

\[
L_{\mathrm{fit}}(\theta)
=
\sum_i w_{\Omega,i}
\left[\Phi(\theta,x_i)-y_0(x_i)\right]^2.
\]

Use full-batch Optax Adam. Proposed defaults:

- Seed: `0`.
- Updates: `20_000`.
- Learning rate: \(10^{-3}\) for the first 10,000 updates, then \(10^{-4}\).
- Preserve the checkpoint with the lowest independently evaluated bulk error, measured every 100 updates.

These optimizer settings are engineering defaults; the attached paper does not specify them.

Record bulk fitting error and boundary value and tangential-derivative errors. Do not assume a small bulk loss guarantees an accurate boundary trace.

### Step 5 — Refine the initial parameters using RK4

This is an artificial-time computation, separate from heat evolution.

At the beginning of refinement pass \(r\), freeze an anchor \(\theta_a\) and define:

\[
c_r(x)=y_0(x)-\Phi(\theta_a,x).
\]

The underlying function-space problem is:

\[
\frac{dz}{d\tau}=c_r,\qquad
z(0)=\Phi(\theta_a),\qquad
0\leq\tau\leq1.
\]

Its exact endpoint is \(z(1)=y_0\).

The parameter-space velocity at current parameters \(q\) is:

\[
F_r(q)
=
\underset{v}{\arg\min}\;
\left\|\Phi'(q)v-c_r\right\|_\Omega^2
+
\varepsilon_{\mathrm{init},r}^2\|v\|_2^2.
\]

With \(J(q)\) the sampled parameter Jacobian, solve:

\[
\begin{bmatrix}
W_\Omega^{1/2}J(q)\\
\varepsilon_{\mathrm{init},r}I
\end{bmatrix}
v
\approx
\begin{bmatrix}
W_\Omega^{1/2}c_r\\
0
\end{bmatrix}.
\]

For \(s=1/M_r\), apply classical RK4:

\[
\begin{aligned}
k_1&=F_r(q),\\
k_2&=F_r(q+\tfrac{s}{2}k_1),\\
k_3&=F_r(q+\tfrac{s}{2}k_2),\\
k_4&=F_r(q+sk_3),\\
q_{\mathrm{new}}
&=q+\frac{s}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

The implementation must preserve these rules:

1. Freeze \(c_r\) across **all steps and stages of the entire pass**.
2. Recompute \(J(q)\) and solve a new least-squares problem at **every RK4 stage**.
3. Keep \(\varepsilon_{\mathrm{init},r}\) constant within a pass.
4. Include only the bulk fitting metric and parameter regularization.
5. Do not include the heat Laplacian, physical time step, or boundary penalty.
6. Combine the four velocities in parameter space.
7. Recompute the frozen correction only when starting the next pass.

Using \(y_0-\Phi(q)\) at each stage would produce a relaxation equation instead. Even its exact function-space solution would retain an \(e^{-1}\) fraction of the initial error at \(\tau=1\).

Use two refinement passes by default:

| Pass | RK4 steps | Regularization |
|---|---:|---:|
| 1 | 100 | \(10^{-4}\) |
| 2 | 200 | \(10^{-5}\) |

The first-pass settings are borrowed from the cited paper’s Section 8.2; the second-pass reduction is an explicit engineering choice consistent with the attached paper’s discussion. [Referenced initialization procedure](https://arxiv.org/html/2501.12118v1#S8.SS2)

After each pass, accept its endpoint only if the independent bulk error decreases; otherwise retain the previous anchor and record the rejected pass. Abort initialization on nonfinite computations.

The finite neural approximation and positive regularization generally prevent an exact endpoint match. Save the achieved errors rather than claiming \(\Phi(\theta_0)=y_0\).

### Step 6 — Assemble equation (4.4)

At physical step \(n\), let:

\[
u_n=\Phi(\theta_n),\qquad
\theta^{(0)}=\theta_n.
\]

Freeze all matrices at \(\theta_n\). Define:

\[
J_\Omega=\partial_\theta\Phi(\theta_n,X_\Omega),
\qquad
L_\Omega=\partial_\theta\Delta\Phi(\theta_n,X_\Omega),
\]

\[
J_\Gamma=\partial_\theta\Phi(\theta_n,X_\Gamma),
\qquad
D_\Gamma=\partial_\theta\partial_{\mathrm{tan}}
\Phi(\theta_n,X_\Gamma).
\]

Multiply the objective in (4.4) by \(h^2\), preserving its minimizer. Assemble:

\[
B=W_\Omega^{1/2}(J_\Omega-hL_\Omega),
\]

\[
C=
\begin{bmatrix}
W_\Gamma^{1/2}J_\Gamma\\
\sqrt{\alpha}\,W_\Gamma^{1/2}D_\Gamma
\end{bmatrix}.
\]

At iteration \(k\), recompute:

\[
r_\Omega^{(k)}
=
W_\Omega^{1/2}
\left[
\Phi(\theta^{(k)})-u_n-h\Delta\Phi(\theta^{(k)})
\right],
\]

\[
r_\Gamma^{(k)}
=
\begin{bmatrix}
W_\Gamma^{1/2}\Phi(\theta^{(k)})\\
\sqrt{\alpha}\,W_\Gamma^{1/2}
\partial_{\mathrm{tan}}\Phi(\theta^{(k)})
\end{bmatrix},
\qquad
d^{(k)}=\theta^{(k)}-\theta_n.
\]

The scaled minimization is:

\[
\boxed{
\min_{\Delta\theta}
\|B\Delta\theta+r_\Omega^{(k)}\|_2^2
+
\|C\Delta\theta+r_\Gamma^{(k)}\|_2^2
+
\frac{\varepsilon^2}{2}
\|\Delta\theta+d^{(k)}\|_2^2
+
\varepsilon^2\|\Delta\theta\|_2^2
}
\]

Use the augmented system:

\[
M=
\begin{bmatrix}
B\\C\\
(\varepsilon/\sqrt2)I\\
\varepsilon I
\end{bmatrix},
\qquad
b^{(k)}=
\begin{bmatrix}
r_\Omega^{(k)}\\
r_\Gamma^{(k)}\\
(\varepsilon/\sqrt2)d^{(k)}\\
0
\end{bmatrix},
\]

\[
\Delta\theta^{(k)}
=
\arg\min_{\Delta\theta}\|M\Delta\theta+b^{(k)}\|_2^2.
\]

For the baseline dimensions, \(M\) has shape `(819,153)`.

The equivalent normal equations are:

\[
\boxed{
\left(B^\top B+C^\top C+\frac32\varepsilon^2I\right)
\Delta\theta^{(k)}
=
-B^\top r_\Omega^{(k)}
-C^\top r_\Gamma^{(k)}
-\frac12\varepsilon^2d^{(k)}
}
\]

Both regularization terms, including the accumulated displacement on the right-hand side, must be retained.

### Step 7 — Execute each physical time step

For each \(n\):

1. Evaluate and retain the actual network state \(u_n=\Phi(\theta_n)\).
2. Assemble \(B,C,M\) at \(\theta_n\).
3. Compute a reduced QR factorization \(M=QR\) once.
4. Perform \(K\) iterations:
   - Recompute the nonlinear residuals at \(\theta^{(k)}\).
   - Solve \(R\Delta\theta^{(k)}=-Q^\top b^{(k)}\).
   - Set \(\theta^{(k+1)}=\theta^{(k)}+\Delta\theta^{(k)}\).
5. Accept \(\theta_{n+1}=\theta^{(K)}\).

JAX provides reduced QR and triangular solves directly. [QR documentation](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.qr.html), [triangular-solve documentation](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.solve_triangular.html)

Specific safeguards against implementation errors:

- The boundary residual contains the current solution, not its difference from \(u_n\).
- The solved increment is already \(\Delta\theta\); do not multiply it by \(h\).
- Frozen matrices remain unchanged through the \(K\) iterations and are rebuilt at the next physical step.
- Physical evolution always uses the computed previous network state.
- Do not add damping, line searches, adaptive regularization, or early termination to the baseline iteration.
- Abort and preserve the last valid state on nonfinite values or a failed linear solve; do not silently alter the method.

The optional normal-equation backend must use the same coefficients and reuse one Cholesky factorization per physical step.

### Step 8 — Record defects and accuracy

For every inner iteration, compute the paper’s linearized defect:

\[
\delta_n^{(k)}
=
\frac{\|M\Delta\theta^{(k)}+b^{(k)}\|_2}{h}.
\]

Record its four component contributions separately.

Also record:

- Independent absolute and relative bulk \(L^2\) errors.
- Boundary \(L^2\), tangential seminorm, and weighted boundary norm.
- Parameter increment norms and linear-solve stationarity.
- \(h\delta_n^{(k)}/\varepsilon^2\) and
  \(h\sum_k\delta_n^{(k)}/\varepsilon^2\).
- Initialization accuracy, runtime, configuration, package versions, and device.

The ratios are diagnostics related to the paper’s assumptions. They do not certify the theoretical bounds because their constants are unknown.

Generate solution and error plots at \(t=0,0.5,1\) for the default even step counts. For arbitrary configurations, label snapshots with their actual time-grid values.

## 3. File structure and interfaces

```text
.
├── project.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── baseline.toml
│   ├── convergence.toml
│   └── figure_5_1.toml
├── src/
│   └── rdpa_heat/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── problem.py
│       ├── model.py
│       ├── quadrature.py
│       ├── operators.py
│       ├── linalg.py
│       ├── initialization.py
│       ├── integrator.py
│       ├── diagnostics.py
│       ├── experiments.py
│       └── plotting.py
├── tests/
│   ├── conftest.py
│   ├── test_model_operators.py
│   ├── test_quadrature_boundary.py
│   ├── test_modified_regularization.py
│   ├── test_rk4_initialization.py
│   └── test_heat_integration.py
└── runs/                         # Generated results; excluded from version control
```

Responsibilities:

- **Problem and model:** analytic heat data, architecture, parameter serialization.
- **Quadrature and operators:** weighted sampling, tangents, spatial and parameter derivatives.
- **Linear algebra:** reusable QR and Cholesky factorizations and solve diagnostics.
- **Initialization:** optimizer warm start, frozen-target refinement passes, stagewise RK4.
- **Integrator:** equation-(4.4) assembly and physical evolution.
- **Experiments and diagnostics:** reproducible runs, convergence sweeps, metrics and plots.

Expose these primary interfaces:

```python
phi(theta, x) -> scalar
build_quadrature(config) -> QuadratureRule

fit_initial_condition(config, quadrature) -> InitializationResult
refine_initial_condition(theta_anchor, config, quadrature) -> InitializationResult

assemble_step(theta_n, h, epsilon, alpha, quadrature) -> StepSystem
advance_step(theta_n, step_system, config) -> StepResult
simulate(theta0, config, quadrature) -> SimulationResult
```

Use typed configuration dataclasses and JAX-compatible numerical result containers. Store the flattened parameter layout alongside initialization checkpoints.

Provide CLI commands:

```bash
python -m rdpa_heat initialize --config configs/baseline.toml
python -m rdpa_heat run --config configs/baseline.toml --alpha 0.2
python -m rdpa_heat sweep --config configs/convergence.toml
```

Allow `run` to load an initialization checkpoint or compute one when none is supplied. Allow overrides for `alpha`, physical regularization, step count, iteration count, quadrature size, initialization schedule, seed, and linear solver.

Each run produces resolved configuration and metadata in JSON, numerical arrays in NPZ, tabular diagnostics in CSV, and plots in PNG/PDF.

## 4. Defaults and implementation sequence

| Setting | Default | Origin |
|---|---|---|
| Domain and initial condition | Square and cosine product above | Section 5.2 |
| Final time | \(T=1\) | Paper experiments |
| Architecture | Four width-six tanh layers; 153 parameters | Section 5.2 |
| Bulk quadrature | \(19\times19\) | Section 5.1 |
| Boundary quadrature | 19 points per edge | Matching Simpson construction |
| Boundary weight | `alpha=0.2`, user configurable | Agreed quadratic convention |
| Inner iterations | \(K=20\) | Section 5.1 |
| Baseline physical steps | \(N=256\) | Engineering default |
| Baseline regularization | \(\varepsilon=10^{-3}\) | Engineering default |
| Linear solver | Augmented QR | Numerical robustness choice |
| Precision | float64 | Section 5.1 |
| Validation quadrature | \(65\times65\), 65 per edge | Engineering default |

Use the installed JAX 0.4.38, jaxlib 0.4.38, and Optax 0.2.4 environment as the initial compatibility baseline. Record exact dependency versions for reproducible experiments.

Implement and validate in this order:

1. Package configuration, heat data, model, and quadrature.
2. Spatial derivatives and parameter Jacobians.
3. Augmented least-squares solver and equation-(4.4) algebra.
4. Optimizer warm start and RK4 refinement.
5. One physical step, then complete trajectories.
6. Diagnostics, checkpointing, command-line interfaces, and plotting.
7. Convergence and paper-setting experiments.

Configure the convergence sweep with:

\[
N\in\{32,64,128,256,512,1024,2048,4096\},
\]

\[
\varepsilon\in\{10^{-1},10^{-2},10^{-3},10^{-4}\}.
\]

Reuse the same initial checkpoint throughout each sweep.

Provide a separate expensive configuration with \(N=2^{14}\) and \(\varepsilon=10^{-5}\), corresponding to the Figure 5.1 caption. Treat this as a reproduction target, with the documented boundary-norm interpretation and unspecified initialization choices clearly recorded.

## 5. Verification and acceptance criteria

**Model, quadrature, and derivatives**

- Verify the parameter count is 153 and output is scalar.
- Verify Simpson integration of constants and cubic polynomials.
- Check bulk area \(4\pi^2\) and boundary length \(8\pi\).
- Confirm \(\Delta y_0=-y_0/2\) and \(\|y_0\|_{L^2(\Omega)}=\pi\).
- Check directional parameter derivatives against finite differences.
- Use \(v(x_1,x_2)=x_1\) to distinguish tangential derivatives from full spatial gradients: its tangential derivative vanishes on vertical edges.

**Modified regularization**

- Compare augmented QR against the derived normal equations on small well-conditioned problems.
- Differentiate the explicit quadratic objective independently and verify stationarity at the computed increment.
- Test nonzero \(d^{(k)}\) to catch omission of the accumulated-displacement term.
- Assert the matrix coefficient is \(3\varepsilon^2/2\) and the displacement coefficient is \(\varepsilon^2/2\).
- Verify multiplying the entire objective by \(h^2\) preserves the increment.
- Test rank-deficient unregularized matrices with positive regularization.
- Verify changing `alpha` changes the squared tangential contribution linearly.

**RK4 initialization**

- Verify the forcing remains unchanged across every stage and step within a pass.
- Verify stage Jacobians are recomputed at their respective stage parameters.
- For a linear parametrization, compare against the exact constant-velocity endpoint.
- For the nonlinear scalar parametrization \(\Phi(q)=q^2\), compare with an independent root solve of
  \[
  c\tau=q^2-q_a^2+\frac{\varepsilon^2}{2}\log(q/q_a).
  \]
  Verify fourth-order convergence under pseudo-step refinement before roundoff dominates.
- Confirm the initializer does not depend on physical `h`, `alpha`, or the Laplacian.
- Verify pass rejection preserves the previous anchor.

**Physical evolution**

- Verify matrices remain fixed within a physical step and are rebuilt between steps.
- Confirm exactly \(K\) updates occur.
- Check that a nonzero previous boundary trace is actively penalized.
- Compare QR and normal-equation trajectories where conditioning permits.
- Evaluate errors on independent quadrature.
- Demonstrate approximately first-order time convergence in the range where initialization, regularization, and quadrature errors do not dominate.
- Repeat representative runs with 37 and 73 quadrature points per axis to identify quadrature limitations.

A completed implementation must pass the algebraic and algorithmic tests and produce reproducible initialization and heat-evolution reports. Agreement with a particular plotted error level is an experimental result to establish, not an assumption built into the tests.
