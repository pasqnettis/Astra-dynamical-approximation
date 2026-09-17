# Specification audit and numerical validation

Reviewed against [project.md](project.md). The implementation covers the requested square-domain heat experiment and the modified iteration in equation (4.4). The issues identified during the audit are fixed below; the full Figure 5.1 experiment remains an unrun reproduction target.

## Findings and fixes

| Finding | Consequence | Resolution |
|---|---|---|
| Missing README referenced by package metadata | Incomplete installation/documentation deliverable | Added README with setup, equations, input conventions, experiment commands, and result interpretation; built a wheel successfully offline. |
| Nonfinite initialization diagnostics were appended directly to JSON history | Saving a failed run could raise a serialization error and obscure the original numerical failure | Preserve nonfinite diagnostic values as explicit strings in failure history and retain the previously validated state. Regression test covers the persisted checkpoint and failure report. |
| Public step API could use a different solver from the stored factorization | QR and Cholesky factors could be interpreted incorrectly, producing a wrong increment | Store the factorization method and reject mismatched use. |
| Public step API could receive parameters different from the assembly anchor | Displacement regularization and frozen Jacobians could refer to different states | Reject an inconsistent anchor. |
| Direct assembly accepted NaN scalars despite positive-value checks | Invalid inputs could enter compiled numerical kernels | Require finite positive h, epsilon, and alpha. |
| Final-time checkpoint could be loaded as time-zero initial data | The run would silently start from the wrong physical state | Reject checkpoint metadata with nonzero physical time. |
| `build_quadrature` accepted only a point count | The configuration-based interface in the specification was missing | Accept either a RunConfig or an odd point count. |
| Plotting attempted to use home-directory caches | Restricted environments generated cache errors and repeated font discovery | Use writable temporary plotting caches, while respecting existing environment overrides. |
| Run completion was recorded before plots finished | A plotting failure could leave misleading completion metadata | Write completion metadata after requested artifacts finish. |

## Numerical logic checked

- The neural model has exactly 153 float64 parameters, with physical coordinates and the prescribed four tanh layers.
- Simpson weights integrate physical area and boundary length; corners retain edge-specific tangent contributions.
- The boundary residual is the current solution, and the derivative block uses sqrt(alpha), so its squared contribution has weight alpha.
- Equation (4.4) retains both regularization blocks. Its normal matrix has `1.5 * epsilon² * I`, and the accumulated-displacement right-hand side has coefficient `-0.5 * epsilon²`.
- Matrices and factors are assembled once per physical step; nonlinear residuals are updated for exactly K iterations. Increments are not multiplied by h.
- RK4 initialization freezes the correction for an entire pass, recomputes the Jacobian at every stage, and excludes the heat operator and boundary penalty. An independent nonlinear scalar problem verifies fourth-order convergence.
- Warm-start selection and refinement acceptance use independent bulk quadrature. Rejected passes preserve the anchor.
- Initial parameters are shared across convergence trajectories. The analytic heat solution enters initialization and evaluation, not the physical residual.
- Numerical failure paths retain valid states and serializable context. No adaptive damping or regularization is introduced.

## Executed experiments

Automated checks: the complete suite passed 53 tests before the final configuration-interface test was added. The final affected test module then passed all 13 tests, including that new test, covering 54 tests in total. Python compilation and CLI help checks also passed. An offline wheel build succeeded. PNG/PDF plotting was rechecked after the cache fix, and all four time-convergence trajectories were verified to contain byte-for-byte identical initial parameter arrays.

```bash
python -m pytest -q
python -m pytest -q tests/test_experiments.py
python -m compileall -q src tests
```

All measurements below use seed 0, float64, alpha=0.2, QR, and the specified 153-parameter architecture. Full run configurations, dependency versions, CSV diagnostics, checkpoints, and figures are stored under `runs/`.

### Default initialization

Command:

```bash
PYTHONPATH=src python -m rdpa_heat initialize --config configs/baseline.toml --output runs/default_initialization
```

| Stage | Independent bulk L2 error |
|---|---:|
| Random parameters | 4.23201810 |
| Best Adam checkpoint | 2.68717178e-2 |
| RK4 pass 1: 100 steps, epsilon=1e-4 | 1.93586941e-4 |
| RK4 pass 2: 200 steps, epsilon=1e-5 | 1.32077447e-5 |

Both passes were accepted. The final sampled maximum initial error was 9.87465398e-6, and the weighted boundary norm was 3.79777617e-5. Initialization took approximately 57 seconds in the validation environment, including compilation; runtime is hardware and workload dependent.

### Default physical run

`N=256`, `K=20`, `epsilon=1e-3`, `T=1`, 19-point solver quadrature, and 65-point independent validation:

| Quantity at T=1 | Value |
|---|---:|
| Bulk L2 error | 9.31948316e-4 |
| Relative bulk L2 error | 4.89090465e-4 |
| Sampled maximum absolute error | 3.00624631e-4 |
| Weighted boundary norm | 7.86537570e-6 |
| Final inner-iteration defect | 2.50350151e-3 |

The run completed all 256 steps. Figures at t=0,0.5,1 and full diagnostics are in `runs/baseline_validation/`. Integration took approximately 39 seconds including compilation, excluding artifact generation.

### Time convergence

Shared initial checkpoint, epsilon=1e-3, K=20, T=1:

| Steps | h | Final bulk L2 error | Observed order |
|---:|---:|---:|---:|
| 8 | 0.125 | 2.88020651e-2 | — |
| 16 | 0.0625 | 1.46388240e-2 | 0.9764 |
| 32 | 0.03125 | 7.38104737e-3 | 0.9879 |
| 64 | 0.015625 | 3.70686273e-3 | 0.9936 |

This demonstrates the expected first-order trend in the tested range. Data and the exported convergence figure are in `runs/time_convergence_validation/`.

### Quadrature sensitivity

At N=16, epsilon=1e-3, K=20, with the same initial checkpoint:

| Solver points per axis | Final bulk L2 error, 65-point validation |
|---:|---:|
| 19 | 1.46388240e-2 |
| 37 | 1.46374637e-2 |
| 73 | 1.46373881e-2 |

Independent 129-point validation changed the 37- and 73-point results by about 5.5e-10. Time-discretization error dominates these representative runs. Artifacts are in `runs/quadrature_37_validation/` and `runs/quadrature_73_validation/`.

## Remaining limits

- The complete 32-trajectory convergence grid and the N=16384, epsilon=1e-5 Figure 5.1 configuration are provided but have not been executed. Their accuracy and runtimes are not claimed here.
- The quadratic boundary norm is the agreed interpretation; it is not literally the squared sum of norms printed in the manuscript. The modified regularization follows (4.4), not the conflicting matrix remark.
- Optimizer settings and the second refinement schedule are documented engineering defaults. Matching the paper's complete experimental environment is not established.
- Recorded defect ratios cannot certify theoretical assumptions involving unknown constants. Initialization error, quadrature, regularization, and finite neural expressivity can limit further time convergence.
- CPU behavior has been exercised. GPU-specific performance has not been tested.
