#!/usr/bin/env python3
"""Run and plot the alpha=1, K=20, float64 heat-equation convergence study.

    python make_convergence_plot.py
    python make_convergence_plot.py --resume
    python make_convergence_plot.py --plot-only

Defaults: eps=(0.1,0.01,0.001,0.0001), N=32,...,4096, T=1;
composite Simpson quadrature with 19 points/axis and 19 points/boundary edge;
independent 65-point/axis error quadrature. Outputs PDF, SVG, 600-dpi PNG,
CSV, exact configurations, shared initialization, and endpoint checkpoints.

Only endpoint errors are needed here. Physical steps are compiled in chunks
using the existing integrator kernels, with finite checks at every step. The
same frozen-Jacobian assembly and all 20 inner iterations are retained. This
avoids exporting hundreds of thousands of unnecessary diagnostic CSV rows.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rdpa_heat.config import RunConfig
from rdpa_heat.diagnostics import (
    environment_metadata, evaluate_state, load_checkpoint, save_checkpoint,
    write_csv, write_json,
)
from rdpa_heat.initialization import fit_initial_condition
from rdpa_heat.integrator import _assemble_step_kernel, _advance_step_kernel, simulate
from rdpa_heat.operators import values
from rdpa_heat.quadrature import build_quadrature

import jax
import jax.numpy as jnp
import numpy as np

EPSILONS = (0.1, 0.01, 0.001, 0.0001)
STEP_COUNTS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
ALPHA, ITERATIONS, FINAL_TIME = 1.0, 20, 1.0


def all_finite(tree):
    return jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)
    ]))


@partial(jax.jit, static_argnames=("count",))
def physical_chunk(theta, rule, h, epsilon, *, count=32):
    """Use production kernels; stop at the last valid complete physical step."""
    def advance(carry, _):
        current, active, completed = carry

        def take_step(q):
            system = _assemble_step_kernel(q, h, epsilon, ALPHA, rule, method="qr")
            result = _advance_step_kernel(q, system, iterations=ITERATIONS, method="qr")
            finite = all_finite(system) & all_finite(result)
            return jnp.where(finite, result.parameters, q), finite, jnp.stack((
                result.diagnostics.defects[-1],
                jnp.max(result.diagnostics.stationarity),
            ))

        candidate, finite, diagnostics = jax.lax.cond(
            active, take_step,
            lambda q: (q, jnp.array(False), jnp.full(2, jnp.nan)), current,
        )
        return (candidate, active & finite, completed + (active & finite)), diagnostics

    return jax.lax.scan(
        advance, (theta, jnp.array(True), jnp.array(0, dtype=jnp.int32)),
        xs=None, length=count,
    )


def verify_chunk(theta, rule, validation):
    """Independently check the optimized driver against public simulate()."""
    checks = []
    for epsilon in (0.1, 0.0001):
        config = RunConfig(final_time=0.02, steps=2, epsilon=epsilon, alpha=ALPHA,
                           iterations=ITERATIONS, quadrature_points=int(round(math.sqrt(len(rule.points)))))
        ordinary = simulate(theta, config, rule)
        (endpoint, finite, completed), _ = physical_chunk(theta, rule, config.h, epsilon, count=2)
        if not bool(finite) or int(completed) != 2:
            raise RuntimeError("Compiled driver failed its equivalence check")
        difference = float(jnp.max(jnp.abs(values(endpoint, validation.points)
                                          - values(ordinary.parameters[-1], validation.points))))
        if difference > 1e-9:
            raise RuntimeError(f"Compiled/public driver disagreement: {difference:g}")
        checks.append({"epsilon": epsilon, "max_field_difference": difference})
    return checks


def run_endpoint(theta0, rule, validation, epsilon, steps, output):
    theta, completed, peak_stationarity = theta0, 0, 0.0
    h = FINAL_TIME / steps
    started = last_update = time.perf_counter()
    while completed < steps:
        count = min(32, steps - completed)
        (theta, finite, advanced), diagnostics = physical_chunk(theta, rule, h, epsilon, count=count)
        completed += int(advanced)  # Synchronize before progress/timing.
        if not bool(finite):
            save_checkpoint(output / "failed_last_valid.npz", theta,
                            {"time": completed * h, "epsilon": epsilon, "steps": steps})
            raise FloatingPointError(f"eps={epsilon:g}, N={steps}: failed at step {completed + 1}")
        peak_stationarity = max(peak_stationarity, float(jnp.max(diagnostics[:, 1])))
        if time.perf_counter() - last_update >= 30:
            print(f"  eps={epsilon:g}, N={steps}: {completed}/{steps} steps", flush=True)
            last_update = time.perf_counter()
    metrics = evaluate_state(theta, FINAL_TIME, validation, ALPHA)
    row = {
        "epsilon": epsilon, "steps": steps, "h": h, "alpha": ALPHA,
        "gauss_newton_iterations": ITERATIONS, "dtype": "float64",
        **metrics, "last_defect": float(diagnostics[-1, 0]),
        "peak_stationarity": peak_stationarity,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not math.isfinite(row["bulk_l2"]) or row["bulk_l2"] <= 0:
        raise FloatingPointError("Invalid error measurement for logarithmic plotting")
    stem = f"eps_{epsilon:g}_n_{steps}"
    save_checkpoint(output / f"{stem}.npz", theta,
                    {"time": FINAL_TIME, "epsilon": epsilon, "steps": steps, "alpha": ALPHA})
    write_json(output / f"{stem}.json", row)
    return row


def add_orders(rows):
    rows = sorted(rows, key=lambda row: (-float(row["epsilon"]), int(row["steps"])))
    previous = {}
    for row in rows:
        epsilon = float(row["epsilon"])
        row.pop("observed_order", None)
        if epsilon in previous:
            last = previous[epsilon]
            row["observed_order"] = math.log(last["bulk_l2"] / row["bulk_l2"]) / math.log(last["h"] / row["h"])
        previous[epsilon] = row
    return rows


def render(rows, metadata, output):
    expected = {(epsilon, steps) for epsilon in EPSILONS for steps in metadata["steps"]}
    completed = {(float(row["epsilon"]), int(row["steps"])) for row in rows}
    if not completed or completed - expected:
        raise ValueError("Plot data must be a nonempty subset of the recorded sweep")
    partial_sweep = completed != expected
    cache = Path(tempfile.gettempdir()) / "rdpa_heat_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    style = {
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "axes.linewidth": 0.8, "lines.linewidth": 1.45,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(7.2, 5.25))
        figure.subplots_adjust(left=0.12, right=0.97, top=0.86 if partial_sweep else 0.91, bottom=0.25)
        palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
        for epsilon, color, marker in zip(EPSILONS, palette, ("o", "s", "^", "D")):
            group = sorted((r for r in rows if float(r["epsilon"]) == epsilon), key=lambda r: r["h"])
            if not group:
                continue
            axis.loglog([r["h"] for r in group], [r["bulk_l2"] for r in group],
                        color=color, marker=marker, markersize=5.4,
                        markerfacecolor="white", markeredgewidth=1.1,
                        label=rf"$\varepsilon = 10^{{{round(math.log10(epsilon))}}}$")
        hs = np.array(sorted({float(r["h"]) for r in rows}))
        # A slope-one guide, offset for readability; not a fitted error curve.
        axis.loglog(hs, 0.55 * math.pi * math.exp(-0.5) / 8 * hs,
                    color="0.25", linestyle="--", linewidth=1.15, label=r"$\mathcal{O}(h)$")
        initial_error = metadata["initial_metrics"]["bulk_l2"]
        axis.axhline(initial_error, color="0.5", linestyle=":", linewidth=1.2,
                     label=r"Initial $L^2$ error")
        title = "Heat equation: regularized parametric implicit Euler"
        if partial_sweep:
            title += f"\nPartial sweep: {len(completed)} of {len(expected)} trajectories"
        axis.set(xlabel=r"Time step $h = T/N$",
                 ylabel=r"$\|u_N-y(T)\|_{L^2(\Omega)}$",
                 title=title)
        axis.grid(which="major", color="0.88", linewidth=0.55)
        axis.grid(which="minor", color="0.94", linewidth=0.4)
        axis.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        axis.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.legend(loc="upper left", ncol=2, fontsize=9.5, framealpha=0.97,
                    edgecolor="0.8", handlelength=2.4, columnspacing=1.2)
        q, v = metadata["quadrature_points"], metadata["validation_points"]
        figure.text(0.5, 0.135,
                    rf"$\alpha=1$, $K=20$, $T=1$; float64; 153 parameters; augmented QR",
                    ha="center", fontsize=9)
        figure.text(0.5, 0.090,
                    rf"Composite Simpson: ${q}\times{q}={q*q}$ bulk points; {q} per boundary edge ({4*q} total)",
                    ha="center", fontsize=9)
        figure.text(0.5, 0.045,
                    rf"Independent $L^2$ error quadrature: composite Simpson, ${v}\times{v}={v*v}$ points",
                    ha="center", fontsize=9)
        stem = "convergence_alpha1_partial" if partial_sweep else "convergence_alpha1"
        for extension in ("pdf", "svg", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=600)
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/publication_alpha1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/default_initialization/initial.npz")
    parser.add_argument("--quad-points", type=int, default=19)
    parser.add_argument("--validation-points", type=int, default=65)
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEP_COUNTS))
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs with matching settings")
    parser.add_argument("--first-epsilon", type=float, choices=EPSILONS,
                        help="Run this epsilon first without changing the sweep or its saved settings")
    parser.add_argument("--plot-only", action="store_true", help="Re-render existing CSV using its recorded settings")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.plot_only:
        metadata = json.loads((output / "metadata.json").read_text())
        with (output / "convergence.csv").open() as handle:
            rows = [dict(row, epsilon=float(row["epsilon"]), h=float(row["h"]), bulk_l2=float(row["bulk_l2"]))
                    for row in csv.DictReader(handle)]
        render(rows, metadata, output)
        print(f"Figure exported to {output}", flush=True)
        return

    steps_list = sorted(set(args.steps))
    if len(steps_list) < 2:
        parser.error("Use at least two positive step counts for a convergence plot")
    config = RunConfig(alpha=ALPHA, iterations=ITERATIONS,
                       quadrature_points=args.quad_points, validation_points=args.validation_points,
                       sweep_steps=tuple(steps_list), sweep_epsilons=EPSILONS)
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("This experiment requires JAX double precision")
    rule, validation = build_quadrature(config), build_quadrature(config.validation_points)
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "src/rdpa_heat").glob("*.py")):
        source_hash.update(path.name.encode() + path.read_bytes())
    settings = {
        "epsilons": list(EPSILONS), "alpha": ALPHA, "iterations": ITERATIONS,
        "final_time": FINAL_TIME, "steps": steps_list, "dtype": "float64", "linear_solver": "qr",
        "quadrature_rule": "composite Simpson", "quadrature_points": args.quad_points,
        "bulk_evaluations": args.quad_points**2, "boundary_evaluations": 4 * args.quad_points,
        "validation_points": args.validation_points, "solver_source_sha256": source_hash.hexdigest(),
    }
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error(f"{output} is not empty; choose another directory or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and (output / "metadata.json").exists():
        metadata = json.loads((output / "metadata.json").read_text())
        if any(metadata.get(key) != value for key, value in settings.items()):
            raise ValueError("Resume settings/source differ from the recorded experiment")
        theta0, _ = load_checkpoint(output / "initial.npz")
    else:
        if args.checkpoint.exists():
            theta0, provenance = load_checkpoint(args.checkpoint)
            if provenance.get("time", 0.0) != 0.0:
                raise ValueError("Use a time-zero initialization checkpoint")
            initialization_source = str(args.checkpoint.resolve())
        else:
            print("Computing Adam + RK4 initial parameters...", flush=True)
            initial = fit_initial_condition(config, rule)
            theta0 = initial.theta
            write_json(output / "initialization_history.json", initial.history)
            initialization_source = "computed with documented default Adam/RK4 schedule"
            provenance = {"config": config.to_dict()}
        initial_metrics = evaluate_state(theta0, 0.0, validation, ALPHA)
        metadata = {
            **settings, "environment": environment_metadata(), "initial_metrics": initial_metrics,
            "initialization_source": initialization_source, "initialization_provenance": provenance,
            "initial_parameters_sha256": hashlib.sha256(np.asarray(theta0).tobytes()).hexdigest(),
            "boundary_norm": "integral(u^2) + alpha * integral(tangential_derivative(u)^2)",
        }
        save_checkpoint(output / "initial.npz", theta0, {"time": 0.0, "source": initialization_source})
        write_json(output / "metadata.json", metadata)
    print(f"float64; alpha=1; K=20; Simpson {args.quad_points}x{args.quad_points} bulk, "
          f"{args.quad_points}/edge; validation {args.validation_points}x{args.validation_points}", flush=True)
    print("Verifying compiled driver against the public integrator...", flush=True)
    write_json(output / "driver_verification.json", verify_chunk(theta0, rule, validation))
    endpoints = output / "endpoints"
    endpoints.mkdir(exist_ok=True)
    # Load every completed record before scheduling new work. In particular,
    # prioritizing another epsilon must not temporarily remove old CSV rows.
    saved = {}
    if args.resume:
        for epsilon in EPSILONS:
            for steps in steps_list:
                record = endpoints / f"eps_{epsilon:g}_n_{steps}.json"
                if record.exists():
                    saved[(epsilon, steps)] = json.loads(record.read_text())
    order = list(EPSILONS)
    if args.first_epsilon is not None:
        order.remove(args.first_epsilon)
        order.insert(0, args.first_epsilon)

    def save_progress(state, current=None):
        if saved:
            write_csv(output / "convergence.csv", add_orders(list(saved.values())))
        write_json(output / "status.json", {
            "status": state, "completed_runs": len(saved),
            "total_runs": len(EPSILONS) * len(steps_list),
            "epsilon_order": order, "current_run": current,
            "completed": {str(eps): sorted(n for e, n in saved if e == eps) for eps in EPSILONS},
        })

    save_progress("running")
    current = None
    try:
        for epsilon in order:
            computed = False
            for steps in steps_list:
                if (epsilon, steps) in saved:
                    print(f"Reusing eps={epsilon:g}, N={steps}", flush=True)
                    continue
                current = {"epsilon": epsilon, "steps": steps}
                save_progress("running", current)
                row = run_endpoint(theta0, rule, validation, epsilon, steps, endpoints)
                print(f"eps={epsilon:g}, N={steps:4d}, h={row['h']:.6g}, "
                      f"L2={row['bulk_l2']:.8e}, {row['elapsed_seconds']:.1f}s", flush=True)
                saved[(epsilon, steps)] = row
                computed = True
                save_progress("running")
            if computed:
                render(list(saved.values()), metadata, output)
    except KeyboardInterrupt:
        save_progress("paused", current)
        if saved:
            render(list(saved.values()), metadata, output)
        print(f"\nPaused; {len(saved)} completed runs saved. Resume with --resume.", flush=True)
        return
    rows = add_orders(list(saved.values()))
    render(rows, metadata, output)
    caption = (
        "Temporal convergence of the regularized parametric implicit Euler method (equation 4.4) "
        "for the homogeneous-Dirichlet heat equation on [-pi,pi]^2, with initial data "
        "cos(x1/2)cos(x2/2). Absolute L2 errors at T=1 are shown for epsilon = "
        "0.1, 0.01, 0.001, 0.0001. All runs use alpha=1, 20 modified Gauss-Newton "
        "iterations per physical step, float64 arithmetic, and augmented QR least squares. "
        f"Composite Simpson quadrature uses {args.quad_points} points per axis "
        f"({args.quad_points**2} bulk evaluations) and {args.quad_points} points per edge "
        f"({4*args.quad_points} edge-associated evaluations). Independent error integration "
        f"uses {args.validation_points} points per axis. The dotted line is the shared initial "
        "L2 error; the dashed line is a slope-one reference. The boundary norm is "
        "integral(u^2) + alpha*integral((tangential derivative u)^2).\n"
    )
    (output / "caption.txt").write_text(caption)
    write_json(output / "status.json", {"status": "complete", "runs": len(rows)})
    print(f"Saved publication figures and numerical data to {output}", flush=True)


if __name__ == "__main__":
    main()
