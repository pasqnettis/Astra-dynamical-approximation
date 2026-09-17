"""Reproducible initialization, trajectories, and shared-checkpoint sweeps."""

from dataclasses import replace
from pathlib import Path
import time

from .config import RunConfig
from .diagnostics import (
    environment_metadata, evaluate_state, inner_iteration_rows, load_checkpoint,
    save_checkpoint, save_simulation, trajectory_metrics, write_csv, write_json,
)
from .initialization import fit_initial_condition, InitializationFailure
from .integrator import simulate, SimulationFailure
from .quadrature import build_quadrature


def prepare_output(output, config):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "environment.json", environment_metadata())
    return output


def _initial_state(config, rule, output, checkpoint=None, progress=None):
    validation = build_quadrature(config.validation_points)
    if checkpoint is not None:
        theta, provenance = load_checkpoint(checkpoint)
        if provenance.get("time", 0.0) != 0.0:
            raise ValueError("A physical-time checkpoint cannot be used as this experiment's initial condition; use initial.npz")
        summary = {
            "source": str(Path(checkpoint).resolve()), "source_metadata": provenance,
            "metrics": evaluate_state(theta, 0.0, validation, config.alpha),
        }
    else:
        start = time.perf_counter()
        try:
            result = fit_initial_condition(config, rule, progress=progress)
        except InitializationFailure as error:
            save_checkpoint(output / "initialization_failed_last_valid.npz", error.theta,
                            {"status": "initialization_failed", "message": str(error)})
            write_json(output / "initialization_history.json", error.history)
            write_json(output / "failure.json", {"stage": "initialization", "message": str(error)})
            raise
        theta = result.theta
        write_json(output / "initialization_history.json", result.history)
        write_csv(output / "initialization_history.csv", result.history)
        summary = {
            "initial_error": result.initial_error, "final_error": result.final_error,
            "elapsed_seconds": time.perf_counter() - start,
            "metrics": evaluate_state(theta, 0.0, validation, config.alpha),
        }
    write_json(output / "initialization.json", summary)
    save_checkpoint(output / "initial.npz", theta, {"config": config.to_dict(), "summary": summary})
    return theta, summary


def initialize_experiment(config: RunConfig, output, progress=None):
    output = prepare_output(output, config)
    rule = build_quadrature(config.quadrature_points)
    _, summary = _initial_state(config, rule, output, progress=progress)
    write_json(output / "status.json", {"status": "complete", "command": "initialize"})
    return summary


def _run_trajectory(theta, config, rule, output, plots, progress):
    try:
        result = simulate(theta, config, rule, progress=progress)
    except SimulationFailure as error:
        partial = error.partial_result
        save_simulation(output / "partial_trajectory.npz", partial)
        if len(partial.parameters):
            save_checkpoint(output / "last_valid.npz", partial.parameters[-1], {
                "status": "simulation_failed", "time": float(partial.times[-1]),
                "failed_step": error.failed_step,
            })
        write_json(output / "failure.json", {"stage": "simulation", "step": error.failed_step, "message": str(error)})
        raise
    validation = build_quadrature(config.validation_points)
    metrics = trajectory_metrics(result, validation, config.alpha)
    save_simulation(output / "trajectory.npz", result)
    save_checkpoint(output / "final.npz", result.parameters[-1], {"config": config.to_dict(), "time": float(result.times[-1])})
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "inner_iterations.csv", inner_iteration_rows(result))
    summary = {
        "status": "complete", "steps": config.steps, "h": config.h,
        "epsilon": config.epsilon, "alpha": config.alpha,
        "initial_metrics": metrics[0], "final_metrics": metrics[-1],
        "integration_seconds": result.elapsed_seconds,
    }
    (output / "report.md").write_text(
        "# Heat-equation run\n\n"
        f"Equation (4.4), {config.linear_solver} least squares, alpha={config.alpha:g}, "
        f"epsilon={config.epsilon:g}, N={config.steps}, K={config.iterations}.\n\n"
        f"Initial independent bulk L2 error: {metrics[0]['bulk_l2']:.8e}.\n\n"
        f"Final independent bulk L2 error: {metrics[-1]['bulk_l2']:.8e}.\n\n"
        f"Final weighted boundary norm: {metrics[-1]['boundary_weighted']:.8e}.\n\n"
        f"Integration runtime (including compilation): {result.elapsed_seconds:.3f} s.\n\n"
        "The boundary norm uses alpha on the squared tangential seminorm. "
        "Defect ratios are diagnostics, not certified theoretical bounds.\n"
    )
    if plots:
        from .plotting import plot_trajectory
        plot_trajectory(result, metrics, validation, output)
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"status": "complete", "command": "run"})
    return summary


def run_experiment(config: RunConfig, output, checkpoint=None, plots=True,
                   initialization_progress=None, progress=None):
    output = prepare_output(output, config)
    rule = build_quadrature(config.quadrature_points)
    theta, _ = _initial_state(config, rule, output, checkpoint, initialization_progress)
    return _run_trajectory(theta, config, rule, output, plots, progress)


def sweep_experiment(config: RunConfig, output, checkpoint=None, plots=True,
                     initialization_progress=None, progress=None):
    output = prepare_output(output, config)
    rule = build_quadrature(config.quadrature_points)
    theta, initialization = _initial_state(config, rule, output, checkpoint, initialization_progress)
    rows = []
    for epsilon in config.sweep_epsilons:
        previous = None
        for steps in sorted(set(config.sweep_steps)):
            run_config = replace(config, epsilon=epsilon, steps=steps)
            child = prepare_output(output / f"eps_{epsilon:.12g}_n_{steps}", run_config)
            # All trajectories receive this exact same vector; no repeated training.
            summary = _run_trajectory(theta, run_config, rule, child, False, progress)
            row = {"epsilon": epsilon, "steps": steps, "h": run_config.h, **summary["final_metrics"]}
            if previous is not None and row["bulk_l2"] > 0 and previous["bulk_l2"] > 0:
                import math
                row["observed_order"] = math.log(previous["bulk_l2"] / row["bulk_l2"]) / math.log(previous["h"] / row["h"])
            rows.append(row)
            previous = row
            write_csv(output / "convergence.csv", rows)
    if plots:
        from .plotting import plot_convergence
        plot_convergence(rows, initialization["metrics"]["bulk_l2"], output)
    write_json(output / "status.json", {"status": "complete", "command": "sweep", "runs": len(rows)})
    return rows
