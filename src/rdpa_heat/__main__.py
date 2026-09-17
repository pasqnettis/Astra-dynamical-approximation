"""Command-line entry point: initialize, run, and convergence sweep."""

import argparse
from dataclasses import replace
from datetime import datetime
import sys

from .config import RefinementPass, load_config, override_config


def _refinement(value):
    try:
        steps, epsilon = value.split(":")
        return RefinementPass(int(steps), float(epsilon))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Expected positive STEPS:EPSILON, e.g. 100:1e-4") from error


def make_parser():
    parser = argparse.ArgumentParser(description="JAX regularized heat solver, equation (4.4)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("initialize", "run", "sweep"):
        command = commands.add_parser(name)
        command.add_argument("--config", help="TOML configuration; omitted fields use documented defaults")
        command.add_argument("--output", help="New or empty result directory")
        command.add_argument("--alpha", type=float)
        command.add_argument("--epsilon", type=float)
        command.add_argument("--steps", type=int)
        command.add_argument("--final-time", type=float)
        command.add_argument("--iterations", type=int)
        command.add_argument("--quadrature-points", type=int)
        command.add_argument("--validation-points", type=int)
        command.add_argument("--seed", type=int)
        command.add_argument("--linear-solver", choices=("qr", "normal"))
        command.add_argument("--adam-steps", type=int)
        command.add_argument("--learning-rate", type=float)
        command.add_argument("--learning-rate-late", type=float)
        command.add_argument("--decay-step", type=int)
        command.add_argument("--validation-every", type=int)
        refinement = command.add_mutually_exclusive_group()
        refinement.add_argument("--init-pass", type=_refinement, action="append", help="Repeat for each RK4 pass; replaces configured schedule")
        refinement.add_argument("--no-refinement", action="store_true")
        command.add_argument("--quiet", action="store_true")
        if name != "initialize":
            command.add_argument("--checkpoint", help="Initialization .npz checkpoint to reuse")
            command.add_argument("--no-plots", action="store_true")
        if name == "sweep":
            command.add_argument("--sweep-steps", type=int, nargs="+")
            command.add_argument("--sweep-epsilons", type=float, nargs="+")
    return parser


def _initialization_progress(event):
    # The detailed history is persisted; console output stays compact.
    stage = event.get("stage", "initialization")
    step = event.get("step", event.get("update", event.get("iteration", 0)))
    if stage != "adam" or step % 1000 == 0:
        print(f"Initialization: {event}", flush=True)


def _step_progress(completed, total, result):
    if completed == 1 or completed == total or completed % max(1, total // 10) == 0:
        print(f"Heat step {completed}/{total}; defect={float(result.diagnostics.defects[-1]):.3e}", flush=True)


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        fields = ("alpha", "epsilon", "steps", "final_time", "iterations", "quadrature_points", "validation_points", "seed", "linear_solver")
        config = override_config(config, **{name: getattr(args, name) for name in fields})
        init_fields = ("adam_steps", "learning_rate", "learning_rate_late", "decay_step", "validation_every")
        init_overrides = {name: getattr(args, name) for name in init_fields if getattr(args, name) is not None}
        if args.no_refinement:
            init_overrides["passes"] = ()
        elif args.init_pass is not None:
            init_overrides["passes"] = tuple(args.init_pass)
        config = replace(config, initialization=replace(config.initialization, **init_overrides))
        if args.command == "sweep":
            config = override_config(config,
                sweep_steps=tuple(args.sweep_steps) if args.sweep_steps is not None else None,
                sweep_epsilons=tuple(args.sweep_epsilons) if args.sweep_epsilons is not None else None)
        output = args.output or f"runs/{args.command}-{datetime.now():%Y%m%d-%H%M%S-%f}"
        from .experiments import initialize_experiment, run_experiment, sweep_experiment
        if args.command == "initialize":
            initialize_experiment(config, output, progress=None if args.quiet else _initialization_progress)
        else:
            function = run_experiment if args.command == "run" else sweep_experiment
            function(config, output, checkpoint=args.checkpoint, plots=not args.no_plots,
                     initialization_progress=None if args.quiet else _initialization_progress,
                     progress=None if args.quiet else _step_progress)
        print(f"Results saved to {output}")
        return 0
    except (ValueError, TypeError, OSError, FloatingPointError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
