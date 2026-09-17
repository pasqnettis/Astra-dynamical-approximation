"""Validated configuration; numerical defaults distinguish paper and engineering choices."""

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
import tomllib


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class RefinementPass:
    steps: int
    epsilon: float

    def __post_init__(self):
        _integer("refinement steps", self.steps)
        _positive("refinement epsilon", self.epsilon)


@dataclass(frozen=True)
class InitializationConfig:
    adam_steps: int = 20_000
    learning_rate: float = 1e-3
    learning_rate_late: float = 1e-4
    decay_step: int = 10_000
    validation_every: int = 100
    passes: tuple[RefinementPass, ...] = field(default_factory=lambda: (
        RefinementPass(100, 1e-4), RefinementPass(200, 1e-5),
    ))

    def __post_init__(self):
        _integer("adam_steps", self.adam_steps, 0)
        _integer("decay_step", self.decay_step, 0)
        _integer("validation_every", self.validation_every)
        _positive("learning_rate", self.learning_rate)
        _positive("learning_rate_late", self.learning_rate_late)
        if not isinstance(self.passes, tuple) or not all(isinstance(p, RefinementPass) for p in self.passes):
            raise ValueError("passes must be a tuple of RefinementPass objects")


@dataclass(frozen=True)
class RunConfig:
    final_time: float = 1.0
    steps: int = 256
    epsilon: float = 1e-3
    alpha: float = 0.2
    iterations: int = 20
    quadrature_points: int = 19
    validation_points: int = 65
    seed: int = 0
    linear_solver: str = "qr"
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    sweep_steps: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096)
    sweep_epsilons: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)

    def __post_init__(self):
        for name in ("final_time", "epsilon", "alpha"):
            _positive(name, getattr(self, name))
        for name in ("steps", "iterations"):
            _integer(name, getattr(self, name))
        for name in ("quadrature_points", "validation_points"):
            n = getattr(self, name)
            _integer(name, n, 3)
            if n % 2 != 1:
                raise ValueError(f"{name} must be odd for Simpson quadrature")
        _integer("seed", self.seed, 0)
        if self.linear_solver not in ("qr", "normal"):
            raise ValueError("linear_solver must be 'qr' or 'normal'")
        if not isinstance(self.initialization, InitializationConfig):
            raise ValueError("initialization must be an InitializationConfig")
        if not self.sweep_steps or not self.sweep_epsilons:
            raise ValueError("sweep steps and epsilons cannot be empty")
        for n in self.sweep_steps:
            _integer("sweep steps", n)
        for eps in self.sweep_epsilons:
            _positive("sweep epsilon", eps)

    @property
    def h(self) -> float:
        return self.final_time / self.steps

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> RunConfig:
    """Load TOML; reject misspelled/unknown fields instead of silently ignoring them."""
    if path is None:
        return RunConfig()
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    allowed = {"simulation", "initialization", "sweep"}
    if set(data) - allowed:
        raise ValueError(f"Unknown configuration sections: {sorted(set(data) - allowed)}")
    init_data = dict(data.get("initialization", {}))
    if "passes" in init_data:
        init_data["passes"] = tuple(RefinementPass(**p) for p in init_data["passes"])
    init = InitializationConfig(**init_data)
    kwargs = dict(data.get("simulation", {}))
    sweep = dict(data.get("sweep", {}))
    if set(sweep) - {"steps", "epsilons"}:
        raise ValueError("Only steps and epsilons are accepted in [sweep]")
    if "steps" in sweep:
        kwargs["sweep_steps"] = tuple(sweep["steps"])
    if "epsilons" in sweep:
        kwargs["sweep_epsilons"] = tuple(sweep["epsilons"])
    return RunConfig(**kwargs, initialization=init)


def override_config(config: RunConfig, **overrides) -> RunConfig:
    """Apply explicit (non-None) CLI overrides using the same validation as TOML."""
    return replace(config, **{key: value for key, value in overrides.items() if value is not None})
