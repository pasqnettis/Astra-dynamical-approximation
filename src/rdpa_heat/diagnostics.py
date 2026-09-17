"""Independent physical error measurements and portable experiment artifacts."""

import csv
import importlib.metadata
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .model import PARAMETER_COUNT, parameter_layout
from .operators import values, tangential_values
from .problem import exact_solution


@jax.jit
def _state_metrics(theta, time, rule, alpha):
    predicted = values(theta, rule.points)
    exact = exact_solution(time, rule.points)
    error = predicted - exact
    bulk_l2 = jnp.sqrt(jnp.sum(rule.weights * error**2))
    reference_l2 = jnp.sqrt(jnp.sum(rule.weights * exact**2))
    boundary = values(theta, rule.boundary_points)
    tangent = tangential_values(theta, rule.boundary_points, rule.tangents)
    boundary_squared = jnp.sum(rule.boundary_weights * boundary**2)
    tangent_squared = jnp.sum(rule.boundary_weights * tangent**2)
    return {
        "time": time,
        "bulk_l2": bulk_l2,
        "relative_l2": bulk_l2 / jnp.maximum(reference_l2, jnp.finfo(jnp.float64).tiny),
        "max_abs_error": jnp.max(jnp.abs(error)),
        "solution_l2": jnp.sqrt(jnp.sum(rule.weights * predicted**2)),
        "boundary_l2": jnp.sqrt(boundary_squared),
        "boundary_tangential": jnp.sqrt(tangent_squared),
        "boundary_weighted": jnp.sqrt(boundary_squared + alpha * tangent_squared),
    }


def evaluate_state(theta, time, rule, alpha=0.2) -> dict[str, float]:
    """Evaluate errors on a rule separate from the solver's quadrature."""
    return {name: float(value) for name, value in _state_metrics(theta, time, rule, alpha).items()}


def trajectory_metrics(result, rule, alpha) -> list[dict]:
    records = []
    for index, (time, theta) in enumerate(zip(result.times, result.parameters)):
        row = {"step": index, **evaluate_state(theta, time, rule, alpha)}
        if index:
            defects = np.asarray(result.diagnostics.defects[index - 1])
            ratios = np.asarray(result.diagnostics.defect_ratios[index - 1])
            row.update(
                last_defect=float(defects[-1]), max_defect=float(defects.max()),
                summed_defect_ratio=float(ratios.sum()),
                max_stationarity=float(np.asarray(result.diagnostics.stationarity[index - 1]).max()),
            )
        records.append(row)
    return records


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    names = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def environment_metadata() -> dict:
    versions = {}
    for name in ("jax", "jaxlib", "optax", "numpy", "scipy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "parameter_count": PARAMETER_COUNT,
        "boundary_norm": "integral(u^2) + alpha * integral(tangential_derivative(u)^2)",
        "method": "equation_4.4_modified_regularization",
    }


def save_checkpoint(path, theta, metadata=None):
    theta = np.asarray(theta)
    if theta.shape != (PARAMETER_COUNT,) or theta.dtype != np.float64 or not np.isfinite(theta).all():
        raise ValueError("Checkpoint requires 153 finite float64 parameters")
    np.savez_compressed(
        path, theta=theta, format_version=np.array(1),
        problem=np.array("heat-square-cosine-v1"),
        parameter_layout=np.array(json.dumps(parameter_layout(), sort_keys=True)),
        metadata=np.array(json.dumps(metadata or {}, allow_nan=False)),
    )


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 1 or str(archive["problem"]) != "heat-square-cosine-v1":
            raise ValueError("Checkpoint format/problem is incompatible")
        if json.loads(str(archive["parameter_layout"])) != parameter_layout():
            raise ValueError("Checkpoint parameter layout is incompatible")
        theta = archive["theta"].copy()
        metadata = json.loads(str(archive["metadata"]))
    if theta.shape != (PARAMETER_COUNT,) or theta.dtype != np.float64 or not np.isfinite(theta).all():
        raise ValueError("Checkpoint must contain 153 finite float64 parameters")
    return jnp.asarray(theta), metadata


def save_simulation(path, result):
    arrays = {"times": np.asarray(result.times), "parameters": np.asarray(result.parameters)}
    arrays.update({name: np.asarray(value) for name, value in result.diagnostics._asdict().items()})
    arrays["component_names"] = np.array(["bulk", "boundary", "accumulated_regularization", "increment_regularization"])
    arrays["parameter_layout"] = np.array(json.dumps(parameter_layout(), sort_keys=True))
    np.savez_compressed(path, **arrays)


def inner_iteration_rows(result):
    for step in range(len(result.times) - 1):
        d = result.diagnostics
        for k in range(d.defects.shape[1]):
            components = np.asarray(d.components[step, k])
            yield {
                "step": step + 1, "time": float(result.times[step + 1]), "iteration": k,
                "defect": float(d.defects[step, k]),
                "bulk_squared": float(components[0]), "boundary_squared": float(components[1]),
                "accumulated_regularization_squared": float(components[2]),
                "increment_regularization_squared": float(components[3]),
                "increment_norm": float(d.increment_norms[step, k]),
                "stationarity": float(d.stationarity[step, k]),
                "defect_ratio": float(d.defect_ratios[step, k]),
            }
