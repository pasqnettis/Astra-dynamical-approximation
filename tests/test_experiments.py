"""Validate configuration, artifact round trips, and numerical-failure reports."""

from dataclasses import replace
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from rdpa_heat.config import InitializationConfig, RunConfig, load_config
from rdpa_heat.diagnostics import load_checkpoint, save_checkpoint
from rdpa_heat import experiments, initialization
from rdpa_heat.model import initialize_parameters


@pytest.mark.parametrize("changes", [
    {"alpha": 0.0}, {"epsilon": float("nan")}, {"steps": 2.5},
    {"quadrature_points": 20}, {"validation_points": 2}, {"linear_solver": "inverse"},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        RunConfig(**changes)


def test_config_files_load_and_unknown_fields_fail(tmp_path):
    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "configs/baseline.toml") == RunConfig()
    assert load_config(root / "configs/figure_5_1.toml").steps == 16384
    assert len(load_config(root / "configs/convergence.toml").sweep_steps) == 8
    invalid = tmp_path / "invalid.toml"
    invalid.write_text("[simulation]\nalhpa = 0.2\n")
    with pytest.raises(TypeError):
        load_config(invalid)


def test_quadrature_accepts_specified_configuration_interface():
    from rdpa_heat.quadrature import build_quadrature

    rule = build_quadrature(config=RunConfig(quadrature_points=5))
    assert rule.points.shape == (25, 2)
    assert rule.boundary_points.shape == (20, 2)


def test_checkpoint_roundtrip_and_layout_validation(tmp_path):
    theta = initialize_parameters(9)
    checkpoint = tmp_path / "initial.npz"
    save_checkpoint(checkpoint, theta, {"seed": 9})
    restored, metadata = load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored, theta)
    assert metadata == {"seed": 9}
    with np.load(checkpoint, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["parameter_layout"] = np.array("[]")
    np.savez(checkpoint, **arrays)
    with pytest.raises(ValueError, match="layout"):
        load_checkpoint(checkpoint)


def test_nonfinite_diagnostics_preserve_serializable_failure_context(tmp_path, monkeypatch):
    theta = initialize_parameters(0)

    def fail(*args, **kwargs):
        history = [{"stage": "valid", "bulk_error": 1.0}]
        initialization._record(history, {"stage": "bad", "bulk_error": float("inf")}, theta, None)

    monkeypatch.setattr(experiments, "fit_initial_condition", fail)
    output = tmp_path / "failed"
    with pytest.raises(initialization.InitializationFailure, match="Nonfinite"):
        experiments.initialize_experiment(RunConfig(), output)
    assert json.loads((output / "failure.json").read_text())["stage"] == "initialization"
    history = json.loads((output / "initialization_history.json").read_text())
    assert history[-1]["bulk_error"] == "inf"
    restored, _ = load_checkpoint(output / "initialization_failed_last_valid.npz")
    np.testing.assert_array_equal(restored, theta)


def test_final_checkpoint_is_not_silently_reused_at_time_zero(tmp_path):
    checkpoint = tmp_path / "final.npz"
    save_checkpoint(checkpoint, initialize_parameters(0), {"time": 1.0})
    with pytest.raises(ValueError, match="physical-time"):
        experiments.run_experiment(RunConfig(), tmp_path / "bad_restart", checkpoint, plots=False)


def test_results_are_not_overwritten(tmp_path):
    (tmp_path / "valuable.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        experiments.prepare_output(tmp_path, RunConfig())
    assert (tmp_path / "valuable.txt").read_text() == "keep"


def test_short_experiment_writes_consistent_artifacts(tmp_path):
    checkpoint = tmp_path / "initial.npz"
    save_checkpoint(checkpoint, initialize_parameters(1))
    config = RunConfig(steps=2, iterations=2, epsilon=0.2, final_time=0.02,
                       quadrature_points=3, validation_points=5,
                       initialization=InitializationConfig(adam_steps=0, passes=()))
    output = tmp_path / "experiment"
    summary = experiments.run_experiment(config, output, checkpoint, plots=False)
    assert summary["status"] == "complete"
    assert summary["final_metrics"]["time"] == config.final_time
    with np.load(output / "trajectory.npz", allow_pickle=False) as data:
        assert data["parameters"].shape == (3, 153)
        assert data["components"].shape == (2, 2, 4)
        np.testing.assert_allclose(data["components"].sum(-1), data["defects"]**2)
    assert (output / "metrics.csv").is_file()
    assert (output / "inner_iterations.csv").is_file()
    assert (output / "report.md").is_file()
