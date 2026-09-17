"""Headless, exportable figures; never imported inside numerical kernels."""

from pathlib import Path
import os
import tempfile
import numpy as np

# CLI runs may execute with a read-only home directory. Keep plotting caches
# writable without requiring access to user-level configuration directories.
_cache_root = Path(tempfile.gettempdir()) / "rdpa_heat_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .operators import values
from .problem import exact_solution


def _save(figure, output, name):
    for extension in ("png", "pdf"):
        figure.savefig(Path(output) / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trajectory(result, metrics, rule, output):
    times = np.asarray(result.times)
    indices = list(dict.fromkeys(int(np.argmin(abs(times - t))) for t in (0, times[-1] / 2, times[-1])))
    n = int(round(np.sqrt(len(rule.points))))
    figure, axes = plt.subplots(2, len(indices), figsize=(4.2 * len(indices), 7), squeeze=False)
    for column, index in enumerate(indices):
        approximation = np.asarray(values(result.parameters[index], rule.points)).reshape(n, n)
        exact = np.asarray(exact_solution(times[index], rule.points)).reshape(n, n)
        for row, array in enumerate((approximation, abs(approximation - exact))):
            artist = axes[row, column].imshow(array.T, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi), cmap="viridis")
            axes[row, column].set(xlabel="$x_1$", ylabel="$x_2$", title=f"{'Solution' if row == 0 else 'Absolute error'}, t={times[index]:.6g}")
            figure.colorbar(artist, ax=axes[row, column], shrink=0.8)
    figure.tight_layout()
    _save(figure, output, "snapshots")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, label in (("bulk_l2", "Bulk L2 error"), ("boundary_weighted", "Weighted boundary norm")):
        axes[0].semilogy(times, [max(row[name], np.finfo(float).tiny) for row in metrics], label=label)
    axes[0].set(xlabel="Physical time", ylabel="Error / norm")
    axes[0].legend()
    if len(times) > 1:
        axes[1].semilogy(times[1:], np.maximum(np.asarray(result.diagnostics.defects)[:, -1], np.finfo(float).tiny))
    axes[1].set(xlabel="Physical time", ylabel="Final inner-iteration defect")
    figure.tight_layout()
    _save(figure, output, "diagnostics")


def plot_convergence(rows, initial_error, output):
    figure, axis = plt.subplots(figsize=(7, 5))
    for epsilon in sorted({row["epsilon"] for row in rows}, reverse=True):
        group = sorted((r for r in rows if r["epsilon"] == epsilon), key=lambda r: r["h"])
        axis.loglog([r["h"] for r in group], [r["bulk_l2"] for r in group], "o-", label=f"epsilon={epsilon:g}")
    if initial_error > 0:
        axis.axhline(initial_error, color="gray", linestyle=":", label="Initial bulk L2 error")
    if rows:
        hs = np.array(sorted({row["h"] for row in rows}))
        scale = max(rows[0]["bulk_l2"], np.finfo(float).tiny) / rows[0]["h"]
        axis.loglog(hs, hs * scale, "k--", alpha=0.5, label="O(h) reference")
    axis.set(xlabel="Physical time step h", ylabel="Bulk L2 error at final time")
    axis.legend()
    figure.tight_layout()
    _save(figure, output, "convergence")
