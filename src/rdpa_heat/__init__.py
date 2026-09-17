"""Regularized dynamical parametric approximation of the square heat equation."""

import jax

# This must precede construction of every numerical array in the package.
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
