"""Pytest fixtures and environment tweaks."""

import os

# Force CPU execution inside tests so importing align doesn't require CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_VISIBLE_DEVICES", "")
