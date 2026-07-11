"""Tests for the selectable JAX test platform."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_selected_jax_platform_executes_array_program(
    pytestconfig: pytest.Config,
) -> None:
    selected = pytestconfig.getoption("--jax-platform")
    devices = jax.devices()
    assert devices
    if selected != "auto":
        assert devices[0].platform == selected

    values = jnp.arange(4, dtype=jnp.float32)
    result = jax.jit(lambda x: jnp.vdot(x, x))(values)
    np.testing.assert_allclose(np.asarray(result), 14.0)
