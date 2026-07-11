"""Shared pytest configuration for the repository test suite."""

import os
import platform
import sys
from importlib import metadata

import pytest

_PLATFORM_ENV = "ALIGN_TEST_PLATFORM"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--jax-platform",
        choices=("auto", "cpu", "mps"),
        default=os.environ.get(_PLATFORM_ENV, "cpu"),
        help=(
            "JAX platform for the test session (default: cpu; may also be set "
            f"with {_PLATFORM_ENV})"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    selected = config.getoption("--jax-platform")
    if selected == "auto":
        return
    if selected == "mps":
        _require_mps_environment()
        os.environ["JAX_PLATFORMS"] = "mps,cpu"
        os.environ.setdefault("JAX_MPS_ASYNC_DISPATCH", "0")
        return

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_VISIBLE_DEVICES"] = ""


def _require_mps_environment() -> None:
    if sys.platform != "darwin" or platform.machine().lower() not in {
        "aarch64",
        "arm64",
    }:
        raise pytest.UsageError("--jax-platform=mps requires Apple Silicon")
    try:
        metadata.version("jax-mps")
    except metadata.PackageNotFoundError as error:
        raise pytest.UsageError(
            "--jax-platform=mps requires `uv sync --extra dev --extra mps`"
        ) from error
