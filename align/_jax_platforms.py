"""Utility functions to configure the JAX backend before importing JAX."""

import os
import shutil
import subprocess


def configure_jax_platforms(
    preference: str | None = None, *, allow_preallocation: bool = False
) -> str:
    """Set JAX platform env vars before importing JAX.

    preference: "cpu" to force CPU-only, "gpu" to prefer GPU and fall back to CPU.
    allow_preallocation: when False, default to disabling XLA preallocation on GPU.
    Returns the normalized preference applied ("cpu" or "gpu").
    """

    if preference is None and os.environ.get("JAX_PLATFORMS"):
        _normalize_jax_platform_env()
        current = os.environ.get("JAX_PLATFORMS", "").lower()
        return "cpu" if current.startswith("cpu") else "gpu"

    pref = (preference or "").strip().lower()
    if pref not in {"cpu", "gpu"}:
        pref = "gpu"

    env_updates: dict[str, str] = {}
    if pref == "cpu":
        env_updates["JAX_PLATFORMS"] = "cpu"
        env_updates["CUDA_VISIBLE_DEVICES"] = ""
        env_updates["JAX_VISIBLE_DEVICES"] = ""
    else:
        if _looks_like_cuda_system():
            env_updates["JAX_PLATFORMS"] = "gpu,cpu"
            if not allow_preallocation:
                env_updates.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        else:
            env_updates["JAX_PLATFORMS"] = "cpu"
            pref = "cpu"

    env_updates["ALIGN_PLATFORM_CHOICE"] = pref
    os.environ.update(env_updates)
    _normalize_jax_platform_env()
    return pref


def _normalize_jax_platform_env() -> None:
    requested = os.environ.get("JAX_PLATFORMS")
    if not requested:
        return

    platforms = [chunk.strip() for chunk in requested.split(",") if chunk.strip()]
    if not platforms:
        return

    if not _looks_like_cuda_system():
        return

    updated = False
    normalized: list[str] = []
    for entry in platforms:
        if entry.lower() == "gpu":
            normalized.append("cuda")
            updated = True
        else:
            normalized.append(entry)

    if updated:
        os.environ["JAX_PLATFORMS"] = ",".join(normalized)


def _looks_like_cuda_system() -> bool:
    if os.environ.get("ROCR_VISIBLE_DEVICES"):
        return False
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env is not None and cuda_env.strip() != "":
        return True
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        proc = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=1)
        if proc.returncode != 0:
            return False
        return "GPU" in (proc.stdout or "")
    except Exception:
        return False
