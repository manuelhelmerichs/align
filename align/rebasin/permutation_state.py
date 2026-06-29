"""Permutation state and projection utilities."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve_lap_maximize(cost: np.ndarray) -> np.ndarray:
    """Solve a square linear assignment problem maximizing ``cost``."""

    mat = np.ascontiguousarray(cost, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"LAP cost must be square, got shape {mat.shape}.")
    row_ind, col_ind = linear_sum_assignment(-mat)
    perm = np.zeros_like(mat, dtype=np.float64)
    perm[row_ind, col_ind] = 1.0
    if np.issubdtype(cost.dtype, np.floating) and cost.dtype != np.float64:
        return perm.astype(cost.dtype, copy=False)
    return perm


def _validate_indices(indices: np.ndarray, size: int) -> np.ndarray:
    idx = np.asarray(indices)
    if idx.ndim != 1:
        raise ValueError(f"Permutation indices must be 1D, got shape {idx.shape}.")
    if idx.shape[0] != size:
        raise ValueError(
            f"Permutation indices have length {idx.shape[0]}, expected {size}."
        )
    int_idx = idx.astype(np.int64)
    if not np.allclose(idx, int_idx):
        raise ValueError("Permutation indices must contain integer values.")
    if sorted(int_idx.tolist()) != list(range(size)):
        raise ValueError(
            f"Permutation indices must be a rearrangement of 0..{size - 1}."
        )
    return int_idx


def as_permutation_matrix(
    value: Any,
    *,
    size: int | None = None,
    dtype=np.float64,
) -> np.ndarray:
    """Return ``value`` as a permutation matrix, accepting indices or matrices."""

    arr = np.asarray(value)
    if arr.ndim == 1:
        matrix_size = int(size) if size is not None else int(arr.shape[0])
        idx = _validate_indices(arr, matrix_size)
        matrix = np.zeros((matrix_size, matrix_size), dtype=dtype)
        matrix[np.arange(matrix_size), idx] = 1.0
        return matrix
    if arr.ndim in {2, 3}:
        if size is not None and arr.shape[-2:] != (size, size):
            raise ValueError(
                f"Permutation matrix has shape {arr.shape}, expected "
                f"(..., {size}, {size})."
            )
        return arr.astype(dtype, copy=False)
    raise ValueError(
        f"Expected permutation as indices or matrix, got shape {arr.shape}."
    )


@functools.partial(jax.jit, static_argnums=(1, 2))
def sinkhorn_operator(
    logits: jnp.ndarray, tau: float = 0.1, n_iters: int = 50
) -> jnp.ndarray:
    """Project logits onto the Birkhoff polytope by Sinkhorn normalization."""

    scaled = logits / tau
    shifted = scaled - jnp.max(scaled, axis=(-2, -1), keepdims=True)
    matrix = jnp.exp(shifted)

    def body(_, mat):
        mat = mat / (jnp.sum(mat, axis=-2, keepdims=True) + 1e-9)
        mat = mat / (jnp.sum(mat, axis=-1, keepdims=True) + 1e-9)
        return mat

    return jax.lax.fori_loop(0, n_iters, body, matrix)


@dataclass
class PermutationState:
    """Solver state keyed by permutation group id."""

    group_order: tuple[str, ...]
    hard: dict[str, Any]
    logits: dict[str, jnp.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(
        cls,
        problem,
        *,
        backend: Literal["jax", "numpy"] = "jax",
        dtype=np.float32,
    ) -> PermutationState:
        eye = jnp.eye if backend == "jax" else np.eye
        hard = {
            group_id: eye(group.size, dtype=dtype)
            for group_id, group in problem.groups.items()
        }
        return cls(group_order=problem.group_order, hard=hard)

    @classmethod
    def from_perms(
        cls,
        problem,
        perms: Mapping[str, Any],
        *,
        backend: Literal["jax", "numpy"] = "numpy",
    ) -> PermutationState:
        """Build a hard state, defaulting unspecified groups to identity."""

        state = cls.identity(problem, backend=backend)
        return state.with_hard(perms)

    def with_hard(self, updates: Mapping[str, Any]) -> PermutationState:
        hard = dict(self.hard)
        hard.update(dict(updates))
        return PermutationState(
            group_order=self.group_order,
            hard=hard,
            logits=self.logits,
            metadata=dict(self.metadata),
        )

    def with_logits(self, logits: Mapping[str, jnp.ndarray]) -> PermutationState:
        return PermutationState(
            group_order=self.group_order,
            hard=dict(self.hard),
            logits=dict(logits),
            metadata=dict(self.metadata),
        )

    def soft(self, tau: float = 0.1, n_iters: int = 50) -> dict[str, jnp.ndarray]:
        """Project stored logits to doubly stochastic matrices."""

        if self.logits is None:
            return {
                gid: jnp.asarray(
                    as_permutation_matrix(self.hard[gid], dtype=np.float32)
                )
                for gid in self.group_order
            }
        return {
            gid: sinkhorn_operator(self.logits[gid], tau=tau, n_iters=n_iters)
            for gid in self.group_order
        }

    def harden(self, method: str = "hungarian") -> PermutationState:
        """Convert current matrices or projected logits to hard permutations."""

        if method.lower() != "hungarian":
            raise ValueError(f"Unsupported hardening method {method!r}.")
        matrices = self.soft() if self.logits is not None else self.hard
        hard: dict[str, Any] = {}
        for group_id in self.group_order:
            matrix = as_permutation_matrix(matrices[group_id])
            if matrix.ndim == 2:
                hard[group_id] = solve_lap_maximize(matrix)
            elif matrix.ndim == 3:
                hard[group_id] = np.stack(
                    [solve_lap_maximize(matrix[idx]) for idx in range(matrix.shape[0])],
                    axis=0,
                )
            else:
                raise ValueError(
                    f"Cannot harden group {group_id!r} with shape {matrix.shape}."
                )
        metadata = dict(self.metadata)
        metadata["hardening_method"] = method.lower()
        return PermutationState(
            group_order=self.group_order,
            hard=hard,
            logits=None,
            metadata=metadata,
        )

    def validate(self, problem, *, hard: bool = False) -> None:
        """Validate shapes, group keys, and optionally hard-permutation constraints."""

        if set(self.group_order) != set(problem.groups):
            raise ValueError("State group_order must contain exactly problem groups.")
        if set(self.hard) != set(problem.groups):
            raise ValueError("State hard mapping must contain exactly problem groups.")
        for group_id, group in problem.groups.items():
            value = np.asarray(self.hard[group_id])
            if value.ndim == 1:
                _validate_indices(value, group.size)
                continue
            if value.shape[-2:] != (group.size, group.size):
                raise ValueError(
                    f"Group {group_id!r} has matrix shape {value.shape}, "
                    f"expected (..., {group.size}, {group.size})."
                )
            matrix = value
            if hard:
                row_error = np.max(np.abs(np.sum(matrix, axis=-1) - 1.0))
                col_error = np.max(np.abs(np.sum(matrix, axis=-2) - 1.0))
                binary_error = np.max(np.minimum(np.abs(matrix), np.abs(matrix - 1.0)))
                if max(float(row_error), float(col_error), float(binary_error)) > 1e-5:
                    raise ValueError(f"Group {group_id!r} is not a hard permutation.")
        if self.logits is not None:
            if set(self.logits) != set(problem.groups):
                raise ValueError("State logits must contain exactly problem groups.")
            for group_id, group in problem.groups.items():
                logits = self.logits[group_id]
                if logits.shape[-2:] != (group.size, group.size):
                    raise ValueError(
                        f"Group {group_id!r} logits shape {logits.shape}, "
                        f"expected (..., {group.size}, {group.size})."
                    )

    def to_artifacts(self, dtype_policy: str = "hard_uint8") -> dict[str, np.ndarray]:
        """Serialize final hard permutations keyed by group id."""

        if dtype_policy != "hard_uint8":
            raise ValueError(
                f"Unsupported permutation artifact policy {dtype_policy!r}."
            )
        return {
            group_id: (as_permutation_matrix(self.hard[group_id]) > 0.5).astype(
                np.uint8
            )
            for group_id in self.group_order
        }


__all__ = [
    "PermutationState",
    "as_permutation_matrix",
    "sinkhorn_operator",
    "solve_lap_maximize",
]
