"""Transform state and permutation-projection utilities."""

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
class TransformState:
    """Solver state keyed by symmetry group id.

    ``matrices`` holds one hard group transform per group (a permutation,
    signed permutation, rotation-pair block, or orthogonal matrix); ``logits``
    optionally carries the Sinkhorn relaxation's pre-projection parameters.
    """

    group_order: tuple[str, ...]
    matrices: dict[str, Any]
    logits: dict[str, jnp.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(
        cls,
        graph,
        *,
        backend: Literal["jax", "numpy"] = "jax",
        dtype=np.float32,
    ) -> TransformState:
        eye = jnp.eye if backend == "jax" else np.eye
        matrices = {
            group_id: eye(group.size, dtype=dtype)
            for group_id, group in graph.groups.items()
        }
        return cls(group_order=graph.group_order, matrices=matrices)

    @classmethod
    def from_transforms(
        cls,
        graph,
        transforms: Mapping[str, Any],
        *,
        backend: Literal["jax", "numpy"] = "numpy",
    ) -> TransformState:
        """Build a hard state, defaulting unspecified groups to identity."""

        state = cls.identity(graph, backend=backend)
        return state.with_matrices(transforms)

    def with_matrices(self, updates: Mapping[str, Any]) -> TransformState:
        matrices = dict(self.matrices)
        matrices.update(dict(updates))
        return TransformState(
            group_order=self.group_order,
            matrices=matrices,
            logits=self.logits,
            metadata=dict(self.metadata),
        )

    def with_logits(self, logits: Mapping[str, jnp.ndarray]) -> TransformState:
        return TransformState(
            group_order=self.group_order,
            matrices=dict(self.matrices),
            logits=dict(logits),
            metadata=dict(self.metadata),
        )

    def soft(self, tau: float = 0.1, n_iters: int = 50) -> dict[str, jnp.ndarray]:
        """Project stored logits to doubly stochastic matrices.

        Groups without logits (e.g. rotation-pair groups excluded from the
        Sinkhorn relaxation) fall back to their current hard matrices.
        """

        logits = self.logits or {}
        return {
            gid: sinkhorn_operator(logits[gid], tau=tau, n_iters=n_iters)
            if gid in logits
            else jnp.asarray(
                as_permutation_matrix(self.matrices[gid], dtype=np.float32)
            )
            for gid in self.group_order
        }

    def harden(
        self, method: str = "hungarian", *, groups: tuple[str, ...] | None = None
    ) -> TransformState:
        """Convert current matrices or projected logits to hard permutations.

        ``groups`` restricts hardening to the listed groups; others keep
        their current matrices untouched. Solvers that relax only the
        permutation-capable groups must pass their group subset so
        rotation/orthogonal matrices held by other groups are not projected
        onto permutations.
        """

        if method.lower() != "hungarian":
            raise ValueError(f"Unsupported hardening method {method!r}.")
        source = self.soft() if self.logits is not None else self.matrices
        selected = set(self.group_order if groups is None else groups)
        hardened: dict[str, Any] = {}
        for group_id in self.group_order:
            if group_id not in selected:
                hardened[group_id] = self.matrices[group_id]
                continue
            matrix = as_permutation_matrix(source[group_id])
            if matrix.ndim == 2:
                hardened[group_id] = solve_lap_maximize(matrix)
            elif matrix.ndim == 3:
                hardened[group_id] = np.stack(
                    [solve_lap_maximize(matrix[idx]) for idx in range(matrix.shape[0])],
                    axis=0,
                )
            else:
                raise ValueError(
                    f"Cannot harden group {group_id!r} with shape {matrix.shape}."
                )
        metadata = dict(self.metadata)
        metadata["hardening_method"] = method.lower()
        return TransformState(
            group_order=self.group_order,
            matrices=hardened,
            logits=None,
            metadata=metadata,
        )

    def validate(self, graph, *, hard: bool = False) -> None:
        """Validate shapes, group keys, and optionally hard-transform constraints.

        With ``hard=True`` every matrix must be a hard member of its group's
        declared transform family: a permutation, a signed permutation (its
        entrywise absolute value is a permutation), or an orthogonal matrix
        (rotation-pair and orthogonal groups).
        """

        if set(self.group_order) != set(graph.groups):
            raise ValueError("State group_order must contain exactly the graph groups.")
        if set(self.matrices) != set(graph.groups):
            raise ValueError(
                "State matrices mapping must contain exactly the graph groups."
            )
        for group_id, group in graph.groups.items():
            value = np.asarray(self.matrices[group_id])
            if value.ndim == 1:
                _validate_indices(value, group.size)
                continue
            if value.shape[-2:] != (group.size, group.size):
                raise ValueError(
                    f"Group {group_id!r} has matrix shape {value.shape}, "
                    f"expected (..., {group.size}, {group.size})."
                )
            if not hard:
                continue
            transform_family = getattr(group, "transform_family", "permutation")
            if transform_family in ("rotation_pairs", "orthogonal"):
                gram = value @ np.swapaxes(value, -1, -2)
                eye = np.eye(group.size, dtype=gram.dtype)
                if float(np.max(np.abs(gram - eye))) > 1e-4:
                    raise ValueError(f"Group {group_id!r} is not orthogonal.")
                continue
            matrix = (
                np.abs(value) if transform_family == "signed_permutation" else value
            )
            row_error = np.max(np.abs(np.sum(matrix, axis=-1) - 1.0))
            col_error = np.max(np.abs(np.sum(matrix, axis=-2) - 1.0))
            binary_error = np.max(np.minimum(np.abs(matrix), np.abs(matrix - 1.0)))
            if max(float(row_error), float(col_error), float(binary_error)) > 1e-5:
                raise ValueError(
                    f"Group {group_id!r} is not a hard {transform_family.replace('_', ' ')}."
                )
        if self.logits is not None:
            unknown = sorted(set(self.logits) - set(graph.groups))
            if unknown:
                raise ValueError(
                    "State logits reference unknown group(s): " + ", ".join(unknown)
                )
            for group_id, logits in self.logits.items():
                size = graph.groups[group_id].size
                if logits.shape[-2:] != (size, size):
                    raise ValueError(
                        f"Group {group_id!r} logits shape {logits.shape}, "
                        f"expected (..., {size}, {size})."
                    )

    def to_artifacts(self) -> dict[str, np.ndarray]:
        """Serialize the final hard transforms keyed by group id.

        Plain permutation matrices are stored as compact ``uint8``; signed
        permutations, rotations, and orthogonal matrices (which are not
        binary) are stored as ``float32``. The distinction is content-based,
        so artifacts stay self-describing and every transform family
        round-trips exactly through
        :func:`align.runtime.artifacts.write_transforms_artifact`.
        """

        artifacts: dict[str, np.ndarray] = {}
        for group_id in self.group_order:
            matrix = as_permutation_matrix(self.matrices[group_id])
            binary_error = float(
                np.max(np.minimum(np.abs(matrix), np.abs(matrix - 1.0)))
            )
            if binary_error <= 1e-5:
                artifacts[group_id] = (matrix > 0.5).astype(np.uint8)
            else:
                artifacts[group_id] = matrix.astype(np.float32)
        return artifacts


__all__ = [
    "TransformState",
    "as_permutation_matrix",
    "sinkhorn_operator",
    "solve_lap_maximize",
]
