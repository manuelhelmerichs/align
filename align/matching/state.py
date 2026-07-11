"""Compact typed transform state and permutation-projection utilities."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment


def _array_namespace(value: Any):
    return jnp if isinstance(value, jax.Array) else np


def _validate_indices(indices: Any, size: int) -> Any:
    """Validate one permutation index vector or a batch of vectors."""

    arr = np.asarray(indices)
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"Permutation indices must be 1D or batched 2D, got shape {arr.shape}."
        )
    if arr.shape[-1] != size:
        raise ValueError(
            f"Permutation indices have length {arr.shape[-1]}, expected {size}."
        )
    int_arr = arr.astype(np.int64)
    if not np.array_equal(arr, int_arr):
        raise ValueError("Permutation indices must contain integer values.")
    rows = int_arr[None, :] if int_arr.ndim == 1 else int_arr
    expected = np.arange(size)
    if any(not np.array_equal(np.sort(row), expected) for row in rows):
        raise ValueError(
            f"Permutation indices must be a rearrangement of 0..{size - 1}."
        )
    xp = _array_namespace(indices)
    return xp.asarray(indices, dtype=xp.int32)


@dataclass(frozen=True)
class PermutationTransform:
    """A permutation represented by gather indices ``new[i] = old[indices[i]]``."""

    indices: Any


@dataclass(frozen=True)
class SignedPermutationTransform:
    """A signed permutation represented by gather indices and output signs."""

    indices: Any
    signs: Any


@dataclass(frozen=True)
class MatrixTransform:
    """A dense continuous transform for rotation-pair or orthogonal groups."""

    matrix: Any
    family: Literal["rotation_pairs", "orthogonal"]


HardTransform = PermutationTransform | SignedPermutationTransform | MatrixTransform


def solve_lap_maximize(cost: np.ndarray) -> np.ndarray:
    """Return gather indices for the square assignment maximizing ``cost``."""

    mat = np.ascontiguousarray(cost, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"LAP cost must be square, got shape {mat.shape}.")
    row_ind, col_ind = linear_sum_assignment(-mat)
    indices = np.empty(mat.shape[0], dtype=np.int32)
    indices[row_ind] = col_ind
    return indices


def _matrix_to_permutation_indices(value: Any, *, size: int) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.shape[-2:] != (size, size) or matrix.ndim not in (2, 3):
        raise ValueError(
            f"Permutation matrix has shape {matrix.shape}, expected (..., {size}, {size})."
        )
    indices = np.argmax(matrix, axis=-1).astype(np.int32)
    try:
        _validate_indices(indices, size)
    except ValueError as exc:
        raise ValueError("Value is not a hard permutation matrix.") from exc
    expected = np.eye(size, dtype=matrix.dtype)[indices]
    if not np.allclose(matrix, expected, rtol=0.0, atol=1e-5):
        raise ValueError("Value is not a hard permutation matrix.")
    return indices


def normalize_hard_transform(group: Any, value: Any) -> HardTransform:
    """Normalize a public hard-transform value into its canonical typed form."""

    size = int(group.size)
    family = str(group.transform_family)
    if isinstance(value, PermutationTransform):
        if family != "permutation":
            raise ValueError(f"Group {group.id!r} requires a {family} transform.")
        return PermutationTransform(_validate_indices(value.indices, size))
    if isinstance(value, SignedPermutationTransform):
        if family not in ("signed_permutation", "orthogonal"):
            raise ValueError(f"Group {group.id!r} does not accept signed permutations.")
        indices = _validate_indices(value.indices, size)
        signs = np.asarray(value.signs)
        if signs.shape != np.asarray(indices).shape or not np.all(
            np.isin(signs, (-1, 1))
        ):
            raise ValueError(
                f"Signed permutation for group {group.id!r} needs ±1 signs matching "
                f"indices shape {np.asarray(indices).shape}."
            )
        xp = _array_namespace(value.signs)
        return SignedPermutationTransform(indices, xp.asarray(value.signs))
    if isinstance(value, MatrixTransform):
        if family != value.family:
            raise ValueError(
                f"Group {group.id!r} declares {family!r}, got {value.family!r}."
            )
        matrix = value.matrix
    else:
        arr = np.asarray(value)
        if family == "permutation":
            indices = (
                _validate_indices(value, size)
                if arr.ndim in (1, 2) and arr.shape[-1] == size and arr.ndim == 1
                else _matrix_to_permutation_indices(value, size=size)
            )
            return PermutationTransform(indices)
        if family in ("signed_permutation", "orthogonal") and arr.ndim in (2, 3):
            try:
                abs_indices = _matrix_to_permutation_indices(np.abs(arr), size=size)
            except ValueError:
                if family == "signed_permutation":
                    raise
            else:
                signs = np.take_along_axis(
                    arr, np.asarray(abs_indices)[..., :, None], axis=-1
                )[..., 0]
                transform = SignedPermutationTransform(
                    _validate_indices(abs_indices, size), np.asarray(signs)
                )
                return normalize_hard_transform(group, transform)
        matrix = value

    arr = np.asarray(matrix)
    if arr.ndim not in (2, 3) or arr.shape[-2:] != (size, size):
        raise ValueError(
            f"Group {group.id!r} has transform shape {arr.shape}, expected "
            f"(..., {size}, {size})."
        )
    gram = arr @ np.swapaxes(arr, -1, -2)
    if not np.allclose(gram, np.eye(size), rtol=0.0, atol=1e-4):
        raise ValueError(f"Group {group.id!r} transform is not orthogonal.")
    return MatrixTransform(matrix, family=family)


def transform_matrix(transform: HardTransform, *, dtype: Any | None = None) -> Any:
    """Materialize a typed transform as a matrix for objectives/relaxations."""

    if isinstance(transform, MatrixTransform):
        xp = _array_namespace(transform.matrix)
        return xp.asarray(transform.matrix, dtype=dtype)
    indices = transform.indices
    xp = _array_namespace(indices)
    idx = xp.asarray(indices, dtype=xp.int32)
    size = int(idx.shape[-1])
    matrix = xp.eye(size, dtype=dtype or xp.float32)[idx]
    if isinstance(transform, SignedPermutationTransform):
        signs = xp.asarray(transform.signs, dtype=matrix.dtype)
        matrix = matrix * signs[..., :, None]
    return matrix


def as_permutation_matrix(
    value: Any, *, size: int | None = None, dtype=np.float64
) -> np.ndarray:
    """Materialize permutation indices/typed transforms as a dense matrix."""

    if isinstance(
        value, (PermutationTransform, SignedPermutationTransform, MatrixTransform)
    ):
        return np.asarray(transform_matrix(value, dtype=dtype))
    arr = np.asarray(value)
    if arr.ndim == 1:
        matrix_size = int(size) if size is not None else int(arr.shape[0])
        indices = _validate_indices(arr, matrix_size)
        return np.eye(matrix_size, dtype=dtype)[np.asarray(indices)]
    if arr.ndim in (2, 3):
        if size is not None and arr.shape[-2:] != (size, size):
            raise ValueError(
                f"Permutation matrix has shape {arr.shape}, expected (..., {size}, {size})."
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
    """Hard typed transforms plus optional Sinkhorn relaxation state."""

    group_order: tuple[str, ...]
    transforms: dict[str, HardTransform]
    logits: dict[str, jnp.ndarray] | None = None
    relaxed_matrices: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(
        cls,
        graph,
        *,
        backend: Literal["jax", "numpy"] = "jax",
        dtype=np.float32,
    ) -> TransformState:
        xp = jnp if backend == "jax" else np
        transforms: dict[str, HardTransform] = {}
        for group_id, group in graph.groups.items():
            indices = xp.arange(group.size, dtype=xp.int32)
            if group.transform_family == "permutation":
                transform: HardTransform = PermutationTransform(indices)
            elif group.transform_family in ("signed_permutation", "orthogonal"):
                transform = SignedPermutationTransform(
                    indices, xp.ones(group.size, dtype=dtype)
                )
            else:
                transform = MatrixTransform(
                    xp.eye(group.size, dtype=dtype), family="rotation_pairs"
                )
            transforms[group_id] = transform
        return cls(group_order=graph.group_order, transforms=transforms)

    @classmethod
    def from_transforms(
        cls,
        graph,
        transforms: Mapping[str, Any],
        *,
        backend: Literal["jax", "numpy"] = "numpy",
    ) -> TransformState:
        """Build a hard state, normalizing inputs to compact typed transforms."""

        state = cls.identity(graph, backend=backend)
        unknown = sorted(set(transforms) - set(graph.groups))
        if unknown:
            raise ValueError("Unknown transform group(s): " + ", ".join(unknown))
        updates = {
            group_id: normalize_hard_transform(graph.groups[group_id], value)
            for group_id, value in transforms.items()
        }
        return state.with_transforms(updates)

    def with_transforms(self, updates: Mapping[str, HardTransform]) -> TransformState:
        transforms = dict(self.transforms)
        transforms.update(dict(updates))
        return TransformState(
            group_order=self.group_order,
            transforms=transforms,
            logits=self.logits,
            relaxed_matrices=self.relaxed_matrices,
            metadata=dict(self.metadata),
        )

    def with_relaxation(
        self, logits: Mapping[str, jnp.ndarray], matrices: Mapping[str, Any]
    ) -> TransformState:
        return TransformState(
            group_order=self.group_order,
            transforms=dict(self.transforms),
            logits=dict(logits),
            relaxed_matrices=dict(matrices),
            metadata=dict(self.metadata),
        )

    def matrix(self, group_id: str, *, dtype: Any | None = None) -> Any:
        if self.relaxed_matrices is not None and group_id in self.relaxed_matrices:
            value = self.relaxed_matrices[group_id]
            xp = _array_namespace(value)
            return xp.asarray(value, dtype=dtype)
        return transform_matrix(self.transforms[group_id], dtype=dtype)

    def harden(self, *, groups: tuple[str, ...]) -> TransformState:
        """Harden selected plain-permutation Sinkhorn groups with Hungarian LAP."""

        selected = set(groups)
        transforms = dict(self.transforms)
        for group_id in groups:
            matrix = np.asarray(self.matrix(group_id))
            if matrix.ndim == 2:
                indices = solve_lap_maximize(matrix)
            elif matrix.ndim == 3:
                indices = np.stack(
                    [solve_lap_maximize(item) for item in matrix], axis=0
                )
            else:
                raise ValueError(
                    f"Cannot harden group {group_id!r} with shape {matrix.shape}."
                )
            transforms[group_id] = PermutationTransform(indices)
        if selected - set(self.group_order):
            raise ValueError("Cannot harden unknown groups.")
        metadata = dict(self.metadata)
        metadata["hardening_method"] = "hungarian"
        return TransformState(
            group_order=self.group_order,
            transforms=transforms,
            metadata=metadata,
        )

    def validate(self, graph, *, hard: bool = False) -> None:
        if set(self.group_order) != set(graph.groups):
            raise ValueError("State group_order must contain exactly the graph groups.")
        if set(self.transforms) != set(graph.groups):
            raise ValueError("State transforms must contain exactly the graph groups.")
        for group_id, group in graph.groups.items():
            transform = self.transforms[group_id]
            family = group.transform_family
            if family == "permutation" and not isinstance(
                transform, PermutationTransform
            ):
                raise ValueError(f"Group {group_id!r} requires permutation indices.")
            if family == "signed_permutation" and not isinstance(
                transform, SignedPermutationTransform
            ):
                raise ValueError(f"Group {group_id!r} requires signed indices/signs.")
            if family == "rotation_pairs" and not (
                isinstance(transform, MatrixTransform)
                and transform.family == "rotation_pairs"
            ):
                raise ValueError(f"Group {group_id!r} requires a rotation matrix.")
            if family == "orthogonal" and not isinstance(
                transform, (SignedPermutationTransform, MatrixTransform)
            ):
                raise ValueError(
                    f"Group {group_id!r} requires an orthogonal transform."
                )
            normalize_hard_transform(group, transform)
        if hard and (self.logits is not None or self.relaxed_matrices is not None):
            raise ValueError(
                "A relaxed TransformState cannot be applied as a hard alignment."
            )
        for mapping, label in (
            (self.logits, "logits"),
            (self.relaxed_matrices, "relaxed matrices"),
        ):
            if mapping is None:
                continue
            unknown = sorted(set(mapping) - set(graph.groups))
            if unknown:
                raise ValueError(f"State {label} reference unknown groups: {unknown}.")
            for group_id, value in mapping.items():
                size = graph.groups[group_id].size
                if value.shape[-2:] != (size, size):
                    raise ValueError(
                        f"Group {group_id!r} {label} shape {value.shape}, expected "
                        f"(..., {size}, {size})."
                    )

    def to_artifacts(self) -> dict[str, np.ndarray]:
        """Serialize hard transforms in their compact declared representation."""

        if self.logits is not None or self.relaxed_matrices is not None:
            raise ValueError(
                "Relaxed transform states cannot be persisted as alignments."
            )
        artifacts: dict[str, np.ndarray] = {}
        for group_id in self.group_order:
            transform = self.transforms[group_id]
            if isinstance(transform, PermutationTransform):
                artifacts[group_id] = np.asarray(transform.indices, dtype=np.int32)
            elif isinstance(transform, SignedPermutationTransform):
                artifacts[group_id] = np.stack(
                    [
                        np.asarray(transform.indices, dtype=np.int32),
                        np.asarray(transform.signs, dtype=np.int32),
                    ],
                    axis=0,
                )
            else:
                artifacts[group_id] = np.asarray(transform.matrix, dtype=np.float32)
        return artifacts


__all__ = [
    "HardTransform",
    "MatrixTransform",
    "PermutationTransform",
    "SignedPermutationTransform",
    "TransformState",
    "as_permutation_matrix",
    "normalize_hard_transform",
    "sinkhorn_operator",
    "solve_lap_maximize",
    "transform_matrix",
]
