"""Synthetic exact-orbit benchmarks for alignment correctness.

The cases in this module are intentionally small and deterministic. They test
known group actions rather than learned checkpoints, so a zero optimality gap is
mathematically known for the exact-orbit problems.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from align.architectures import (
    ArchitectureRecipe,
    LayerNormMHATransformerRecipe,
    MLPRecipe,
    ResidualConvNetRecipe,
    RMSNormGQARoPETransformerRecipe,
)
from align.canonicalization import ScaleCanonicalizer, ScaleState
from align.matching import (
    TransformState,
    default_lap_schedule,
    match_sample,
    resolve_calibration_kwargs,
)
from align.symmetry import (
    AxisBinding,
    GQARoPECircuitConstraint,
    MHACircuitConstraint,
    SymmetryGraph,
    SymmetryGroup,
    TensorSpec,
)

ParamTree = Mapping[str, Any]
ApplyFn = Callable[[ParamTree, jax.Array], jax.Array]


@dataclass
class SyntheticOrbitCase:
    """A synthetic pair of functionally equivalent parameter trees."""

    name: str
    recipe: ArchitectureRecipe | None
    problem: SymmetryGraph
    reference: ParamTree
    target: ParamTree
    inputs: jax.Array
    apply_fn: ApplyFn
    expected_transforms: Mapping[str, np.ndarray]
    expected_residual_channel_ties: tuple[tuple[str, ...], ...] = ()
    known_optimum_distance: float | None = 0.0


@dataclass(frozen=True)
class AlignmentBenchmarkMetrics:
    """Quality metrics for one exact synthetic alignment run."""

    function_drift_max: float
    distance_before: float
    distance_after: float
    distance_reduction: float
    optimality_gap: float | None
    transform_validity_error: float
    recovered_transform_error: float
    residual_constraint_violations: int


@dataclass
class AlignmentBenchmarkResult:
    """Output of an alignment benchmark run."""

    aligned_params: ParamTree
    transforms: Mapping[str, Any]
    diagnostics: dict[str, Any] | None
    metrics: AlignmentBenchmarkMetrics


@dataclass(frozen=True)
class PerformanceMeasurement:
    """Basic timing and memory measurements for a matching benchmark."""

    cold_time_s: float
    warm_throughput_samples_s: float
    peak_memory_bytes: int
    compilation_time_s: float


def permutation_matrix(indices: Sequence[int]) -> np.ndarray:
    """Return ``P`` with ``P[new_index, old_index] = 1``."""

    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("Permutation indices must be one-dimensional.")
    n = int(idx.shape[0])
    if sorted(idx.tolist()) != list(range(n)):
        raise ValueError(f"Invalid permutation indices for size {n}: {idx.tolist()}")
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[np.arange(n), idx] = 1.0
    return matrix


def _natural_key(value: str) -> list[int | str]:
    import re

    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", value)
        if part
    ]


def _descend(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = mapping
    for key in path:
        node = node[key]  # type: ignore[index]
    return node


def _flatten_arrays(params: ParamTree) -> tuple[list[np.ndarray], Any]:
    leaves, treedef = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(np.asarray, params)
    )
    return [np.ravel(np.asarray(leaf, dtype=np.float64)) for leaf in leaves], treedef


def tree_l2_distance(left: ParamTree, right: ParamTree) -> float:
    """Return the Euclidean distance between two matching parameter PyTrees."""

    left_leaves, left_def = _flatten_arrays(left)
    right_leaves, right_def = _flatten_arrays(right)
    if left_def != right_def:
        raise ValueError("Parameter trees have different structures.")
    total = 0.0
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        if lhs.shape != rhs.shape:
            raise ValueError(f"Leaf shapes differ: {lhs.shape} != {rhs.shape}.")
        diff = lhs - rhs
        total += float(np.dot(diff, diff))
    return float(np.sqrt(total))


def mlp_apply(params: ParamTree, x: jax.Array) -> jax.Array:
    """Evaluate a ReLU MLP stored with Flax Dense kernels ``(in_dim, out_dim)``."""

    layers = params["params"]["fcn"]  # type: ignore[index]
    names = sorted(layers, key=_natural_key)
    h = x
    for name in names[:-1]:
        layer = layers[name]
        w = jnp.asarray(layer["kernel"])
        b = jnp.asarray(layer["bias"])
        h = jnp.maximum(0.0, h @ w + b)
    last = layers[names[-1]]
    return h @ jnp.asarray(last["kernel"]) + jnp.asarray(last["bias"])


def _conv1x1(x: jax.Array, kernel: jax.Array, bias: jax.Array) -> jax.Array:
    return jnp.einsum("bhwc,co->bhwo", x, kernel[0, 0]) + bias


def _residual_conv_apply(params: ParamTree, x: jax.Array) -> jax.Array:
    core = params["core"]  # type: ignore[index]
    h0 = _conv1x1(
        x,
        jnp.asarray(core["Conv_0"]["kernel"]),
        jnp.asarray(core["Conv_0"]["bias"]),
    )
    h0 = jnp.maximum(0.0, h0)
    h1 = _conv1x1(
        h0,
        jnp.asarray(core["Conv_1"]["kernel"]),
        jnp.asarray(core["Conv_1"]["bias"]),
    )
    h1 = jnp.maximum(0.0, h1)
    h2 = _conv1x1(
        h1,
        jnp.asarray(core["Conv_2"]["kernel"]),
        jnp.asarray(core["Conv_2"]["bias"]),
    )
    h = jnp.maximum(0.0, h0 + h2)
    pooled = jnp.mean(h, axis=(1, 2))
    dense = core["Dense_0"]
    return pooled @ jnp.asarray(dense["kernel"]) + jnp.asarray(dense["bias"])


def _frn(x: jax.Array, module: Mapping[str, Any]) -> jax.Array:
    """Filter Response Normalization over spatial axes, with learned eps."""

    nu2 = jnp.mean(jnp.square(x), axis=(1, 2), keepdims=True)
    eps = jnp.abs(jnp.asarray(module["eps"]))
    normed = x / jnp.sqrt(nu2 + eps)
    return jnp.asarray(module["gamma"]) * normed + jnp.asarray(module["beta"])


def _tlu(x: jax.Array, tau: jax.Array) -> jax.Array:
    return jnp.maximum(x, jnp.asarray(tau))


def _frn_residual_conv_apply(params: ParamTree, x: jax.Array) -> jax.Array:
    """FRN/TLU residual stack: Conv+FRN+TLU, branch, post-add TLU, dense head."""

    core = params["core"]  # type: ignore[index]
    h0 = _conv1x1(
        x,
        jnp.asarray(core["Conv_0"]["kernel"]),
        jnp.asarray(core["Conv_0"]["bias"]),
    )
    h0 = _tlu(_frn(h0, core["FRN_0"]), core["FRN_0"]["tau"])
    h1 = _conv1x1(
        h0,
        jnp.asarray(core["Conv_1"]["kernel"]),
        jnp.asarray(core["Conv_1"]["bias"]),
    )
    h1 = _tlu(_frn(h1, core["FRN_1"]), core["FRN_1"]["tau"])
    h2 = _conv1x1(
        h1,
        jnp.asarray(core["Conv_2"]["kernel"]),
        jnp.asarray(core["Conv_2"]["bias"]),
    )
    h2 = _frn(h2, core["FRN_2"])
    h = _tlu(h0 + h2, core["FRN_2"]["tau"])
    pooled = jnp.mean(h, axis=(1, 2))
    dense = core["Dense_0"]
    return pooled @ jnp.asarray(dense["kernel"]) + jnp.asarray(dense["bias"])


def make_frn_residual_conv_params(
    key: jax.Array,
    *,
    input_channels: int = 2,
    residual_channels: int = 4,
    branch_channels: int = 3,
    output_channels: int = 2,
) -> dict[str, Any]:
    """Random parameters in the SMILE-style Conv_*/FRN_* tree layout."""

    def _frn_module(rng_key: jax.Array, size: int) -> dict[str, jax.Array]:
        g_key, b_key, t_key = jax.random.split(rng_key, 3)
        return {
            "gamma": 1.0 + 0.3 * jax.random.normal(g_key, (size,), dtype=jnp.float32),
            "beta": 0.3 * jax.random.normal(b_key, (size,), dtype=jnp.float32),
            "tau": 0.3 * jax.random.normal(t_key, (size,), dtype=jnp.float32),
            "eps": 1e-6 * jnp.ones((size,), dtype=jnp.float32),
        }

    k0, k1, k2, kd, f0, f1, f2 = jax.random.split(key, 7)

    def _conv(rng_key: jax.Array, cin: int, cout: int) -> dict[str, jax.Array]:
        return {
            "kernel": jax.random.normal(rng_key, (1, 1, cin, cout), dtype=jnp.float32),
            "bias": 0.1 * jax.random.normal(rng_key, (cout,), dtype=jnp.float32),
        }

    return {
        "core": {
            "Conv_0": _conv(k0, input_channels, residual_channels),
            "FRN_0": _frn_module(f0, residual_channels),
            "Conv_1": _conv(k1, residual_channels, branch_channels),
            "FRN_1": _frn_module(f1, branch_channels),
            "Conv_2": _conv(k2, branch_channels, residual_channels),
            "FRN_2": _frn_module(f2, residual_channels),
            "Dense_0": {
                "kernel": jax.random.normal(
                    kd, (residual_channels, output_channels), dtype=jnp.float32
                ),
                "bias": 0.1
                * jax.random.normal(kd, (output_channels,), dtype=jnp.float32),
            },
        }
    }


def make_frn_residual_conv_orbit_case(seed: int = 0) -> SyntheticOrbitCase:
    """FRN/TLU residual case whose target carries known affine scales + transforms.

    The forward transform composes random positive per-channel scales on both
    channel groups (the exact post-norm affine symmetry: FRN gamma/beta/tau
    divided, consumers multiplied, conv kernels and eps untouched) with random
    channel permutations. ``known_optimum_distance`` of 0 presumes
    ``canonicalize=True``: permutation-only matching cannot remove the affine
    scales.
    """

    key = jax.random.PRNGKey(seed)
    param_key, input_key = jax.random.split(key)
    reference = make_frn_residual_conv_params(param_key)
    residual_topology = {
        "nodes": [
            {
                "name": "residual_add",
                "type": "add",
                "inputs": ["core/Conv_0", "core/Conv_2"],
            }
        ]
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=residual_topology
    )
    problem = recipe.build_graph(reference)

    rng = np.random.default_rng(seed + 977)
    scales = {
        group_id: np.exp(rng.uniform(-1.0, 1.0, size=group.size)).astype(np.float32)
        for group_id, group in problem.groups.items()
    }
    forward_perms = {
        group_id: permutation_matrix(rng.permutation(group.size))
        for group_id, group in problem.groups.items()
    }
    scaled = problem.apply_scales(reference, _scale_state(problem, scales))
    target = problem.apply_transforms(
        scaled, TransformState.from_transforms(problem, forward_perms)
    )
    inputs = jax.random.normal(input_key, (7, 3, 3, 2), dtype=jnp.float32)

    return SyntheticOrbitCase(
        name=f"frn_residual_conv_orbit_seed_{seed}",
        recipe=recipe,
        problem=problem,
        reference=reference,
        target=target,
        inputs=inputs,
        apply_fn=_frn_residual_conv_apply,
        expected_transforms=_inverse_permutations(forward_perms),
        expected_residual_channel_ties=(("core/Conv_0", "core/Conv_2"),),
    )


def _split_concat_conv_apply(params: ParamTree, x: jax.Array) -> jax.Array:
    core = params["core"]  # type: ignore[index]
    stem = _conv1x1(
        x,
        jnp.asarray(core["Stem"]["kernel"]),
        jnp.asarray(core["Stem"]["bias"]),
    )
    stem = jnp.maximum(0.0, stem)
    branch_a = _conv1x1(
        stem,
        jnp.asarray(core["BranchA"]["kernel"]),
        jnp.asarray(core["BranchA"]["bias"]),
    )
    branch_b = _conv1x1(
        stem,
        jnp.asarray(core["BranchB"]["kernel"]),
        jnp.asarray(core["BranchB"]["bias"]),
    )
    joined = jnp.concatenate(
        [jnp.maximum(0.0, branch_a), jnp.maximum(0.0, branch_b)],
        axis=-1,
    )
    pooled = jnp.mean(joined, axis=(1, 2))
    dense = core["Dense_0"]
    return pooled @ jnp.asarray(dense["kernel"]) + jnp.asarray(dense["bias"])


def _layer_norm(x: jax.Array, module: Mapping[str, Any]) -> jax.Array:
    """Flax-compatible LayerNorm (eps 1e-6, biased variance, optional bias)."""

    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normed = (x - mean) / jnp.sqrt(var + 1e-6)
    normed = normed * jnp.asarray(module["scale"])
    if "bias" in module:
        normed = normed + jnp.asarray(module["bias"])
    return normed


def _gelu(x: jax.Array) -> jax.Array:
    """Approximate GELU without backend plugins replacing its autodiff rules."""

    x = jnp.asarray(x)
    sqrt_2_over_pi = jnp.asarray(np.sqrt(2.0 / np.pi), dtype=x.dtype)
    cdf = 0.5 * (
        1.0
        + jnp.tanh(sqrt_2_over_pi * (x + jnp.asarray(0.044715, dtype=x.dtype) * x**3))
    )
    return x * cdf


def mhdpa_apply(
    module: Mapping[str, Any], x: jax.Array, *, causal: bool = False
) -> jax.Array:
    """Flax MultiHeadDotProductAttention forward on ``(B, T, d)`` inputs.

    Kernels are head-structured: q/k/v ``(d, H, dk)``, out ``(H, dk, d)``.
    """

    def _project(name: str) -> jax.Array:
        proj = jnp.einsum("btd,dhk->bthk", x, jnp.asarray(module[name]["kernel"]))
        if "bias" in module[name]:
            proj = proj + jnp.asarray(module[name]["bias"])
        return proj

    query, key, value = _project("query"), _project("key"), _project("value")
    head_dim = query.shape[-1]
    query = query / jnp.sqrt(jnp.asarray(head_dim, dtype=query.dtype))
    scores = jnp.einsum("bqhk,bshk->bhqs", query, key)
    if causal:
        seq_len = x.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(mask[None, None], scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("bhqs,bshk->bqhk", weights, value)
    out = jnp.einsum("bqhk,hkd->bqd", context, jnp.asarray(module["out"]["kernel"]))
    if "bias" in module["out"]:
        out = out + jnp.asarray(module["out"]["bias"])
    return out


def layernorm_mha_transformer_apply(params: ParamTree, tokens: jax.Array) -> jax.Array:
    """Forward pass matching the bayesmates GPT layout and semantics.

    Note the bayesmates component is not standard pre-LN: each LayerNorm *replaces*
    the stream (``x = LN(x); x = x + attn(x)``).
    """

    embedding = params["TokenEmbedding_0"]  # type: ignore[index]
    x = jnp.asarray(embedding["Embedding"]["embedding"])[tokens]
    positions = jnp.asarray(embedding["PositionEmbedding"]["embedding"])
    x = x + positions[: tokens.shape[1]][None, :, :]

    block_names = sorted(
        (name for name in params if str(name).startswith("Block_")),
        key=_natural_key,
    )
    for name in block_names:
        component = params[name]  # type: ignore[index]
        x = _layer_norm(x, component["LayerNorm_0"])
        attention = component["MaskedMultiHeadSelfAttention_0"][
            "MultiHeadDotProductAttention_0"
        ]
        x = x + mhdpa_apply(attention, x, causal=True)
        x = _layer_norm(x, component["LayerNorm_1"])
        ffn = component["FullyConnected_0"]
        hidden = _gelu(
            x @ jnp.asarray(ffn["FFN_layer0"]["kernel"])
            + jnp.asarray(ffn["FFN_layer0"]["bias"])
        )
        x = x + (
            hidden @ jnp.asarray(ffn["FFN_layer1"]["kernel"])
            + jnp.asarray(ffn["FFN_layer1"]["bias"])
        )

    x = _layer_norm(x, params["LayerNorm_0"])  # type: ignore[index]
    return x @ jnp.asarray(params["DenseLogits"]["kernel"])  # type: ignore[index]


def make_layernorm_mha_transformer_params(
    *,
    key: jax.Array,
    vocab_size: int = 7,
    context_len: int = 5,
    d_model: int = 8,
    num_heads: int = 2,
    num_blocks: int = 2,
    ffn_dim: int = 12,
) -> dict[str, Any]:
    """Random parameters in the bayesmates GPT tree layout."""

    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads.")
    head_dim = d_model // num_heads

    def _normal(rng_key: jax.Array, shape: tuple[int, ...], scale: float) -> jax.Array:
        return scale * jax.random.normal(rng_key, shape, dtype=jnp.float32)

    keys = iter(jax.random.split(key, 16 * num_blocks + 8))
    params: dict[str, Any] = {
        "TokenEmbedding_0": {
            "Embedding": {"embedding": _normal(next(keys), (vocab_size, d_model), 0.5)},
            "PositionEmbedding": {
                "embedding": _normal(next(keys), (context_len, d_model), 0.5)
            },
        },
        "LayerNorm_0": {
            "scale": 1.0 + _normal(next(keys), (d_model,), 0.1),
            "bias": _normal(next(keys), (d_model,), 0.1),
        },
    }
    for block_index in range(num_blocks):
        params[f"Block_{block_index}"] = {
            "LayerNorm_0": {
                "scale": 1.0 + _normal(next(keys), (d_model,), 0.1),
                "bias": _normal(next(keys), (d_model,), 0.1),
            },
            "LayerNorm_1": {
                "scale": 1.0 + _normal(next(keys), (d_model,), 0.1),
                "bias": _normal(next(keys), (d_model,), 0.1),
            },
            "MaskedMultiHeadSelfAttention_0": {
                "MultiHeadDotProductAttention_0": {
                    "query": {
                        "kernel": _normal(
                            next(keys), (d_model, num_heads, head_dim), 0.6
                        ),
                        "bias": _normal(next(keys), (num_heads, head_dim), 0.1),
                    },
                    "key": {
                        "kernel": _normal(
                            next(keys), (d_model, num_heads, head_dim), 0.6
                        ),
                        "bias": _normal(next(keys), (num_heads, head_dim), 0.1),
                    },
                    "value": {
                        "kernel": _normal(
                            next(keys), (d_model, num_heads, head_dim), 0.6
                        ),
                        "bias": _normal(next(keys), (num_heads, head_dim), 0.1),
                    },
                    "out": {
                        "kernel": _normal(
                            next(keys), (num_heads, head_dim, d_model), 0.6
                        ),
                        "bias": _normal(next(keys), (d_model,), 0.1),
                    },
                }
            },
            "FullyConnected_0": {
                "FFN_layer0": {
                    "kernel": _normal(next(keys), (d_model, ffn_dim), 0.6),
                    "bias": _normal(next(keys), (ffn_dim,), 0.1),
                },
                "FFN_layer1": {
                    "kernel": _normal(next(keys), (ffn_dim, d_model), 0.6),
                    "bias": _normal(next(keys), (d_model,), 0.1),
                },
            },
        }
    params["DenseLogits"] = {"kernel": _normal(next(keys), (d_model, vocab_size), 0.6)}
    return params


def make_layernorm_mha_transformer_orbit_case(seed: int = 0) -> SyntheticOrbitCase:
    """GPT-style case whose target is a known symmetry copy of the reference.

    The forward permutations cover the residual stream, per-component head
    permutations, per-slot qk/vo intra-head permutations, and FFN hidden
    permutations. Expected aligning permutations are the wreath-product
    inverse: intra expectations are transposed *and* re-indexed by the head
    permutation (the aligning intra for reference slot ``j`` inverts the
    forward intra of the slot the head permutation moved to ``j``).
    """

    key = jax.random.PRNGKey(seed)
    param_key, input_key, perm_key = jax.random.split(key, 3)
    reference = make_layernorm_mha_transformer_params(key=param_key)
    recipe = LayerNormMHATransformerRecipe()
    problem = recipe.build_graph(reference)

    rng = np.random.default_rng(int(jax.random.randint(perm_key, (), 0, 2**31 - 1)))
    forward_perms = {
        group_id: permutation_matrix(rng.permutation(group.size))
        for group_id, group in problem.groups.items()
    }
    target = problem.apply_transforms(
        reference, TransformState.from_transforms(problem, forward_perms)
    )

    expected: dict[str, np.ndarray] = {}
    attention_circuits = {
        constraint.head_group: constraint
        for constraint in problem.constraints
        if isinstance(constraint, MHACircuitConstraint)
    }
    intra_reindex: dict[str, str] = {}
    for constraint in attention_circuits.values():
        head_matrix = np.asarray(forward_perms[constraint.head_group])
        sigma = np.argmax(head_matrix, axis=1)  # new slot i took old head sigma[i]
        sigma_inv = np.argsort(sigma)
        for slot_groups in (constraint.qk_groups, constraint.vo_groups):
            for slot, group_id in enumerate(slot_groups):
                intra_reindex[group_id] = slot_groups[int(sigma_inv[slot])]
    for group_id in forward_perms:
        source = intra_reindex.get(group_id, group_id)
        expected[group_id] = np.asarray(forward_perms[source]).T

    tokens = jax.random.randint(input_key, (4, 5), 0, 7, dtype=jnp.int32)
    return SyntheticOrbitCase(
        name=f"layernorm_mha_transformer_exact_orbit_seed_{seed}",
        recipe=recipe,
        problem=problem,
        reference=reference,
        target=target,
        inputs=tokens,
        apply_fn=layernorm_mha_transformer_apply,
        expected_transforms=expected,
    )


def make_layernorm_mha_transformer_scaled_orbit_case(
    seed: int = 0,
) -> SyntheticOrbitCase:
    """Transformer orbit whose target also carries qk/vo circuit scales.

    The target composes the known permutation copy with random positive
    diagonal scales on every intra-head qk/vo group (the exact attention
    circuit symmetry). Permutation-only matching cannot reach the reference;
    balancing canonicalization removes the scales first, after which the
    expected permutations are those of the pure-permutation case.
    ``known_optimum_distance`` of 0 therefore presumes ``canonicalize=True``.
    """

    case = make_layernorm_mha_transformer_orbit_case(seed=seed)
    rng = np.random.default_rng(seed + 977)
    intra_groups = [
        group_id
        for constraint in case.problem.constraints
        if isinstance(constraint, MHACircuitConstraint)
        for group_id in (
            *constraint.qk_groups,
            *constraint.vo_groups,
        )
    ]
    scales = {
        group_id: np.exp(
            rng.uniform(-1.5, 1.5, size=case.problem.groups[group_id].size)
        ).astype(np.float32)
        for group_id in intra_groups
    }
    target = case.problem.apply_scales(case.target, _scale_state(case.problem, scales))
    return SyntheticOrbitCase(
        name=f"layernorm_mha_transformer_scaled_orbit_seed_{seed}",
        recipe=case.recipe,
        problem=case.problem,
        reference=case.reference,
        target=target,
        inputs=case.inputs,
        apply_fn=case.apply_fn,
        expected_transforms=case.expected_transforms,
    )


def _rms_norm(x: jax.Array, scale: Any, *, eps: float = 1e-6) -> jax.Array:
    """RMSNorm: no mean subtraction, no bias (exactly orthogonal-equivariant)."""

    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms) * jnp.asarray(scale)


def _rope_angles(seq_len: int, head_dim: int, *, base: float = 10000.0) -> jax.Array:
    """Rotation angles ``(T, head_dim/2)`` for half-split rotary embeddings."""

    half = head_dim // 2
    inv_freq = base ** (-jnp.arange(half, dtype=jnp.float32) * 2.0 / head_dim)
    return jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv_freq[None, :]


def _rope_rotate(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate half-split pairs ``(p, p + dk/2)`` of the trailing axis."""

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def gqa_attention_apply(module: Mapping[str, Any], x: jax.Array) -> jax.Array:
    """Causal grouped-query attention with rotary embeddings on ``(B, T, d)``.

    Kernel layout is kv-group-structured and bias-free: query
    ``(d, G, R, dk)``, key/value ``(d, G, dk)``, out ``(G, R, dk, d)``.
    RoPE uses the half-split pairing ``(p, p + dk/2)`` (LLaMA convention).
    """

    query_kernel = jnp.asarray(module["query"]["kernel"])
    seq_len = x.shape[1]
    head_dim = query_kernel.shape[-1]
    angles = _rope_angles(seq_len, head_dim)
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]

    query = jnp.einsum("btd,dgrk->btgrk", x, query_kernel)
    key = jnp.einsum("btd,dgk->btgk", x, jnp.asarray(module["key"]["kernel"]))
    value = jnp.einsum("btd,dgk->btgk", x, jnp.asarray(module["value"]["kernel"]))
    query = _rope_rotate(query, cos[..., None, :], sin[..., None, :])
    key = _rope_rotate(key, cos, sin)
    query = query / jnp.sqrt(jnp.asarray(head_dim, dtype=query.dtype))

    scores = jnp.einsum("btgrk,bsgk->bgrts", query, key)
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    scores = jnp.where(mask[None, None, None], scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("bgrts,bsgk->btgrk", weights, value)
    return jnp.einsum("btgrk,grkd->btd", context, jnp.asarray(module["out"]["kernel"]))


def rmsnorm_gqa_rope_transformer_apply(
    params: ParamTree, tokens: jax.Array
) -> jax.Array:
    """Forward pass for the RMSNorm + RoPE + GQA reference decoder.

    Standard pre-norm residual components (``x = x + attn(RMS(x))``, then
    ``x = x + ffn(RMS(x))``), GELU FFN, final RMSNorm, bias-free logits head.
    No positional embedding: positions enter only through RoPE.
    """

    x = jnp.asarray(params["Embed_0"]["embedding"])[tokens]  # type: ignore[index]
    block_names = sorted(
        (name for name in params if str(name).startswith("Block_")),
        key=_natural_key,
    )
    for name in block_names:
        component = params[name]  # type: ignore[index]
        h = _rms_norm(x, component["RMSNorm_0"]["scale"])
        x = x + gqa_attention_apply(component["GQAttention_0"], h)
        h = _rms_norm(x, component["RMSNorm_1"]["scale"])
        ffn = component["FFN_0"]
        hidden = _gelu(h @ jnp.asarray(ffn["FFN_layer0"]["kernel"]))
        x = x + hidden @ jnp.asarray(ffn["FFN_layer1"]["kernel"])
    x = _rms_norm(x, params["RMSNorm_f"]["scale"])  # type: ignore[index]
    return x @ jnp.asarray(params["DenseLogits"]["kernel"])  # type: ignore[index]


def make_rmsnorm_gqa_rope_transformer_params(
    *,
    key: jax.Array,
    vocab_size: int = 7,
    d_model: int = 8,
    num_kv_groups: int = 2,
    heads_per_group: int = 2,
    head_dim: int = 4,
    num_blocks: int = 2,
    ffn_dim: int = 12,
) -> dict[str, Any]:
    """Random parameters in the RMSNorm + RoPE + GQA reference tree layout."""

    if head_dim % 2:
        raise ValueError("head_dim must be even for rotary embeddings.")

    def _normal(rng_key: jax.Array, shape: tuple[int, ...], scale: float) -> jax.Array:
        return scale * jax.random.normal(rng_key, shape, dtype=jnp.float32)

    keys = iter(jax.random.split(key, 8 * num_blocks + 4))
    params: dict[str, Any] = {
        "Embed_0": {"embedding": _normal(next(keys), (vocab_size, d_model), 0.5)},
        "RMSNorm_f": {"scale": 1.0 + _normal(next(keys), (d_model,), 0.2)},
    }
    for block_index in range(num_blocks):
        params[f"Block_{block_index}"] = {
            "RMSNorm_0": {"scale": 1.0 + _normal(next(keys), (d_model,), 0.2)},
            "GQAttention_0": {
                "query": {
                    "kernel": _normal(
                        next(keys),
                        (d_model, num_kv_groups, heads_per_group, head_dim),
                        0.6,
                    )
                },
                "key": {
                    "kernel": _normal(
                        next(keys), (d_model, num_kv_groups, head_dim), 0.6
                    )
                },
                "value": {
                    "kernel": _normal(
                        next(keys), (d_model, num_kv_groups, head_dim), 0.6
                    )
                },
                "out": {
                    "kernel": _normal(
                        next(keys),
                        (num_kv_groups, heads_per_group, head_dim, d_model),
                        0.6,
                    )
                },
            },
            "RMSNorm_1": {"scale": 1.0 + _normal(next(keys), (d_model,), 0.2)},
            "FFN_0": {
                "FFN_layer0": {"kernel": _normal(next(keys), (d_model, ffn_dim), 0.6)},
                "FFN_layer1": {"kernel": _normal(next(keys), (ffn_dim, d_model), 0.6)},
            },
        }
    params["DenseLogits"] = {"kernel": _normal(next(keys), (d_model, vocab_size), 0.6)}
    return params


def _gqa_intra_reindex(problem: SymmetryGraph, forward_perms) -> dict[str, str]:
    """Reference-slot reindexing of per-slot intra groups by the kv permutation."""

    reindex: dict[str, str] = {}
    for constraint in problem.constraints:
        if not isinstance(constraint, GQARoPECircuitConstraint):
            continue
        kv_matrix = np.asarray(forward_perms[constraint.kv_group])
        sigma = np.argmax(kv_matrix, axis=1)  # new slot i took old group sigma[i]
        sigma_inv = np.argsort(sigma)
        for slot_groups in (
            constraint.query_head_groups,
            constraint.qk_groups,
            constraint.vo_groups,
        ):
            for slot, group_id in enumerate(slot_groups):
                reindex[group_id] = slot_groups[int(sigma_inv[slot])]
    return reindex


def signed_permutation_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    """Random signed permutation ``M`` with ``M[i, j_i] = sigma_i``."""

    matrix = permutation_matrix(rng.permutation(size))
    signs = rng.choice([-1.0, 1.0], size=size)
    return matrix * signs[:, None]


def rotation_pairs_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    """Random half-split per-pair rotation matrix of even ``size``."""

    half = size // 2
    angles = rng.uniform(-np.pi, np.pi, size=half)
    cos, sin = np.cos(angles), np.sin(angles)
    idx = np.arange(half)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[idx, idx] = cos
    matrix[idx + half, idx + half] = cos
    matrix[idx + half, idx] = sin
    matrix[idx, idx + half] = -sin
    return matrix


def _rmsnorm_gqa_rope_transformer_case(
    seed: int,
    *,
    name: str,
    signed: bool,
    rotations: bool,
    scales: bool,
) -> SyntheticOrbitCase:
    """Shared constructor for RMSNorm/RoPE/GQA transformer orbit cases.

    The forward transform draws, per group and within its declared class:
    permutations everywhere, signed permutations on signed/orthogonal-capable
    groups when ``signed``, per-pair qk rotations when ``rotations``
    (identity otherwise — rotation groups admit no permutations). ``scales``
    additionally composes the family's exact diagonal scales: RMSNorm gamma
    scales (signed helper), pair-tiled qk scales, and vo scales through
    ``apply_scales``. Expected aligning transform_family are the orthogonal
    inverses (transposes), with per-slot groups re-indexed by the kv
    permutation as in the flat-attention case.
    """

    from align.canonicalization.rmsnorm_gqa_rope import (
        apply_rms_gamma_scales,
        gqa_rope_circuit_constraints,
        rmsnorm_scale_constraints,
    )

    key = jax.random.PRNGKey(seed)
    param_key, input_key, perm_key = jax.random.split(key, 3)
    reference = make_rmsnorm_gqa_rope_transformer_params(key=param_key)
    recipe = RMSNormGQARoPETransformerRecipe()
    problem = recipe.build_graph(reference)

    rng = np.random.default_rng(int(jax.random.randint(perm_key, (), 0, 2**31 - 1)))
    forward: dict[str, np.ndarray] = {}
    for group_id, group in problem.groups.items():
        if group.transform_family == "rotation_pairs":
            forward[group_id] = (
                rotation_pairs_matrix(rng, group.size)
                if rotations
                else np.eye(group.size)
            )
        elif signed and group.transform_family in ("signed_permutation", "orthogonal"):
            forward[group_id] = signed_permutation_matrix(rng, group.size)
        else:
            forward[group_id] = permutation_matrix(rng.permutation(group.size))
    target = problem.apply_transforms(
        reference, TransformState.from_transforms(problem, forward)
    )

    if scales:
        gamma_scales = {
            constraint.scale: np.exp(
                rng.uniform(
                    -1.0,
                    1.0,
                    size=problem.tensors[constraint.scale].shape[0],
                )
            ).astype(np.float32)
            for constraint in rmsnorm_scale_constraints(problem)
        }
        target = apply_rms_gamma_scales(problem, target, gamma_scales)
        circuit_scales = {}
        for constraint in gqa_rope_circuit_constraints(problem):
            head_dim = constraint.head_dim
            for group_id in constraint.qk_groups:
                pair = np.exp(rng.uniform(-1.0, 1.0, size=head_dim // 2))
                circuit_scales[group_id] = np.concatenate([pair, pair]).astype(
                    np.float32
                )
            for group_id in constraint.vo_groups:
                circuit_scales[group_id] = np.exp(
                    rng.uniform(-1.0, 1.0, size=head_dim)
                ).astype(np.float32)
        target = problem.apply_scales(target, _scale_state(problem, circuit_scales))

    intra_reindex = _gqa_intra_reindex(problem, forward)
    expected: dict[str, np.ndarray] = {}
    for group_id in forward:
        source = intra_reindex.get(group_id, group_id)
        expected[group_id] = np.asarray(forward[source]).T

    tokens = jax.random.randint(input_key, (4, 5), 0, 7, dtype=jnp.int32)
    return SyntheticOrbitCase(
        name=f"{name}_seed_{seed}",
        recipe=recipe,
        problem=problem,
        reference=reference,
        target=target,
        inputs=tokens,
        apply_fn=rmsnorm_gqa_rope_transformer_apply,
        expected_transforms=expected,
    )


def make_rmsnorm_gqa_rope_transformer_orbit_case(seed: int = 0) -> SyntheticOrbitCase:
    """RMSNorm/RoPE/GQA case whose target is a known permutation copy.

    Forward permutations cover the residual stream, per-component kv-group
    permutations, per-group query-head permutations, and per-group vo
    intra-head permutations; qk rotation groups stay at identity (they admit
    no permutations). Expected aligning permutations follow the
    wreath-product inverse used by the flat-attention case: per-slot intra
    expectations are transposed and re-indexed by the kv-group permutation.
    """

    return _rmsnorm_gqa_rope_transformer_case(
        seed,
        name="rmsnorm_gqa_rope_transformer_exact_orbit",
        signed=False,
        rotations=False,
        scales=False,
    )


def make_rmsnorm_gqa_rope_transformer_scaled_orbit_case(
    seed: int = 0,
) -> SyntheticOrbitCase:
    """Orbit carrying the full implemented *discrete* + scale symmetry.

    The target composes signed permutations on the stream and vo groups
    (plain permutations elsewhere) with all exact diagonal scales: RMSNorm
    gamma scales, pair-tied qk scales, and vo circuit scales.
    Permutation-only matching cannot reach the reference;
    ``known_optimum_distance`` of 0 presumes ``canonicalize=True``. Recovery is
    exact (signs are discrete).
    """

    return _rmsnorm_gqa_rope_transformer_case(
        seed,
        name="rmsnorm_gqa_rope_transformer_scaled_orbit",
        signed=True,
        rotations=False,
        scales=True,
    )


def make_rmsnorm_gqa_rope_transformer_rotated_orbit_case(
    seed: int = 0,
) -> SyntheticOrbitCase:
    """Orbit additionally carrying random per-pair qk rotations.

    Extends the scaled orbit with random rotations on every qk
    ``rotation_pairs`` group — the full implemented symmetry of the family.
    Rotations are recovered by the closed-form per-pair projection inside
    ``lap`` sweeps; recovered transform_family match the expected transposes up to
    float32 trigonometry rather than exactly.
    """

    return _rmsnorm_gqa_rope_transformer_case(
        seed,
        name="rmsnorm_gqa_rope_transformer_rotated_orbit",
        signed=True,
        rotations=True,
        scales=True,
    )


def make_rmsnorm_gqa_rope_transformer_orthogonal_orbit_case(
    seed: int = 0,
) -> SyntheticOrbitCase:
    """Gamma-folded case whose target is an orthogonal stream copy.

    The reference is the *canonicalized* base (gammas folded to 1, circuits
    balanced); the target applies one random orthogonal matrix to the stream
    group (identity elsewhere), which is exact because the folded gammas are
    invariant. The recipe declares ``stream_transform_family="orthogonal"``, so a
    ``procrustes`` schedule step recovers the stream in closed form.
    """

    from align.canonicalization import ScaleCanonicalizer

    key = jax.random.PRNGKey(seed)
    param_key, input_key, perm_key = jax.random.split(key, 3)
    base = make_rmsnorm_gqa_rope_transformer_params(key=param_key)
    recipe = RMSNormGQARoPETransformerRecipe(stream_transform_family="orthogonal")
    problem = recipe.build_graph(base)
    reference, _, _ = ScaleCanonicalizer().canonicalize(problem, base)

    rng = np.random.default_rng(int(jax.random.randint(perm_key, (), 0, 2**31 - 1)))
    d_model = problem.groups["stream"].size
    q_matrix, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    forward = {"stream": q_matrix}
    target = problem.apply_transforms(
        reference, TransformState.from_transforms(problem, forward)
    )

    expected = {
        group_id: np.asarray(forward[group_id]).T
        if group_id in forward
        else np.eye(group.size)
        for group_id, group in problem.groups.items()
    }
    tokens = jax.random.randint(input_key, (4, 5), 0, 7, dtype=jnp.int32)
    return SyntheticOrbitCase(
        name=f"rmsnorm_gqa_rope_transformer_orthogonal_orbit_seed_{seed}",
        recipe=recipe,
        problem=problem,
        reference=reference,
        target=target,
        inputs=tokens,
        apply_fn=rmsnorm_gqa_rope_transformer_apply,
        expected_transforms=expected,
    )


def _make_dense_params(
    *,
    key: jax.Array,
    sizes: Sequence[int],
) -> dict[str, Any]:
    keys = jax.random.split(key, len(sizes) - 1)
    fcn: dict[str, dict[str, jax.Array]] = {}
    for idx, (layer_key, in_dim, out_dim) in enumerate(
        zip(keys, sizes[:-1], sizes[1:], strict=True)
    ):
        weight_key, bias_key = jax.random.split(layer_key)
        column_signature = jnp.linspace(-0.25, 0.25, out_dim, dtype=jnp.float32)
        fcn[f"Dense_{idx}"] = {
            "kernel": 0.7
            * jax.random.normal(weight_key, (in_dim, out_dim), dtype=jnp.float32)
            + column_signature[None, :],
            "bias": 0.1 * jax.random.normal(bias_key, (out_dim,), dtype=jnp.float32)
            + 0.05 * column_signature,
        }
    return {"params": {"fcn": fcn}}


def _scale_state(
    problem: SymmetryGraph, scales: Mapping[str, np.ndarray]
) -> ScaleState:
    """Build a :class:`ScaleState` from per-group positive scale vectors."""

    return ScaleState.from_scales(problem, dict(scales))


def _inverse_permutations(
    transforms: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {gid: np.asarray(perm).T for gid, perm in transforms.items()}


def make_mlp_orbit_case(seed: int = 0) -> SyntheticOrbitCase:
    """Build a dense MLP case with known scales and hidden permutations."""

    key = jax.random.PRNGKey(seed)
    param_key, input_key = jax.random.split(key)
    reference = _make_dense_params(key=param_key, sizes=(3, 5, 4, 2))
    recipe = MLPRecipe()
    problem = recipe.build_graph(reference)

    scales = {
        "mlp/h0": np.array([1.7, 0.6, 2.3, 1.1, 0.8], dtype=np.float32),
        "mlp/h1": np.array([0.7, 1.6, 1.2, 2.1], dtype=np.float32),
    }
    forward_perms = {
        "mlp/h0": permutation_matrix([2, 4, 1, 0, 3]),
        "mlp/h1": permutation_matrix([3, 1, 0, 2]),
    }
    scaled = problem.apply_scales(reference, _scale_state(problem, scales))
    target = problem.apply_transforms(
        scaled, TransformState.from_transforms(problem, forward_perms)
    )
    inputs = jax.random.normal(input_key, (13, 3), dtype=jnp.float32)

    return SyntheticOrbitCase(
        name=f"mlp_exact_orbit_seed_{seed}",
        recipe=recipe,
        problem=problem,
        reference=reference,
        target=target,
        inputs=inputs,
        apply_fn=mlp_apply,
        expected_transforms=_inverse_permutations(forward_perms),
    )


def make_residual_conv_orbit_case(seed: int = 0) -> SyntheticOrbitCase:
    """Build a residual 1x1-conv case with a known tied channel stream."""

    key = jax.random.PRNGKey(seed)
    k0, k1, k2, kd, input_key = jax.random.split(key, 5)
    residual_channels = 4
    branch_channels = 3
    reference = {
        "core": {
            "Conv_0": {
                "kernel": jax.random.normal(
                    k0, (1, 1, 2, residual_channels), dtype=jnp.float32
                ),
                "bias": 0.1
                * jax.random.normal(k0, (residual_channels,), dtype=jnp.float32),
            },
            "Conv_1": {
                "kernel": jax.random.normal(
                    k1,
                    (1, 1, residual_channels, branch_channels),
                    dtype=jnp.float32,
                ),
                "bias": 0.1
                * jax.random.normal(k1, (branch_channels,), dtype=jnp.float32),
            },
            "Conv_2": {
                "kernel": jax.random.normal(
                    k2,
                    (1, 1, branch_channels, residual_channels),
                    dtype=jnp.float32,
                ),
                "bias": 0.1
                * jax.random.normal(k2, (residual_channels,), dtype=jnp.float32),
            },
            "Dense_0": {
                "kernel": jax.random.normal(
                    kd, (residual_channels, branch_channels), dtype=jnp.float32
                ),
                "bias": 0.1
                * jax.random.normal(kd, (branch_channels,), dtype=jnp.float32),
            },
        }
    }
    residual_topology = {
        "nodes": [
            {
                "name": "residual_add",
                "type": "add",
                "inputs": ["core/Conv_0", "core/Conv_2"],
            }
        ]
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=residual_topology
    )
    problem = recipe.build_graph(reference)
    group_id = problem.metadata["group_order"][0]
    forward_perms = {group_id: permutation_matrix([2, 0, 3, 1])}
    target = problem.apply_transforms(
        reference, TransformState.from_transforms(problem, forward_perms)
    )
    inputs = jax.random.normal(input_key, (7, 3, 3, 2), dtype=jnp.float32)

    return SyntheticOrbitCase(
        name=f"residual_conv_exact_orbit_seed_{seed}",
        recipe=recipe,
        problem=problem,
        reference=reference,
        target=target,
        inputs=inputs,
        apply_fn=_residual_conv_apply,
        expected_transforms=_inverse_permutations(forward_perms),
        expected_residual_channel_ties=(("core/Conv_0", "core/Conv_2"),),
    )


def make_split_concat_conv_orbit_case(seed: int = 0) -> SyntheticOrbitCase:
    """Build a branch/fan-out 1x1-conv case with sliced concat head bindings."""

    key = jax.random.PRNGKey(seed)
    k_stem, k_a, k_b, k_dense, input_key = jax.random.split(key, 5)
    input_channels = 2
    stem_channels = 3
    branch_a_channels = 2
    branch_b_channels = 4
    output_channels = 2
    reference = {
        "core": {
            "Stem": {
                "kernel": jax.random.normal(
                    k_stem,
                    (1, 1, input_channels, stem_channels),
                    dtype=jnp.float32,
                ),
                # Deterministic channel anchors keep the exact-orbit LAP probe
                # focused on fan-out/concat bindings instead of a seed-specific
                # coordinate-descent local optimum.
                "bias": jnp.linspace(2.0, 10.0, stem_channels, dtype=jnp.float32),
            },
            "BranchA": {
                "kernel": jax.random.normal(
                    k_a,
                    (1, 1, stem_channels, branch_a_channels),
                    dtype=jnp.float32,
                ),
                "bias": jnp.linspace(2.0, 10.0, branch_a_channels, dtype=jnp.float32),
            },
            "BranchB": {
                "kernel": jax.random.normal(
                    k_b,
                    (1, 1, stem_channels, branch_b_channels),
                    dtype=jnp.float32,
                ),
                "bias": jnp.linspace(2.0, 11.0, branch_b_channels, dtype=jnp.float32),
            },
            "Dense_0": {
                "kernel": jax.random.normal(
                    k_dense,
                    (branch_a_channels + branch_b_channels, output_channels),
                    dtype=jnp.float32,
                ),
                "bias": 0.1
                * jax.random.normal(k_dense, (output_channels,), dtype=jnp.float32),
            },
        }
    }
    groups = {
        "stem": SymmetryGroup(id="stem", size=stem_channels),
        "branch_a": SymmetryGroup(id="branch_a", size=branch_a_channels),
        "branch_b": SymmetryGroup(id="branch_b", size=branch_b_channels),
    }

    def _tensor(path: tuple[str, ...]) -> TensorSpec:
        tensor_id = "/".join(path)
        return TensorSpec(
            id=tensor_id,
            path=path,
            shape=tuple(int(dim) for dim in np.shape(_descend(reference, path))),
        )

    tensor_paths = (
        ("core", "Stem", "kernel"),
        ("core", "Stem", "bias"),
        ("core", "BranchA", "kernel"),
        ("core", "BranchA", "bias"),
        ("core", "BranchB", "kernel"),
        ("core", "BranchB", "bias"),
        ("core", "Dense_0", "kernel"),
        ("core", "Dense_0", "bias"),
    )
    tensors = {"/".join(path): _tensor(path) for path in tensor_paths}
    problem = SymmetryGraph(
        groups=groups,
        tensors=tensors,
        axis_bindings=(
            AxisBinding("core/Stem/kernel", axis=-1, group="stem", role="out"),
            AxisBinding("core/Stem/bias", axis=0, group="stem", role="out"),
            AxisBinding("core/BranchA/kernel", axis=-2, group="stem", role="in"),
            AxisBinding("core/BranchA/kernel", axis=-1, group="branch_a", role="out"),
            AxisBinding("core/BranchA/bias", axis=0, group="branch_a", role="out"),
            AxisBinding("core/BranchB/kernel", axis=-2, group="stem", role="in"),
            AxisBinding("core/BranchB/kernel", axis=-1, group="branch_b", role="out"),
            AxisBinding("core/BranchB/bias", axis=0, group="branch_b", role="out"),
            AxisBinding(
                "core/Dense_0/kernel",
                axis=0,
                group="branch_a",
                start=0,
                stop=branch_a_channels,
                role="in",
            ),
            AxisBinding(
                "core/Dense_0/kernel",
                axis=0,
                group="branch_b",
                start=branch_a_channels,
                stop=branch_a_channels + branch_b_channels,
                role="in",
            ),
        ),
        metadata={
            "architecture": "split_concat_conv",
            "group_order": tuple(groups),
        },
    )
    problem.validate(reference)
    forward_perms = {
        "stem": permutation_matrix([2, 0, 1]),
        "branch_a": permutation_matrix([1, 0]),
        "branch_b": permutation_matrix([2, 0, 3, 1]),
    }
    target = problem.apply_transforms(
        reference,
        TransformState(group_order=problem.group_order, matrices=forward_perms),
    )
    inputs = jax.random.normal(input_key, (9, 3, 3, input_channels), dtype=jnp.float32)

    return SyntheticOrbitCase(
        name=f"split_concat_conv_exact_orbit_seed_{seed}",
        recipe=None,
        problem=problem,
        reference=reference,
        target=target,
        inputs=inputs,
        apply_fn=_split_concat_conv_apply,
        expected_transforms=_inverse_permutations(forward_perms),
    )


def transform_validity_error(
    transforms: Mapping[str, Any], problem: SymmetryGraph | None = None
) -> float:
    """Measure hard group-transform validity per declared class.

    Plain permutation groups are checked for row/column sums and
    integrality; signed-permutation groups apply the same check to the
    entrywise absolute value; rotation-pair and orthogonal groups report the
    orthogonality residual ``max |M M^T - I|``.
    """

    worst = 0.0
    for group_id, transform in transforms.items():
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return float("inf")
        transform_family = (
            problem.groups[group_id].transform_family if problem is not None else None
        )
        if transform_family in ("rotation_pairs", "orthogonal"):
            residual = np.max(np.abs(matrix @ matrix.T - np.eye(matrix.shape[0])))
            worst = max(worst, float(residual))
            continue
        if transform_family == "signed_permutation":
            matrix = np.abs(matrix)
        row_error = np.max(np.abs(np.sum(matrix, axis=1) - 1.0))
        col_error = np.max(np.abs(np.sum(matrix, axis=0) - 1.0))
        binary_error = np.max(np.minimum(np.abs(matrix), np.abs(matrix - 1.0)))
        worst = max(worst, float(row_error), float(col_error), float(binary_error))
    return worst


def recovered_transform_error(
    actual: Mapping[str, Any], expected: Mapping[str, np.ndarray]
) -> float:
    """Return the worst entrywise error against known exact transforms."""

    worst = 0.0
    for group_id, expected_transform in expected.items():
        if group_id not in actual:
            return float("inf")
        actual_transform = np.asarray(actual[group_id], dtype=np.float64)
        expected_arr = np.asarray(expected_transform, dtype=np.float64)
        if actual_transform.shape != expected_arr.shape:
            return float("inf")
        worst = max(worst, float(np.max(np.abs(actual_transform - expected_arr))))
    return worst


def residual_constraint_violations(
    problem: SymmetryGraph, expected_ties: Sequence[Sequence[str]]
) -> int:
    """Count expected residual channel ties that are not represented by one group."""

    conv_to_group: dict[str, str] = {}
    for binding in problem.axis_bindings:
        if (
            binding.role == "out"
            and binding.tensor_id.endswith("/kernel")
            and len(problem.tensors[binding.tensor_id].shape) >= 2
        ):
            conv_to_group[binding.tensor_id[: -len("/kernel")]] = binding.group

    violations = 0
    for tied_convs in expected_ties:
        groups = [conv_to_group.get(conv) for conv in tied_convs]
        if any(group is None for group in groups):
            violations += 1
            continue
        if len(set(groups)) != 1:
            violations += 1
    return violations


def _canonicalize_if_requested(
    case: SyntheticOrbitCase, params: ParamTree, enabled: bool
) -> ParamTree:
    if not enabled:
        return params
    canonical, _, _ = ScaleCanonicalizer().canonicalize(
        case.problem,
        params,
        include_bias_in_norm=True,
    )
    return canonical


def run_alignment_benchmark(
    case: SyntheticOrbitCase,
    *,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[Mapping[str, Any]] | None = None,
    canonicalize: bool = False,
    rng_seed: int = 0,
) -> AlignmentBenchmarkResult:
    """Run one synthetic alignment case and compute correctness metrics."""

    reference = _canonicalize_if_requested(case, case.reference, canonicalize)
    target = _canonicalize_if_requested(case, case.target, canonicalize)
    resolved_kwargs = resolve_calibration_kwargs(
        objective_kwargs,
        problem=case.problem,
        params=reference,
        apply_fn=case.apply_fn,
        inputs=case.inputs,
    )
    aligned, transforms, diagnostics = match_sample(
        case.problem,
        reference,
        target,
        objective=objective,
        objective_kwargs=resolved_kwargs,
        schedule=schedule or default_lap_schedule(),
        rng_key=jax.random.PRNGKey(rng_seed),
    )

    target_predictions = np.asarray(case.apply_fn(case.target, case.inputs))
    aligned_predictions = np.asarray(case.apply_fn(aligned, case.inputs))
    function_drift = float(np.max(np.abs(target_predictions - aligned_predictions)))
    before = tree_l2_distance(reference, target)
    after = tree_l2_distance(reference, aligned)
    optimum = case.known_optimum_distance
    optimality_gap = None if optimum is None else max(after - optimum, 0.0)

    metrics = AlignmentBenchmarkMetrics(
        function_drift_max=function_drift,
        distance_before=before,
        distance_after=after,
        distance_reduction=before - after,
        optimality_gap=optimality_gap,
        transform_validity_error=transform_validity_error(transforms, case.problem),
        recovered_transform_error=recovered_transform_error(
            transforms, case.expected_transforms
        ),
        residual_constraint_violations=residual_constraint_violations(
            case.problem, case.expected_residual_channel_ties
        ),
    )
    return AlignmentBenchmarkResult(
        aligned_params=aligned,
        transforms=transforms,
        diagnostics=diagnostics,
        metrics=metrics,
    )


@dataclass(frozen=True)
class RobustnessRecord:
    """One (case, schedule, seed) cell of a robustness sweep."""

    case_name: str
    schedule_name: str
    seed: int
    metrics: AlignmentBenchmarkMetrics


def default_schedule_grid(
    *,
    lap_max_sweeps: int = 25,
    sinkhorn_max_steps: int = 150,
    sinkhorn_lr: float = 0.1,
    sinkhorn_tau: float = 0.3,
) -> dict[str, list[dict[str, Any]]]:
    """Return the named solver schedules exercised by robustness sweeps."""

    lap_step = {"solver": "lap", "max_sweeps": lap_max_sweeps, "tolerance": 0.0}
    sinkhorn_step = {
        "solver": "sinkhorn",
        "max_steps": sinkhorn_max_steps,
        "learning_rate": sinkhorn_lr,
        "tau": sinkhorn_tau,
        "sinkhorn_iterations": 20,
        "init_scale": 0.01,
        "tolerance": 0.0,
    }
    return {
        "lap": [dict(lap_step)],
        "sinkhorn": [dict(sinkhorn_step)],
        "lap_sinkhorn_lap": [
            dict(lap_step),
            dict(sinkhorn_step),
            {"solver": "lap", "max_sweeps": 5, "tolerance": 0.0},
        ],
    }


def run_robustness_sweep(
    *,
    seeds: Sequence[int],
    schedules: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    case_factory: Callable[..., SyntheticOrbitCase] = make_mlp_orbit_case,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    canonicalize: bool = True,
) -> list[RobustnessRecord]:
    """Sweep exact-orbit seeds against named solver schedules.

    Covers LAP, pure Sinkhorn, and mixed schedules by default so solver
    robustness is measured for every scheduling mode the pipeline supports.
    """

    schedules = schedules or default_schedule_grid()
    records: list[RobustnessRecord] = []
    for seed in seeds:
        case = case_factory(seed=seed)
        for schedule_name, schedule in schedules.items():
            result = run_alignment_benchmark(
                case,
                objective=objective,
                objective_kwargs=objective_kwargs,
                schedule=schedule,
                canonicalize=canonicalize,
                rng_seed=seed,
            )
            records.append(
                RobustnessRecord(
                    case_name=case.name,
                    schedule_name=schedule_name,
                    seed=seed,
                    metrics=result.metrics,
                )
            )
    return records


def measure_matching_performance(
    case: SyntheticOrbitCase,
    *,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[Mapping[str, Any]] | None = None,
    canonicalize: bool = False,
    warm_repetitions: int = 3,
    rng_seed: int = 0,
) -> PerformanceMeasurement:
    """Measure cold time, warm throughput, peak memory, and first-call overhead."""

    if warm_repetitions < 1:
        raise ValueError("warm_repetitions must be at least 1.")

    tracemalloc.start()
    try:
        cold_start = time.perf_counter()
        run_alignment_benchmark(
            case,
            objective=objective,
            objective_kwargs=objective_kwargs,
            schedule=schedule,
            canonicalize=canonicalize,
            rng_seed=rng_seed,
        )
        cold_time = time.perf_counter() - cold_start

        warm_start = time.perf_counter()
        for idx in range(warm_repetitions):
            run_alignment_benchmark(
                case,
                objective=objective,
                objective_kwargs=objective_kwargs,
                schedule=schedule,
                canonicalize=canonicalize,
                rng_seed=rng_seed + idx + 1,
            )
        warm_time = time.perf_counter() - warm_start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    warm_per_sample = warm_time / warm_repetitions
    throughput = warm_repetitions / warm_time if warm_time > 0.0 else float("inf")
    compilation_time = max(cold_time - warm_per_sample, 0.0)
    return PerformanceMeasurement(
        cold_time_s=float(cold_time),
        warm_throughput_samples_s=float(throughput),
        peak_memory_bytes=int(peak),
        compilation_time_s=float(compilation_time),
    )
