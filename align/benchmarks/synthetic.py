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

from align.alignment import (
    AlignmentProblem,
    AxisBinding,
    PermutationGroup,
    TensorSpec,
)
from align.architectures import (
    ArchitectureAdapter,
    DenseMLPAdapter,
    ResNetAdapter,
)
from align.normalization import ScaleNormalizer, ScaleState
from align.rebasin import (
    PermutationState,
    default_lap_schedule,
    rebasin_single_sample,
)

ParamTree = Mapping[str, Any]
ApplyFn = Callable[[ParamTree, jax.Array], jax.Array]


@dataclass
class SyntheticOrbitCase:
    """A synthetic pair of functionally equivalent parameter trees."""

    name: str
    adapter: ArchitectureAdapter | None
    problem: AlignmentProblem
    reference: ParamTree
    target: ParamTree
    inputs: jax.Array
    apply_fn: ApplyFn
    expected_permutations: Mapping[str, np.ndarray]
    expected_residual_ties: tuple[tuple[str, ...], ...] = ()
    known_optimum_distance: float | None = 0.0


@dataclass(frozen=True)
class AlignmentBenchmarkMetrics:
    """Quality metrics for one exact synthetic alignment run."""

    function_drift_max: float
    distance_before: float
    distance_after: float
    distance_reduction: float
    optimality_gap: float | None
    permutation_validity_error: float
    recovered_permutation_error: float
    residual_constraint_violations: int


@dataclass
class AlignmentBenchmarkResult:
    """Output of an alignment benchmark run."""

    aligned_params: ParamTree
    permutations: Mapping[str, Any]
    aux: dict[str, Any] | None
    metrics: AlignmentBenchmarkMetrics


@dataclass(frozen=True)
class PerformanceMeasurement:
    """Basic timing and memory measurements for a rebasin benchmark."""

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


def dense_mlp_apply(params: ParamTree, x: jax.Array) -> jax.Array:
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
    problem: AlignmentProblem, scales: Mapping[str, np.ndarray]
) -> ScaleState:
    """Build a :class:`ScaleState` from per-group positive scale vectors."""

    return ScaleState.from_scales(problem, dict(scales))


def _inverse_permutations(perms: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {gid: np.asarray(perm).T for gid, perm in perms.items()}


def make_dense_mlp_orbit_case(seed: int = 0) -> SyntheticOrbitCase:
    """Build a dense MLP case with known scales and hidden permutations."""

    key = jax.random.PRNGKey(seed)
    param_key, input_key = jax.random.split(key)
    reference = _make_dense_params(key=param_key, sizes=(3, 5, 4, 2))
    adapter = DenseMLPAdapter()
    problem = adapter.build_problem(reference)

    scales = {
        "layer_0": np.array([1.7, 0.6, 2.3, 1.1, 0.8], dtype=np.float32),
        "layer_1": np.array([0.7, 1.6, 1.2, 2.1], dtype=np.float32),
    }
    forward_perms = {
        "layer_0": permutation_matrix([2, 4, 1, 0, 3]),
        "layer_1": permutation_matrix([3, 1, 0, 2]),
    }
    scaled = problem.apply_scales(reference, _scale_state(problem, scales))
    target = problem.apply(scaled, PermutationState.from_perms(problem, forward_perms))
    inputs = jax.random.normal(input_key, (13, 3), dtype=jnp.float32)

    return SyntheticOrbitCase(
        name=f"dense_mlp_exact_orbit_seed_{seed}",
        adapter=adapter,
        problem=problem,
        reference=reference,
        target=target,
        inputs=inputs,
        apply_fn=dense_mlp_apply,
        expected_permutations=_inverse_permutations(forward_perms),
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
    module_graph = {
        "nodes": [
            {
                "name": "residual_add",
                "type": "add",
                "inputs": ["core/Conv_0", "core/Conv_2"],
            }
        ]
    }
    adapter = ResNetAdapter(layer_root="core", module_graph=module_graph)
    problem = adapter.build_problem(reference)
    group_id = problem.metadata["group_order"][0]
    forward_perms = {group_id: permutation_matrix([2, 0, 3, 1])}
    target = problem.apply(
        reference, PermutationState.from_perms(problem, forward_perms)
    )
    inputs = jax.random.normal(input_key, (7, 3, 3, 2), dtype=jnp.float32)

    return SyntheticOrbitCase(
        name=f"residual_conv_exact_orbit_seed_{seed}",
        adapter=adapter,
        problem=problem,
        reference=reference,
        target=target,
        inputs=inputs,
        apply_fn=_residual_conv_apply,
        expected_permutations=_inverse_permutations(forward_perms),
        expected_residual_ties=(("core/Conv_0", "core/Conv_2"),),
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
                "bias": 0.1
                * jax.random.normal(k_stem, (stem_channels,), dtype=jnp.float32),
            },
            "BranchA": {
                "kernel": jax.random.normal(
                    k_a,
                    (1, 1, stem_channels, branch_a_channels),
                    dtype=jnp.float32,
                ),
                "bias": 0.1
                * jax.random.normal(k_a, (branch_a_channels,), dtype=jnp.float32),
            },
            "BranchB": {
                "kernel": jax.random.normal(
                    k_b,
                    (1, 1, stem_channels, branch_b_channels),
                    dtype=jnp.float32,
                ),
                "bias": 0.1
                * jax.random.normal(k_b, (branch_b_channels,), dtype=jnp.float32),
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
        "stem": PermutationGroup(id="stem", size=stem_channels),
        "branch_a": PermutationGroup(id="branch_a", size=branch_a_channels),
        "branch_b": PermutationGroup(id="branch_b", size=branch_b_channels),
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
    problem = AlignmentProblem(
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
    target = problem.apply(
        reference,
        PermutationState(group_order=problem.group_order, hard=forward_perms),
    )
    inputs = jax.random.normal(input_key, (9, 3, 3, input_channels), dtype=jnp.float32)

    return SyntheticOrbitCase(
        name=f"split_concat_conv_exact_orbit_seed_{seed}",
        adapter=None,
        problem=problem,
        reference=reference,
        target=target,
        inputs=inputs,
        apply_fn=_split_concat_conv_apply,
        expected_permutations=_inverse_permutations(forward_perms),
    )


def permutation_validity_error(perms: Mapping[str, Any]) -> float:
    """Measure hard permutation validity using row/column sums and integrality."""

    worst = 0.0
    for perm in perms.values():
        matrix = np.asarray(perm, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return float("inf")
        row_error = np.max(np.abs(np.sum(matrix, axis=1) - 1.0))
        col_error = np.max(np.abs(np.sum(matrix, axis=0) - 1.0))
        binary_error = np.max(np.minimum(np.abs(matrix), np.abs(matrix - 1.0)))
        worst = max(worst, float(row_error), float(col_error), float(binary_error))
    return worst


def recovered_permutation_error(
    actual: Mapping[str, Any], expected: Mapping[str, np.ndarray]
) -> float:
    """Return the worst entrywise error against known exact permutations."""

    worst = 0.0
    for group_id, expected_perm in expected.items():
        if group_id not in actual:
            return float("inf")
        actual_perm = np.asarray(actual[group_id], dtype=np.float64)
        expected_arr = np.asarray(expected_perm, dtype=np.float64)
        if actual_perm.shape != expected_arr.shape:
            return float("inf")
        worst = max(worst, float(np.max(np.abs(actual_perm - expected_arr))))
    return worst


def residual_constraint_violations(
    problem: AlignmentProblem, expected_ties: Sequence[Sequence[str]]
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


def _normalize_if_requested(
    case: SyntheticOrbitCase, params: ParamTree, normalize: bool
) -> ParamTree:
    if not normalize:
        return params
    normalized, _, _ = ScaleNormalizer().normalize(
        case.problem,
        params,
        task_type="regression",
        normalize_biases=True,
    )
    return normalized


def run_alignment_benchmark(
    case: SyntheticOrbitCase,
    *,
    objective: str = "l2_weight",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[Mapping[str, Any]] | None = None,
    normalize: bool = False,
    rng_seed: int = 0,
) -> AlignmentBenchmarkResult:
    """Run one synthetic alignment case and compute correctness metrics."""

    reference = _normalize_if_requested(case, case.reference, normalize)
    target = _normalize_if_requested(case, case.target, normalize)
    aligned, perms, aux = rebasin_single_sample(
        case.problem,
        reference,
        target,
        objective=objective,
        objective_kwargs=dict(objective_kwargs or {}),
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
        permutation_validity_error=permutation_validity_error(perms),
        recovered_permutation_error=recovered_permutation_error(
            perms, case.expected_permutations
        ),
        residual_constraint_violations=residual_constraint_violations(
            case.problem, case.expected_residual_ties
        ),
    )
    return AlignmentBenchmarkResult(
        aligned_params=aligned,
        permutations=perms,
        aux=aux,
        metrics=metrics,
    )


def run_lap_robustness_sweep(
    *,
    seeds: Sequence[int],
    max_sweeps_values: Sequence[int],
) -> list[tuple[int, int, AlignmentBenchmarkMetrics]]:
    """Sweep dense exact-orbit seeds and LAP sweep budgets."""

    results: list[tuple[int, int, AlignmentBenchmarkMetrics]] = []
    for seed in seeds:
        case = make_dense_mlp_orbit_case(seed=seed)
        for max_sweeps in max_sweeps_values:
            result = run_alignment_benchmark(
                case,
                schedule=[{"solver": "lap", "max_sweeps": int(max_sweeps), "tol": 0.0}],
                normalize=True,
                rng_seed=seed,
            )
            results.append((seed, int(max_sweeps), result.metrics))
    return results


def measure_rebasin_performance(
    case: SyntheticOrbitCase,
    *,
    objective: str = "l2_weight",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[Mapping[str, Any]] | None = None,
    normalize: bool = False,
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
            normalize=normalize,
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
                normalize=normalize,
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
