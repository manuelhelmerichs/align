"""Activation-Gram (relative-Fisher) metrics for weight matching.

For a linear module with incoming activation ``a`` and weight residual
``Delta W``, the expected squared pre-activation change is

    E ||a @ Delta W||^2 = tr(Delta W.T @ E[a a.T] @ Delta W).

The helpers in this module store that activation Gram together with the tensor
axes it acts on.  This explicit contract is important for convolutional and
structured kernels, where the incoming coordinates may span several axes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _coerce_sample_spec(tensor_id: str, spec: Mapping[str, Any]):
    unknown = set(spec) - {"input_axes", "activations"}
    if unknown:
        raise ValueError(
            f"Activation sample spec for tensor {tensor_id!r} has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    if "input_axes" not in spec:
        raise ValueError(
            f"Activation sample spec for tensor {tensor_id!r} needs 'input_axes'."
        )
    axes = tuple(int(axis) for axis in spec["input_axes"])
    activations = spec.get("activations")
    return axes, activations


def estimate_activation_gram_metrics(
    graph,
    params: Mapping[str, Any],
    activation_fn: Callable[[Mapping[str, Any], Any], Mapping[str, Mapping[str, Any]]],
    inputs: Any,
    *,
    damping: float = 1.0,
    normalize: bool = True,
    max_gram_bytes: int = 512 * 1024**2,
) -> dict[str, dict[str, Any]]:
    """Estimate fixed activation-Gram metrics at ``params``.

    ``activation_fn(params, inputs)`` must return one entry for every graph
    tensor.  Each entry has ``input_axes`` (tensor axes flattened into the
    incoming coordinate) and ``activations`` with shape
    ``(...samples, *input_shape)``.  For tensors with no incoming axes (biases,
    normalization vectors, and other Euclidean terms), set ``input_axes=()``;
    their metric is the scalar identity and ``activations`` may be ``None``.

    Damping adds one global ``damping * mean_metric * I`` floor to every Gram,
    including scalar Euclidean terms.  Normalization then makes the average
    coordinate weight, counted with each tensor's multiplicity, equal to one.
    """

    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}.")
    samples = dict(activation_fn(params, inputs))
    expected_ids = set(graph.tensors)
    if set(samples) != expected_ids:
        missing = sorted(expected_ids - set(samples))
        extra = sorted(set(samples) - expected_ids)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise ValueError(
            "Activation samples must cover every graph tensor ("
            + "; ".join(details)
            + ")."
        )

    metrics: dict[str, dict[str, Any]] = {}
    weighted_trace = 0.0
    coordinate_count = 0
    for tensor_id, tensor_spec in graph.tensors.items():
        axes, activations = _coerce_sample_spec(tensor_id, samples[tensor_id])
        ndim = len(tensor_spec.shape)
        invalid = [axis for axis in axes if axis < -ndim or axis >= ndim]
        if invalid:
            raise ValueError(
                f"Activation sample axes for tensor {tensor_id!r} are out of bounds "
                f"for {ndim} dimensions: {invalid}."
            )
        normalized_axes = tuple(axis % ndim for axis in axes)
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError(
                f"Activation sample axes for tensor {tensor_id!r} are not unique: "
                f"{axes}."
            )
        input_shape = tuple(tensor_spec.shape[axis] for axis in normalized_axes)
        input_size = int(np.prod(input_shape, dtype=np.int64)) if input_shape else 1
        gram_bytes = input_size * input_size * np.dtype(np.float64).itemsize
        if gram_bytes > int(max_gram_bytes):
            raise MemoryError(
                f"Activation Gram for tensor {tensor_id!r} would require "
                f"{gram_bytes / 1024**2:.1f} MiB, exceeding the configured "
                f"limit of {max_gram_bytes / 1024**2:.1f} MiB. Reduce the "
                "declared input axes or use a structured/chunked metric."
            )
        if not normalized_axes:
            if activations is not None:
                arr = np.asarray(activations)
                if arr.size and not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"Activations for tensor {tensor_id!r} contain non-finite values."
                    )
            gram = np.ones((1, 1), dtype=np.float64)
        else:
            if activations is None:
                raise ValueError(
                    f"Activation sample spec for tensor {tensor_id!r} needs "
                    "'activations' when input_axes is non-empty."
                )
            arr = np.asarray(activations, dtype=np.float64)
            if (
                arr.ndim < len(input_shape)
                or tuple(arr.shape[-len(input_shape) :]) != input_shape
            ):
                raise ValueError(
                    f"Activations for tensor {tensor_id!r} have shape {arr.shape}; "
                    f"expected trailing input shape {input_shape}."
                )
            flat = arr.reshape((-1, input_size))
            if flat.shape[0] == 0:
                raise ValueError(
                    f"Activations for tensor {tensor_id!r} contain no samples."
                )
            if not np.all(np.isfinite(flat)):
                raise ValueError(
                    f"Activations for tensor {tensor_id!r} contain non-finite values."
                )
            gram = flat.T @ flat / float(flat.shape[0])
        multiplicity = int(np.prod(tensor_spec.shape, dtype=np.int64)) // input_size
        weighted_trace += multiplicity * float(np.trace(gram))
        coordinate_count += int(np.prod(tensor_spec.shape, dtype=np.int64))
        metrics[tensor_id] = {
            "input_axes": normalized_axes,
            "gram": np.asarray(gram, dtype=np.float64),
        }

    mean_metric = weighted_trace / coordinate_count
    if mean_metric <= 0.0:
        raise ValueError("The complete activation-Gram metric is identically zero.")
    scale = mean_metric * (1.0 + damping) if normalize else 1.0
    metrics = {
        tensor_id: {
            "input_axes": metric["input_axes"],
            "gram": (
                metric["gram"]
                + damping
                * mean_metric
                * np.eye(metric["gram"].shape[0], dtype=np.float64)
            )
            / scale,
        }
        for tensor_id, metric in metrics.items()
    }
    return metrics


def save_activation_gram_metrics_npz(
    metrics: Mapping[str, Mapping[str, Any]], path: str | Path
) -> None:
    """Serialize activation-Gram metrics without pickle/object arrays."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = []
    arrays: dict[str, np.ndarray] = {}
    for index, (tensor_id, metric) in enumerate(metrics.items()):
        key = f"gram_{index}"
        metadata.append(
            {
                "tensor_id": str(tensor_id),
                "input_axes": [int(axis) for axis in metric["input_axes"]],
                "key": key,
            }
        )
        arrays[key] = np.asarray(metric["gram"])
    arrays["__metadata__"] = np.asarray(
        json.dumps({"format": "align.activation_gram.v1", "metrics": metadata})
    )
    np.savez(target, **arrays)


def load_activation_gram_metrics_npz(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load the explicit activation-Gram NPZ contract."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if "__metadata__" not in payload:
            raise ValueError(
                f"Activation-Gram file {path!s} lacks the '__metadata__' entry."
            )
        metadata = json.loads(str(payload["__metadata__"]))
        if metadata.get("format") != "align.activation_gram.v1":
            raise ValueError(
                f"Unsupported activation-Gram format {metadata.get('format')!r}."
            )
        result = {}
        for entry in metadata["metrics"]:
            key = str(entry["key"])
            if key not in payload:
                raise ValueError(f"Activation-Gram file {path!s} lacks array {key!r}.")
            result[str(entry["tensor_id"])] = {
                "input_axes": tuple(int(axis) for axis in entry["input_axes"]),
                "gram": np.asarray(payload[key]),
            }
        return result


def resolve_relative_fisher_calibration_kwargs(
    objective_kwargs: Mapping[str, Any] | None,
    *,
    graph,
    params: Mapping[str, Any],
    activation_fn: Callable[[Mapping[str, Any], Any], Mapping[str, Mapping[str, Any]]]
    | None,
    inputs: Any,
) -> dict[str, Any]:
    """Resolve ``calibration='relative_fisher'`` into tensor metrics."""

    kwargs = dict(objective_kwargs or {})
    spec = kwargs.get("calibration")
    if spec != "relative_fisher":
        return kwargs
    kwargs.pop("calibration")
    damping = float(kwargs.pop("calibration_damping", 1.0))
    if activation_fn is None or inputs is None:
        raise ValueError(
            "calibration='relative_fisher' requires an activation_fn and "
            "calibration inputs; for real pipelines, precompute metrics and "
            "pass 'metrics_path' instead."
        )
    kwargs["tensor_metrics"] = estimate_activation_gram_metrics(
        graph,
        params,
        activation_fn,
        inputs,
        damping=damping,
    )
    return kwargs


__all__ = [
    "estimate_activation_gram_metrics",
    "load_activation_gram_metrics_npz",
    "resolve_relative_fisher_calibration_kwargs",
    "save_activation_gram_metrics_npz",
]
