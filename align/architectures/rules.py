"""Composable component rules: the unit of alignment-problem construction.

A component rule emits one named component (its permutation groups, axis bindings,
constraints, and :class:`~align.symmetry.ComponentSpec`) into a shared
:class:`~align.architectures.graph_builder.SymmetryGraphBuilder`. Architecture recipes are
thin recipes that discover where components live in a parameter tree and compose
component rules; all actual construction happens here. Because identical components
from different architectures share emission code, their component signatures match
by construction and can be aligned across networks
(:func:`align.matching.match_component_across`).
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..symmetry import GraphConstraint
from ..symmetry.tensor_ops import _descend
from .graph_builder import SymmetryGraphBuilder

_ATTENTION_CHILDREN = ("query", "key", "value", "out")


def _shape(node: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.shape(node))


def is_attention_module(node: Any) -> bool:
    """True for flax-style attention modules with query/key/value/out kernels."""

    if not isinstance(node, Mapping):
        return False
    for child in _ATTENTION_CHILDREN:
        value = node.get(child)
        if not isinstance(value, Mapping) or "kernel" not in value:
            return False
    return True


def is_layernorm_module(node: Any) -> bool:
    """True for LayerNorm-style modules (1-D scale, optional bias)."""

    return (
        isinstance(node, Mapping)
        and "scale" in node
        and not isinstance(node["scale"], Mapping)
        and len(_shape(node["scale"])) == 1
        and set(node) <= {"scale", "bias"}
    )


def is_rmsnorm_module(node: Any) -> bool:
    """True for RMSNorm modules (1-D scale, no bias)."""

    return is_layernorm_module(node) and "bias" not in node


def is_dense_module(node: Any) -> bool:
    """True for dense modules (2-D kernel, optional bias)."""

    return (
        isinstance(node, Mapping)
        and "kernel" in node
        and not isinstance(node["kernel"], Mapping)
        and len(_shape(node["kernel"])) == 2
        and set(node) <= {"kernel", "bias"}
    )


class SymmetryRule(ABC):
    """Builds one component of an alignment problem into a :class:`SymmetryGraphBuilder`."""

    kind: ClassVar[str]

    @abstractmethod
    def build(self, builder: SymmetryGraphBuilder):
        """Emit the component's groups, bindings, constraints, and component spec."""


@dataclass
class DenseChainRule(SymmetryRule):
    """A chain of dense layers with permutable hidden junctions.

    Used for plain MLP stacks, transformer FFNs (``input_group`` and
    ``output_group`` tied to the residual stream), and classifier head chains
    (``input_group`` only). Hidden groups are named ``{component_id}/h{j}``; the
    component spec is emitted only when the chain has hidden junctions.
    """

    kind: ClassVar[str] = "dense_stack"

    component_id: str
    layer_paths: tuple[tuple[str, ...], ...]
    input_group: str | None = None
    output_group: str | None = None

    def build(self, builder: SymmetryGraphBuilder) -> tuple[str, ...]:
        if not self.layer_paths:
            raise ValueError(f"Dense component {self.component_id!r} has no layers.")
        layers: list[tuple[tuple[str, ...], int, int, bool]] = []
        for path in self.layer_paths:
            module = _descend(builder.params, path)
            if not isinstance(module, Mapping) or "kernel" not in module:
                raise ValueError(
                    f"Dense component {self.component_id!r} layer {'/'.join(path)} must "
                    "be a module with a 'kernel'."
                )
            kernel_shape = _shape(module["kernel"])
            if len(kernel_shape) != 2:
                raise ValueError(
                    f"Dense component {self.component_id!r} layer {'/'.join(path)} kernel "
                    f"must be 2-D, got shape {kernel_shape}."
                )
            layers.append((path, kernel_shape[0], kernel_shape[1], "bias" in module))
            builder.tensor((*path, "kernel"))
            if "bias" in module:
                builder.tensor((*path, "bias"))
        for (prev_path, _, prev_out, _), (path, din, _, _) in zip(
            layers, layers[1:], strict=False
        ):
            if prev_out != din:
                raise ValueError(
                    f"Dense component {self.component_id!r} layers {'/'.join(prev_path)} "
                    f"and {'/'.join(path)} do not chain: {prev_out} vs {din}."
                )

        hidden_groups: list[str] = []
        for junction, (_, _, dout, _) in enumerate(layers[:-1]):
            hidden_groups.append(
                builder.add_group(f"{self.component_id}/h{junction}", dout)
            )

        for index, (path, _, _, has_bias) in enumerate(layers):
            kernel_path = (*path, "kernel")
            if index == 0:
                if self.input_group is not None:
                    builder.bind(kernel_path, 0, self.input_group, role="in")
            else:
                builder.bind(kernel_path, 0, hidden_groups[index - 1], role="in")
            if index < len(layers) - 1:
                builder.bind(kernel_path, 1, hidden_groups[index], role="out")
                if has_bias:
                    builder.bind((*path, "bias"), 0, hidden_groups[index], role="out")
            elif self.output_group is not None:
                builder.bind(kernel_path, 1, self.output_group, role="out")
                if has_bias:
                    builder.bind((*path, "bias"), 0, self.output_group, role="out")

        if hidden_groups:
            builder.add_component(
                self.component_id,
                self.kind,
                tuple(hidden_groups),
                metadata={"layers": ["/".join(path) for path in self.layer_paths]},
            )
        return tuple(hidden_groups)


@dataclass
class MHAAttentionRule(SymmetryRule):
    """One multi-head attention module: inter-head + per-slot intra groups.

    Emits the wreath-product structure: a head group shared by q/k/v/out, one
    qk intra group per head slot (selector bindings on q and k), one vo intra
    group per slot (v and out), the stream in/out bindings, and the
    ``attention_block`` constraint consumed by the structured LAP update.

    Intra-head binding roles encode the exact diagonal circuit symmetry for
    the scale action: query/value carry ``out`` (divided by the group scale)
    while key and the out kernel carry ``in`` (multiplied), so
    ``apply_scales`` preserves ``q·k`` scores and the value/out contraction
    exactly. Permutations ignore roles, so matching is unaffected.
    """

    kind: ClassVar[str] = "attention"

    component_id: str
    module_path: tuple[str, ...]
    stream_group: str

    def build(self, builder: SymmetryGraphBuilder) -> tuple[str, ...]:
        module = _descend(builder.params, self.module_path)
        if not is_attention_module(module):
            raise ValueError(
                f"Attention component {self.component_id!r}: no query/key/value/out "
                f"modules at {'/'.join(self.module_path)}."
            )
        d_model = int(builder.groups[self.stream_group].size)
        q_shape = _shape(module["query"]["kernel"])
        if len(q_shape) != 3 or q_shape[0] != d_model:
            raise ValueError(
                f"Attention component {self.component_id!r} query kernel must be "
                f"(d_model={d_model}, heads, head_dim), got {q_shape}."
            )
        _, num_heads, head_dim = q_shape
        for name in ("key", "value"):
            shape = _shape(module[name]["kernel"])
            if shape != q_shape:
                raise ValueError(
                    f"Attention component {self.component_id!r} {name} kernel shape "
                    f"{shape} does not match query kernel {q_shape}."
                )
        out_shape = _shape(module["out"]["kernel"])
        if out_shape != (num_heads, head_dim, d_model):
            raise ValueError(
                f"Attention component {self.component_id!r} out kernel must be "
                f"{(num_heads, head_dim, d_model)}, got {out_shape}."
            )
        for name in ("query", "key", "value"):
            if "bias" in module[name] and _shape(module[name]["bias"]) != (
                num_heads,
                head_dim,
            ):
                raise ValueError(
                    f"Attention component {self.component_id!r} {name} bias must be "
                    f"(heads, head_dim), got {_shape(module[name]['bias'])}."
                )
        if "bias" in module["out"] and _shape(module["out"]["bias"]) != (d_model,):
            raise ValueError(
                f"Attention component {self.component_id!r} out bias must be "
                f"(d_model,), got {_shape(module['out']['bias'])}."
            )

        heads_group = builder.add_group(f"{self.component_id}/heads", num_heads)
        qk_groups: list[str] = []
        vo_groups: list[str] = []
        for slot in range(num_heads):
            # The diagonal circuit symmetry includes sign flips: flipping one
            # intra-head dimension in both circuit factors preserves QK / OV.
            qk_groups.append(
                builder.add_group(
                    f"{self.component_id}/qk{slot}",
                    head_dim,
                    transform_family="signed_permutation",
                )
            )
            vo_groups.append(
                builder.add_group(
                    f"{self.component_id}/vo{slot}",
                    head_dim,
                    transform_family="signed_permutation",
                )
            )

        role_tensor_ids: dict[str, str] = {}
        for name, intra_groups, intra_role in (
            ("query", qk_groups, "out"),
            ("key", qk_groups, "in"),
            ("value", vo_groups, "out"),
        ):
            kernel_path = (*self.module_path, name, "kernel")
            role_tensor_ids[name] = builder.bind(
                kernel_path, 0, self.stream_group, role="in"
            )
            builder.bind(kernel_path, 1, heads_group)
            for slot, intra_group in enumerate(intra_groups):
                builder.bind(
                    kernel_path, 2, intra_group, role=intra_role, selector=((1, slot),)
                )
            if "bias" in module[name]:
                bias_path = (*self.module_path, name, "bias")
                builder.bind(bias_path, 0, heads_group)
                for slot, intra_group in enumerate(intra_groups):
                    builder.bind(
                        bias_path,
                        1,
                        intra_group,
                        role=intra_role,
                        selector=((0, slot),),
                    )
        out_kernel_path = (*self.module_path, "out", "kernel")
        role_tensor_ids["out"] = builder.bind(out_kernel_path, 0, heads_group)
        for slot, vo_group in enumerate(vo_groups):
            builder.bind(out_kernel_path, 1, vo_group, role="in", selector=((0, slot),))
        builder.bind(out_kernel_path, 2, self.stream_group, role="out")
        if "bias" in module["out"]:
            builder.bind(
                (*self.module_path, "out", "bias"), 0, self.stream_group, role="out"
            )

        all_groups = (heads_group, *qk_groups, *vo_groups)
        builder.add_constraint(
            GraphConstraint(
                kind="attention_block",
                groups=all_groups,
                tensors=tuple(role_tensor_ids[name] for name in _ATTENTION_CHILDREN),
                metadata={
                    "head_group": heads_group,
                    "qk_groups": tuple(qk_groups),
                    "vo_groups": tuple(vo_groups),
                    "tensors": dict(role_tensor_ids),
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                },
            )
        )
        builder.add_component(
            self.component_id,
            self.kind,
            all_groups,
            metadata={"num_heads": num_heads, "head_dim": head_dim},
        )
        return all_groups


@dataclass
class GQARoPEAttentionRule(SymmetryRule):
    """One grouped-query attention module with RoPE on the qk circuit.

    Emits the GQA quotient of the attention head symmetry: a kv-group
    permutation of size ``G`` shared by query/key/value/out, one query-head
    group of size ``H/G`` per kv slot (selector bindings on the query and out
    kernels), one signed vo intra group per kv slot, and one qk
    ``rotation_pairs`` group per kv slot: rotary embeddings restrict the qk
    circuit symmetry to per frequency-pair scaled rotations (see the
    modern-transformer subsection of docs/theory.md), so the qk groups carry
    no permutations — solvers update them by the closed-form per-pair
    rotation projection, and the normalization plan balances the pair scales
    through the same bindings.

    Expected kernel layout (flat ``(d, H, dk)`` trees reshape losslessly with
    query head ``i`` in kv group ``i // (H/G)``): query ``(d, G, H/G, dk)``,
    key/value ``(d, G, dk)``, out ``(G, H/G, dk, d)``. Biases are rejected —
    the exact RoPE symmetry model assumes bias-free projections (LLaMA-style).
    """

    kind: ClassVar[str] = "gqa_attention"

    component_id: str
    module_path: tuple[str, ...]
    stream_group: str

    def build(self, builder: SymmetryGraphBuilder) -> tuple[str, ...]:
        module = _descend(builder.params, self.module_path)
        if not is_attention_module(module):
            raise ValueError(
                f"GQA attention component {self.component_id!r}: no query/key/value/out "
                f"modules at {'/'.join(self.module_path)}."
            )
        for name in _ATTENTION_CHILDREN:
            if "bias" in module[name]:
                raise ValueError(
                    f"GQA attention component {self.component_id!r} has a bias on "
                    f"{name!r}; the RoPE/GQA symmetry model requires bias-free "
                    "projections."
                )
        d_model = int(builder.groups[self.stream_group].size)
        q_shape = _shape(module["query"]["kernel"])
        if len(q_shape) != 4 or q_shape[0] != d_model:
            raise ValueError(
                f"GQA attention component {self.component_id!r} query kernel must be "
                f"(d_model={d_model}, kv_groups, heads_per_group, head_dim), "
                f"got {q_shape}."
            )
        _, num_kv_groups, heads_per_group, head_dim = q_shape
        if head_dim % 2:
            raise ValueError(
                f"GQA attention component {self.component_id!r} head_dim {head_dim} must "
                "be even for rotary embeddings."
            )
        kv_shape = (d_model, num_kv_groups, head_dim)
        for name in ("key", "value"):
            shape = _shape(module[name]["kernel"])
            if shape != kv_shape:
                raise ValueError(
                    f"GQA attention component {self.component_id!r} {name} kernel must "
                    f"be {kv_shape}, got {shape}."
                )
        out_shape = _shape(module["out"]["kernel"])
        if out_shape != (num_kv_groups, heads_per_group, head_dim, d_model):
            raise ValueError(
                f"GQA attention component {self.component_id!r} out kernel must be "
                f"{(num_kv_groups, heads_per_group, head_dim, d_model)}, got "
                f"{out_shape}."
            )

        kv_group = builder.add_group(f"{self.component_id}/kv", num_kv_groups)
        qh_groups: list[str] = []
        qk_groups: list[str] = []
        vo_groups: list[str] = []
        for slot in range(num_kv_groups):
            qh_groups.append(
                builder.add_group(f"{self.component_id}/qh{slot}", heads_per_group)
            )
            qk_groups.append(
                builder.add_group(
                    f"{self.component_id}/qk{slot}",
                    head_dim,
                    transform_family="rotation_pairs",
                )
            )
            vo_groups.append(
                builder.add_group(
                    f"{self.component_id}/vo{slot}",
                    head_dim,
                    transform_family="signed_permutation",
                )
            )

        query_path = (*self.module_path, "query", "kernel")
        key_path = (*self.module_path, "key", "kernel")
        value_path = (*self.module_path, "value", "kernel")
        out_path = (*self.module_path, "out", "kernel")

        role_tensor_ids: dict[str, str] = {}
        role_tensor_ids["query"] = builder.bind(
            query_path, 0, self.stream_group, role="in"
        )
        builder.bind(query_path, 1, kv_group)
        for slot, qh in enumerate(qh_groups):
            builder.bind(query_path, 2, qh, selector=((1, slot),))
        for slot, qk in enumerate(qk_groups):
            builder.bind(query_path, 3, qk, role="out", selector=((1, slot),))
        role_tensor_ids["key"] = builder.bind(key_path, 0, self.stream_group, role="in")
        builder.bind(key_path, 1, kv_group)
        for slot, qk in enumerate(qk_groups):
            builder.bind(key_path, 2, qk, role="in", selector=((1, slot),))
        role_tensor_ids["value"] = builder.bind(
            value_path, 0, self.stream_group, role="in"
        )
        builder.bind(value_path, 1, kv_group)
        for slot, vo in enumerate(vo_groups):
            builder.bind(value_path, 2, vo, role="out", selector=((1, slot),))
        role_tensor_ids["out"] = builder.bind(out_path, 0, kv_group)
        for slot, qh in enumerate(qh_groups):
            builder.bind(out_path, 1, qh, selector=((0, slot),))
        for slot, vo in enumerate(vo_groups):
            builder.bind(out_path, 2, vo, role="in", selector=((0, slot),))
        builder.bind(out_path, 3, self.stream_group, role="out")

        all_groups = (kv_group, *qh_groups, *qk_groups, *vo_groups)
        builder.add_constraint(
            GraphConstraint(
                kind="gqa_attention_block",
                groups=all_groups,
                tensors=tuple(role_tensor_ids[name] for name in _ATTENTION_CHILDREN),
                metadata={
                    "kv_group": kv_group,
                    "query_head_groups": tuple(qh_groups),
                    "qk_groups": tuple(qk_groups),
                    "vo_groups": tuple(vo_groups),
                    "tensors": dict(role_tensor_ids),
                    "num_kv_groups": num_kv_groups,
                    "heads_per_group": heads_per_group,
                    "head_dim": head_dim,
                    "rope_pairing": "half",
                },
            )
        )
        builder.add_component(
            self.component_id,
            self.kind,
            all_groups,
            metadata={
                "num_kv_groups": num_kv_groups,
                "heads_per_group": heads_per_group,
                "head_dim": head_dim,
            },
        )
        return all_groups


@dataclass
class ResidualStreamRule(SymmetryRule):
    """The residual stream: one global group over ``d_model``.

    Binds LayerNorm/RMSNorm parameters and embedding-style tensors (trailing
    ``d_model`` axis) to the stream group. Attention and FFN components tie their
    stream-facing axes to the same group via ``input_group``/``output_group``.

    ``transform_family`` declares the stream's symmetry class: ``"permutation"``
    for LayerNorm stacks; ``"signed_permutation"`` or ``"orthogonal"`` for
    RMSNorm stacks (whose streams admit the full orthogonal group). Beyond
    plain permutations the norms must be RMSNorm (scale only), and the scale
    vectors bind ``permute_only``: they permute with the stream but are
    exempt from signs and orthogonal maps, which their consumers absorb.
    """

    kind: ClassVar[str] = "residual_stream"

    d_model: int
    layernorm_paths: tuple[tuple[str, ...], ...] = ()
    feature_leaves: tuple[tuple[str, ...], ...] = ()
    group_id: str = "stream"
    transform_family: str = "permutation"

    def build(self, builder: SymmetryGraphBuilder) -> str:
        builder.add_group(
            self.group_id, self.d_model, transform_family=self.transform_family
        )
        norm_scope = (
            "linear" if self.transform_family == "permutation" else "permute_only"
        )
        for path in self.layernorm_paths:
            module = _descend(builder.params, path)
            if not is_layernorm_module(module):
                raise ValueError(
                    f"Stream component: {'/'.join(path)} is not a LayerNorm module."
                )
            if self.transform_family != "permutation" and not is_rmsnorm_module(module):
                raise ValueError(
                    f"Stream component: {'/'.join(path)} carries a bias; the "
                    f"{self.transform_family!r} stream symmetry holds only for "
                    "RMSNorm (scale-only) stacks."
                )
            scale_shape = _shape(module["scale"])
            if scale_shape != (self.d_model,):
                raise ValueError(
                    f"LayerNorm {'/'.join(path)} scale has shape {scale_shape}, "
                    f"expected ({self.d_model},)."
                )
            builder.bind(
                (*path, "scale"),
                0,
                self.group_id,
                role="out",
                transform_scope=norm_scope,
            )
            if "bias" in module:
                builder.bind((*path, "bias"), 0, self.group_id, role="out")
        for path in self.feature_leaves:
            value = _descend(builder.params, path)
            shape = _shape(value)
            if not shape or shape[-1] != self.d_model:
                raise ValueError(
                    f"Stream feature tensor {'/'.join(path)} has shape {shape}; "
                    f"expected a trailing d_model axis of {self.d_model}."
                )
            builder.bind(path, -1, self.group_id, role="out")
        builder.add_component(self.component_id, self.kind, (self.group_id,))
        return self.group_id

    @property
    def component_id(self) -> str:
        return self.group_id


__all__ = [
    "MHAAttentionRule",
    "SymmetryRule",
    "DenseChainRule",
    "GQARoPEAttentionRule",
    "ResidualStreamRule",
    "is_attention_module",
    "is_dense_module",
    "is_layernorm_module",
    "is_rmsnorm_module",
]
