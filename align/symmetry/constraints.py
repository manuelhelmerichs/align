"""Typed architecture constraints carried by a :class:`SymmetryGraph`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MHACircuitConstraint:
    """Coupled head and intra-head groups for one MHA circuit."""

    head_group: str
    qk_groups: tuple[str, ...]
    vo_groups: tuple[str, ...]
    query: str
    key: str
    value: str
    out: str

    @property
    def groups(self) -> tuple[str, ...]:
        return (self.head_group, *self.qk_groups, *self.vo_groups)

    @property
    def tensors(self) -> tuple[str, ...]:
        return (self.query, self.key, self.value, self.out)


@dataclass(frozen=True)
class GQARoPECircuitConstraint:
    """Coupled quotient groups and tensors for one GQA + RoPE circuit."""

    kv_group: str
    query_head_groups: tuple[str, ...]
    qk_groups: tuple[str, ...]
    vo_groups: tuple[str, ...]
    query: str
    key: str
    value: str
    out: str
    num_kv_groups: int
    heads_per_group: int
    head_dim: int

    @property
    def groups(self) -> tuple[str, ...]:
        return (
            self.kv_group,
            *self.query_head_groups,
            *self.qk_groups,
            *self.vo_groups,
        )

    @property
    def tensors(self) -> tuple[str, ...]:
        return (self.query, self.key, self.value, self.out)


@dataclass(frozen=True)
class RMSNormScaleConstraint:
    """One RMSNorm scale tensor and the consumer axes that absorb its scale."""

    scale: str
    consumers: tuple[tuple[str, int], ...]

    @property
    def groups(self) -> tuple[str, ...]:
        return ()

    @property
    def tensors(self) -> tuple[str, ...]:
        return (self.scale, *(tensor_id for tensor_id, _ in self.consumers))


@dataclass(frozen=True)
class ResidualChannelTie:
    """Residual branches that have been canonicalized to one channel group."""

    groups: tuple[str, ...]
    tensors: tuple[str, ...]
    members: tuple[str, ...]
    source: str


type ConstraintRecord = (
    MHACircuitConstraint
    | GQARoPECircuitConstraint
    | RMSNormScaleConstraint
    | ResidualChannelTie
)


__all__ = [
    "ConstraintRecord",
    "GQARoPECircuitConstraint",
    "MHACircuitConstraint",
    "RMSNormScaleConstraint",
    "ResidualChannelTie",
]
