"""Lightweight dataclasses shared between config and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SCHEDULE_FIELDS = frozenset(
    {
        "solver",
        "max_sweeps",
        "max_steps",
        "tol",
        "tau",
        "lr",
        "n_sinkhorn_iters",
        "init_scale",
        "record_loss_history",
        "harden",
        "groups",
        "components",
    }
)


def _require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean, got {type(value).__name__}.")
    return value


def _validate_schedule_fields(payload: dict[str, Any]) -> None:
    unknown = sorted(set(payload) - _SCHEDULE_FIELDS)
    if unknown:
        raise ValueError("Unknown rebasin schedule option(s): " + ", ".join(unknown))


@dataclass(frozen=True)
class SolverStep:
    """JAX-free config representation of one graph solver schedule step."""

    solver: str
    max_sweeps: int = 25
    max_steps: int = 200
    tol: float = 0.0
    tau: float = 0.1
    lr: float = 1e-2
    n_sinkhorn_iters: int = 50
    init_scale: float = 1e-2
    record_loss_history: bool = False
    harden: bool = True
    groups: tuple[str, ...] | None = None
    components: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.groups is not None and self.components is not None:
            raise ValueError(
                "A rebasin schedule step may set 'groups' or 'components', not both."
            )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> SolverStep:
        _validate_schedule_fields(payload)
        if "solver" not in payload:
            raise ValueError("Each rebasin schedule entry must define 'solver'.")
        groups = payload.get("groups")
        components = payload.get("components")
        return cls(
            solver=str(payload["solver"]).lower(),
            max_sweeps=int(payload.get("max_sweeps", 25)),
            max_steps=int(payload.get("max_steps", 200)),
            tol=float(payload.get("tol", 0.0)),
            tau=float(payload.get("tau", 0.1)),
            lr=float(payload.get("lr", 1e-2)),
            n_sinkhorn_iters=int(payload.get("n_sinkhorn_iters", 50)),
            init_scale=float(payload.get("init_scale", 1e-2)),
            record_loss_history=_require_bool(
                "rebasin.schedule.record_loss_history",
                payload.get("record_loss_history", False),
            ),
            harden=_require_bool(
                "rebasin.schedule.harden", payload.get("harden", True)
            ),
            groups=tuple(str(group) for group in groups)
            if groups is not None
            else None,
            components=tuple(str(component) for component in components)
            if components is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "solver": self.solver,
            "tol": self.tol,
        }
        if self.solver in ("lap", "procrustes"):
            payload["max_sweeps"] = self.max_sweeps
        elif self.solver == "sinkhorn":
            payload.update(
                {
                    "max_steps": self.max_steps,
                    "tau": self.tau,
                    "lr": self.lr,
                    "n_sinkhorn_iters": self.n_sinkhorn_iters,
                    "init_scale": self.init_scale,
                    "record_loss_history": self.record_loss_history,
                    "harden": self.harden,
                }
            )
        if self.groups is not None:
            payload["groups"] = list(self.groups)
        if self.components is not None:
            payload["components"] = list(self.components)
        return payload


__all__ = ["SolverStep"]
