"""Individual verification checks.

Each check is kernel-computed wherever it asserts mathematics; `computed_by`
names the kernel so the bright line between "model reasoned" and "kernel
computed" survives into the checks.json artifact.
"""

from typing import Any

from pydantic import BaseModel, Field

from noether.kernels.base import Capability, ComputedResult, KernelAdapter, KernelTask
from noether.npr.ast import Expr, Index
from noether.npr.validate import ValidationError, validate_expression

DEFAULT_METRIC_SPECS: list[dict[str, Any]] = [
    {"kind": "random-diagonal", "seed": 7, "dim": 4},
    {"kind": "random-diagonal", "seed": 23, "dim": 4},
]


class CheckResult(BaseModel):
    name: str
    rung: str  # "V0".."V4"
    passed: bool
    detail: str
    computed_by: str  # kernel name, or "structural"
    artifacts: list[ComputedResult] = Field(default_factory=list)


class WellFormedCheck:
    """V0: index balance and expected free indices, structural."""

    name = "well-formed"
    rung = "V0"

    def __init__(self, expected_free: list[Index] | None = None):
        self.expected_free = expected_free

    def run(self, expr: Expr, adapters: dict[str, KernelAdapter]) -> CheckResult:
        try:
            validate_expression(expr, self.expected_free)
            return CheckResult(
                name=self.name,
                rung=self.rung,
                passed=True,
                detail="index balance and free indices OK",
                computed_by="structural",
            )
        except ValidationError as exc:
            return CheckResult(
                name=self.name,
                rung=self.rung,
                passed=False,
                detail=str(exc),
                computed_by="structural",
            )


class _ComponentCheck:
    """Shared machinery: run a component check on every default background."""

    check_kind: str
    name: str
    rung: str

    def __init__(self, metric_specs: list[dict[str, Any]] | None = None):
        self.metric_specs = metric_specs or DEFAULT_METRIC_SPECS

    def _payload(self, expr: Expr, spec: dict[str, Any]) -> dict[str, Any]:
        return {"check": self.check_kind, "expr": expr.model_dump(), "metric": spec}

    def run(self, expr: Expr, adapters: dict[str, KernelAdapter]) -> CheckResult:
        kernel = _component_kernel(adapters)
        artifacts: list[ComputedResult] = []
        for spec in self.metric_specs:
            task = KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description=f"{self.name} on {spec}",
                payload=self._payload(expr, spec),
            )
            result = kernel.run(task)
            artifacts.append(result)
            if not result.value["passed"]:
                return CheckResult(
                    name=self.name,
                    rung=self.rung,
                    passed=False,
                    detail=f"failed on {spec}: {result.value['detail']}",
                    computed_by=kernel.name,
                    artifacts=artifacts,
                )
        return CheckResult(
            name=self.name,
            rung=self.rung,
            passed=True,
            detail=f"holds on {len(self.metric_specs)} background(s)",
            computed_by=kernel.name,
            artifacts=artifacts,
        )


class SymmetricCheck(_ComponentCheck):
    """V1: rank-2 result symmetric, spot-checked on explicit backgrounds."""

    check_kind = "symmetric"
    name = "symmetric-rank2"
    rung = "V1"


class DivergenceFreeCheck(_ComponentCheck):
    """V2: covariant divergence vanishes (e.g. contracted Bianchi)."""

    check_kind = "divergence-zero"
    name = "divergence-free"
    rung = "V2"


class EqualOnBackgroundCheck(_ComponentCheck):
    """V3-style comparison: two expressions agree on explicit backgrounds."""

    check_kind = "equal"
    name = "equal-on-background"
    rung = "V3"

    def __init__(self, rhs: Expr, metric_specs: list[dict[str, Any]] | None = None):
        super().__init__(metric_specs)
        self.rhs = rhs

    def _payload(self, expr: Expr, spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "check": "equal",
            "lhs": expr.model_dump(),
            "rhs": self.rhs.model_dump(),
            "metric": spec,
        }


def _component_kernel(adapters: dict[str, KernelAdapter]) -> KernelAdapter:
    for adapter in adapters.values():
        if Capability.COMPONENT_EVAL in adapter.capabilities() and adapter.available():
            return adapter
    raise RuntimeError("no available kernel provides COMPONENT_EVAL")
