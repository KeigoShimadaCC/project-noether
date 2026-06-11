"""SymPy kernel adapter: COMPONENT_EVAL checks on explicit metrics.

Supported task payloads (capability COMPONENT_EVAL):
  {"check": "zero",            "expr": <NPR Expr dict>, "metric": <spec>}
  {"check": "symmetric",       "expr": <rank-2 Expr dict>, "metric": <spec>}
  {"check": "divergence-zero", "expr": <rank-2 down-down Expr dict>, "metric": <spec>}
  {"check": "equal",           "lhs": <Expr dict>, "rhs": <Expr dict>, "metric": <spec>}

Metric specs:
  {"kind": "random-diagonal", "seed": <int>, "dim": <int>}
  {"kind": "two-sphere"}
"""

import time
from typing import Any

import sympy as sp
from pydantic import TypeAdapter

from noether.kernels.base import (
    Capability,
    ComputedResult,
    KernelRawOutput,
    KernelScript,
    KernelTask,
)
from noether.kernels.sympy_kernel.evaluator import all_zero, evaluate
from noether.kernels.sympy_kernel.geometry import (
    ComponentGeometry,
    random_diagonal_metric,
    two_sphere,
)
from noether.npr.ast import Expr

_EXPR = TypeAdapter(Expr)


def _geometry_for(spec: dict[str, Any]) -> ComponentGeometry:
    kind = spec.get("kind")
    if kind == "random-diagonal":
        return random_diagonal_metric(seed=int(spec["seed"]), dim=int(spec.get("dim", 4)))
    if kind == "two-sphere":
        return two_sphere()
    raise ValueError(f"unknown metric spec kind {kind!r}")


class SympyKernelAdapter:
    name = "sympy"

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return sp.__version__

    def capabilities(self) -> set[Capability]:
        return {Capability.COMPONENT_EVAL}

    def run(self, task: KernelTask, npr: Any = None) -> ComputedResult:
        if task.capability is not Capability.COMPONENT_EVAL:
            raise ValueError(f"sympy kernel does not provide {task.capability}")
        payload = task.payload
        check = payload["check"]
        geom = _geometry_for(payload["metric"])
        start = time.monotonic()

        if check == "zero":
            expr = _EXPR.validate_python(payload["expr"])
            value, _free = evaluate(expr, geom)
            passed, detail = all_zero(value)
        elif check == "symmetric":
            expr = _EXPR.validate_python(payload["expr"])
            value, free = evaluate(expr, geom)
            if len(free) != 2:
                raise ValueError("symmetric check needs a rank-2 expression")
            residue = value - sp.permutedims(value, (1, 0))
            passed, detail = all_zero(residue)
        elif check == "divergence-zero":
            expr = _EXPR.validate_python(payload["expr"])
            value, free = evaluate(expr, geom)
            if len(free) != 2 or any(ix.variance != "down" for ix in free):
                raise ValueError("divergence check needs a rank-2 down-down expression")
            grad = geom.covariant_derivative(value, ["down", "down"])  # [a, mu, nu]
            grad_up = geom.raise_first_index(grad, 0)
            div = sp.tensorcontraction(grad_up, (0, 1))
            passed, detail = all_zero(div)
        elif check == "equal":
            lhs = _EXPR.validate_python(payload["lhs"])
            rhs = _EXPR.validate_python(payload["rhs"])
            lv, lf = evaluate(lhs, geom)
            rv, rf = evaluate(rhs, geom)
            if [ix.model_dump() for ix in lf] != [ix.model_dump() for ix in rf]:
                passed, detail = False, f"free index mismatch: {lf} vs {rf}"
            else:
                passed, detail = all_zero(lv - rv if lf else sp.simplify(lv - rv))
        else:
            raise ValueError(f"unknown check {check!r}")

        duration = time.monotonic() - start
        script = KernelScript(
            kernel_name=self.name,
            language="python-sympy",
            source=_reproduction_script(payload),
        )
        raw = KernelRawOutput(stdout=detail, returncode=0, duration_s=round(duration, 3))
        return ComputedResult(
            kernel_name=self.name,
            kernel_version=self.version(),
            script=script,
            raw=raw,
            value={"passed": passed, "detail": detail, "check": check},
            notes=[f"metric spec: {payload['metric']}"],
        )


def _reproduction_script(payload: dict[str, Any]) -> str:
    """A standalone script that re-runs this exact check."""
    return (
        "# Reproduction script (noether sympy kernel)\n"
        "from noether.kernels.base import Capability, KernelTask\n"
        "from noether.kernels.sympy_kernel import SympyKernelAdapter\n"
        f"task = KernelTask(capability=Capability.COMPONENT_EVAL,\n"
        f"                  description={payload.get('check', '')!r},\n"
        f"                  payload={payload!r})\n"
        "result = SympyKernelAdapter().run(task)\n"
        "print(result.value)\n"
        "assert result.value['passed'], result.value['detail']\n"
    )
