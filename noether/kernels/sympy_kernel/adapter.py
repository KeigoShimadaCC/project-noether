"""SymPy kernel adapter: COMPONENT_EVAL checks on explicit metrics.

Supported task payloads (capability COMPONENT_EVAL):
  {"check": "zero",            "expr": <NPR Expr dict>, "metric": <spec>}
  {"check": "symmetric",       "expr": <rank-2 Expr dict>, "metric": <spec>}
  {"check": "divergence-zero", "expr": <rank-2 down-down Expr dict>, "metric": <spec>}
  {"check": "equal",           "lhs": <Expr dict>, "rhs": <Expr dict>, "metric": <spec>}
  {"check": "palatini-projective-inert", "metric": <spec>, "seed": <int>}
  {"check": "adm-gr-1p2"}  (no metric spec: builds its own foliated 1+2
                            background and runs every ADM split/constraint
                            check in noether.kernels.sympy_kernel.adm)

All checks accept an optional "fields" spec binding extra named tensors:
  {"phi": {"kind": "random-scalar", "seed": 7},
   "A":   {"kind": "random-covector", "seed": 3},
   "F":   {"kind": "random-antisymmetric", "seed": 5}}

Metric specs:
  {"kind": "random-diagonal", "seed": <int>, "dim": <int>}
  {"kind": "two-sphere"}

The palatini check builds Gamma = LC(g) + delta^lam_nu A_mu with a seeded
random covector A, computes Ricci(Gamma) from the general affine formula, and
asserts (a) its symmetric part equals Ricci(g) and (b) the Palatini metric
equation R_{(mu nu)} - 1/2 g_{mu nu} R~ equals the Einstein tensor of g.
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
from noether.kernels.sympy_kernel.adm import adm_sample_1p2
from noether.kernels.sympy_kernel.evaluator import all_zero, evaluate
from noether.kernels.sympy_kernel.geometry import (
    ComponentGeometry,
    projective_connection,
    random_antisymmetric,
    random_covector,
    random_diagonal_metric,
    random_scalar_field,
    ricci_of_connection,
    sparse_diagonal_metric,
    two_sphere,
    warped_product_4d,
)
from noether.npr.ast import Expr

_EXPR = TypeAdapter(Expr)


def _geometry_for(spec: dict[str, Any]) -> ComponentGeometry:
    kind = spec.get("kind")
    if kind == "random-diagonal":
        return random_diagonal_metric(seed=int(spec["seed"]), dim=int(spec.get("dim", 4)))
    if kind == "sparse-diagonal":
        return sparse_diagonal_metric(
            seed=int(spec["seed"]), dim=int(spec.get("dim", 4)), curved=int(spec.get("curved", 3))
        )
    if kind == "two-sphere":
        return two_sphere()
    if kind == "warped-product-4d":
        return warped_product_4d()
    raise ValueError(f"unknown metric spec kind {kind!r}")


def _fields_for(spec: dict[str, Any], geom: ComponentGeometry) -> dict[str, tuple[Any, list[str]]]:
    out: dict[str, tuple[Any, list[str]]] = {}
    for name, fs in (spec or {}).items():
        kind, seed = fs.get("kind"), int(fs.get("seed", 0))
        if kind == "random-scalar":
            out[name] = (random_scalar_field(seed, geom.coords), [])
        elif kind == "random-covector":
            out[name] = (random_covector(seed, geom.coords), ["down"])
        elif kind == "random-antisymmetric":
            out[name] = (random_antisymmetric(seed, geom.coords), ["down", "down"])
        else:
            raise ValueError(f"unknown field spec kind {kind!r}")
    return out


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
        if check == "adm-gr-1p2":
            return self._run_adm(payload)
        geom = _geometry_for(payload["metric"])
        fields = _fields_for(payload.get("fields", {}), geom)
        start = time.monotonic()

        if check == "zero":
            expr = _EXPR.validate_python(payload["expr"])
            value, _free = evaluate(expr, geom, fields=fields)
            passed, detail = all_zero(value)
        elif check == "symmetric":
            expr = _EXPR.validate_python(payload["expr"])
            value, free = evaluate(expr, geom, fields=fields)
            if len(free) != 2:
                raise ValueError("symmetric check needs a rank-2 expression")
            residue = value - sp.permutedims(value, (1, 0))
            passed, detail = all_zero(residue)
        elif check == "divergence-zero":
            expr = _EXPR.validate_python(payload["expr"])
            value, free = evaluate(expr, geom, fields=fields)
            if len(free) != 2 or any(ix.variance != "down" for ix in free):
                raise ValueError("divergence check needs a rank-2 down-down expression")
            grad = geom.covariant_derivative(value, ["down", "down"])  # [a, mu, nu]
            grad_up = geom.raise_first_index(grad, 0)
            div = sp.tensorcontraction(grad_up, (0, 1))
            passed, detail = all_zero(div)
        elif check == "equal":
            lhs = _EXPR.validate_python(payload["lhs"])
            rhs = _EXPR.validate_python(payload["rhs"])
            lv, lf = evaluate(lhs, geom, fields=fields)
            rv, rf = evaluate(rhs, geom, fields=fields)
            if [ix.model_dump() for ix in lf] != [ix.model_dump() for ix in rf]:
                passed, detail = False, f"free index mismatch: {lf} vs {rf}"
            else:
                passed, detail = all_zero(lv - rv if lf else sp.simplify(lv - rv))
        elif check == "palatini-projective-inert":
            passed, detail = _palatini_projective_inert(geom, int(payload.get("seed", 0)))
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

    def _run_adm(self, payload: dict[str, Any]) -> ComputedResult:
        start = time.monotonic()
        results = adm_sample_1p2().run_all()
        passed = all(ok for ok, _ in results.values())
        detail = "; ".join(
            f"{name}: {'PASS' if ok else 'FAIL'} ({d})" for name, (ok, d) in results.items()
        )
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
            value={"passed": passed, "detail": detail, "check": "adm-gr-1p2"},
            notes=["adm background: deterministic nondegenerate 1+2 sample (adm_sample_1p2)"],
        )


def _palatini_projective_inert(geom: ComponentGeometry, seed: int) -> tuple[bool, str]:
    n = geom.dim
    cov = random_covector(seed, geom.coords)
    gamma = projective_connection(geom, cov)
    ric = ricci_of_connection(geom.coords, gamma)
    sym_part = (ric + sp.permutedims(ric, (1, 0))) / 2
    ok_sym, det_sym = all_zero(sym_part - geom.ricci)
    if not ok_sym:
        return False, f"R_(mu nu)(LC + projective) != Ricci(g): {det_sym}"
    rtilde = sum(geom.g_inv[a, b] * ric[a, b] for a in range(n) for b in range(n))
    eom = sym_part - sp.Rational(1, 2) * rtilde * sp.ImmutableDenseNDimArray(geom.g)
    ok_eom, det_eom = all_zero(eom - geom.einstein)
    if not ok_eom:
        return False, f"Palatini metric equation != Einstein(g): {det_eom}"
    return True, "symmetric Ricci part and metric equation both reduce to the Levi-Civita ones"


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
