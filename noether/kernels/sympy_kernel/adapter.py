"""SymPy kernel adapter: COMPONENT_EVAL checks on explicit metrics.

Supported task payloads (capability COMPONENT_EVAL):
  {"check": "zero",            "expr": <NPR Expr dict>, "metric": <spec>}
  {"check": "symmetric",       "expr": <rank-2 Expr dict>, "metric": <spec>}
  {"check": "divergence-zero", "expr": <rank-2 down-down Expr dict>, "metric": <spec>}
  {"check": "equal",           "lhs": <Expr dict>, "rhs": <Expr dict>, "metric": <spec>}
  {"check": "palatini-projective-inert", "metric": <spec>, "seed": <int>}
  {"check": "palatini-ricci-shift-is-dA", "metric": <spec>,
   "connection_seed": <int>, "covector_seed": <int>}
  {"check": "palatini-projective-inert-general", "metric": <spec>,
   "connection_seed": <int>, "covector_seed": <int>}
  {"check": "adm-gr-1p2"}  (no metric spec: builds its own foliated 1+2
                            background and runs every ADM split/constraint
                            check in noether.kernels.sympy_kernel.adm)
  {"check": "spectrum-scalar-tensor-minkowski"}  (no metric spec: runs every
                            linearization/diagonalization check in
                            noether.kernels.sympy_kernel.linearized)

All checks accept an optional "fields" spec binding extra named tensors:
  {"phi": {"kind": "random-scalar", "seed": 7},
   "A":   {"kind": "random-covector", "seed": 3},
   "F":   {"kind": "random-antisymmetric", "seed": 5}}

Metric specs:
  {"kind": "random-diagonal", "seed": <int>, "dim": <int>}
  {"kind": "two-sphere"}

The palatini checks build Gamma = LC(g) + delta^lam_nu A_mu with a seeded
random covector A, compute Ricci(Gamma) from the general affine formula, and
assert (a) its symmetric part equals Ricci(g) and (b) the Palatini metric
equation R_{(mu nu)} - 1/2 g_{mu nu} R~ equals the Einstein tensor of g.

The palatini-ricci-shift-is-dA check verifies R(Gamma + projective) - R(Gamma)
= dA (exterior derivative of A) componentwise on random general-connection
backgrounds.

The palatini-projective-inert-general check verifies the Palatini metric
equation is unchanged by the projective shift on random general-connection
backgrounds (not just Levi-Civita + projective).
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
    Array,
    ComponentGeometry,
    _clean,
    exterior_derivative_of_1form,
    projective_connection,
    random_affine_connection,
    random_antisymmetric,
    random_covector,
    random_diagonal_metric,
    random_scalar_field,
    ricci_of_connection,
    sparse_diagonal_metric,
    two_sphere,
    warped_product_4d,
)
from noether.kernels.sympy_kernel.linearized import spectrum_checks
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
            return self._run_suite(
                payload,
                lambda: adm_sample_1p2().run_all(),
                "adm background: deterministic nondegenerate 1+2 sample (adm_sample_1p2)",
            )
        if check == "spectrum-scalar-tensor-minkowski":
            return self._run_suite(
                payload,
                spectrum_checks,
                "linearized around Minkowski; anchor checks recompute the full "
                "eval-3 equations with exact ComponentGeometry",
            )
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
        elif check == "palatini-ricci-shift-is-dA":
            conn_seed = int(payload.get("connection_seed", 7))
            cov_seed = int(payload.get("covector_seed", 3))
            passed, detail = _palatini_ricci_shift_is_dA(
                geom, conn_seed, cov_seed
            )
        elif check == "palatini-projective-inert-general":
            conn_seed = int(payload.get("connection_seed", 7))
            cov_seed = int(payload.get("covector_seed", 3))
            passed, detail = _palatini_projective_inert_general(
                geom, conn_seed, cov_seed
            )
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

    def _run_suite(
        self,
        payload: dict[str, Any],
        suite: Any,
        note: str,
    ) -> ComputedResult:
        """Run a named suite of (ok, detail) checks as one kernel task."""
        start = time.monotonic()
        results: dict[str, tuple[bool, str]] = suite()
        passed = all(ok for ok, _ in results.values())
        detail = "; ".join(
            f"{name}: {'PASS' if ok else 'FAIL'} ({d})" for name, (ok, d) in results.items()
        )
        checks = {name: ("True" if ok else "False") for name, (ok, _) in results.items()}
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
            value={"passed": passed, "detail": detail, "check": payload["check"], "checks": checks},
            notes=[note],
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


    return True, "symmetric Ricci part and metric equation both reduce to the Levi-Civita ones"


def _palatini_ricci_shift_is_dA(
    geom: ComponentGeometry, conn_seed: int, cov_seed: int
) -> tuple[bool, str]:
    """Verify R(Gamma + projective) - R(Gamma) = dA on a random connection.

    The projective shift is Gamma^lam_{mu nu} -> Gamma^lam_{mu nu} + delta^lam_nu A_mu.
    Under this shift the Ricci tensor changes by exactly the exterior derivative dA:
      R_{sigma nu}(Gamma + proj) - R_{sigma nu}(Gamma) = dA_{sigma nu}
                                                      = partial_sigma A_nu - partial_nu A_sigma

    This identity holds for ANY affine connection Gamma, not just Levi-Civita.
    It is the reason the Palatini metric equation (which depends only on the
    symmetric part of Ricci) is projective-invariant.

    Convention: noether-default-v1 + metric-affine-v1.
    """
    n = geom.dim
    coords = geom.coords
    # Random general connection (asymmetric, torsion allowed)
    gamma = random_affine_connection(conn_seed, coords)
    # Random covector for the projective shift
    A = random_covector(cov_seed, coords)
    # Build the shifted connection: Gamma + delta^lam_nu A_mu
    gamma_shifted = sp.MutableDenseNDimArray(gamma)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                if lam == nu:
                    gamma_shifted[lam, mu, nu] = _clean(
                        gamma_shifted[lam, mu, nu] + A[mu]
                    )
    gamma_shifted = Array(gamma_shifted)
    # Compute Ricci of both connections
    ric_orig = ricci_of_connection(coords, gamma)
    ric_shifted = ricci_of_connection(coords, gamma_shifted)
    # The shift should equal dA = partial_sigma A_nu - partial_nu A_sigma
    dA = exterior_derivative_of_1form(coords, A)
    residue = sp.MutableDenseNDimArray(ric_shifted)
    for sig in range(n):
        for nu in range(n):
            residue[sig, nu] = _clean(ric_shifted[sig, nu] - ric_orig[sig, nu] - dA[sig, nu])
    ok, det = all_zero(Array(residue))
    if not ok:
        return False, f"R(Gamma+proj) - R(Gamma) != dA: {det}"
    return True, (
        "Ricci shift under projective transformation equals dA "
        "on random general connection"
    )


def _palatini_projective_inert_general(
    geom: ComponentGeometry, conn_seed: int, cov_seed: int
) -> tuple[bool, str]:
    """Verify the Palatini metric equation is projective-invariant on a
    random general-connection background.

    The Palatini metric equation is:
      R_{(mu nu)}(Gamma) - 1/2 g_{mu nu} g^{ab} R_{ab}(Gamma) = 0

    Under the projective shift Gamma -> Gamma + delta^lam_nu A_mu, the Ricci
    tensor shifts by dA (antisymmetric), so the symmetric part R_{(mu nu)} is
    unchanged, the scalar R~ is unchanged (g^{ab} contracted with antisymmetric
    dA vanishes), and hence the entire metric equation is invariant.

    This test verifies the invariance on a random general-connection background
    (not just Levi-Civita + projective), which is the general statement.

    Convention: noether-default-v1 + metric-affine-v1.
    """
    n = geom.dim
    coords = geom.coords
    g = geom.g
    g_inv = geom.g_inv
    # Random general connection
    gamma = random_affine_connection(conn_seed, coords)
    # Random covector for the projective shift
    A = random_covector(cov_seed, coords)
    # Build the shifted connection
    gamma_shifted = sp.MutableDenseNDimArray(gamma)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                if lam == nu:
                    gamma_shifted[lam, mu, nu] = _clean(
                        gamma_shifted[lam, mu, nu] + A[mu]
                    )
    gamma_shifted = Array(gamma_shifted)
    # Ricci of both connections
    ric_orig = ricci_of_connection(coords, gamma)
    ric_shifted = ricci_of_connection(coords, gamma_shifted)
    # Palatini metric equation: R_{(mu nu)} - 1/2 g_{mu nu} R~
    # Original
    sym_orig = (ric_orig + sp.permutedims(ric_orig, (1, 0))) / 2
    rtilde_orig = _clean(
        sum(g_inv[a, b] * ric_orig[a, b] for a in range(n) for b in range(n))
    )
    eom_orig = sp.MutableDenseNDimArray(sym_orig)
    for a in range(n):
        for b in range(n):
            eom_orig[a, b] = _clean(sym_orig[a, b] - sp.Rational(1, 2) * g[a, b] * rtilde_orig)
    # Shifted
    sym_shifted = (ric_shifted + sp.permutedims(ric_shifted, (1, 0))) / 2
    rtilde_shifted = _clean(
        sum(g_inv[a, b] * ric_shifted[a, b] for a in range(n) for b in range(n))
    )
    eom_shifted = sp.MutableDenseNDimArray(sym_shifted)
    for a in range(n):
        for b in range(n):
            eom_shifted[a, b] = _clean(
                sym_shifted[a, b] - sp.Rational(1, 2) * g[a, b] * rtilde_shifted
            )
    # The two metric equations must agree componentwise
    residue = sp.MutableDenseNDimArray(eom_shifted)
    for a in range(n):
        for b in range(n):
            residue[a, b] = _clean(eom_shifted[a, b] - eom_orig[a, b])
    ok, det = all_zero(Array(residue))
    if not ok:
        return False, f"Palatini metric equation changed under projective shift: {det}"
    return True, (
        "Palatini metric equation is projective-invariant on random "
        "general-connection background"
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
