"""Executable eval 5 (docs/04_EVALS.md): Gauss-Bonnet acceptance gate.

Layers:
  1. Elicitation gate: the symbolic-dimension question must block planning.
  2. Structural V0 on the Lanczos target (explicit literature form).
  3. SymPy component verification (Lovelock properties, all on explicit
     curved backgrounds):
       - D=4: H == 0 identically, on a sparse random metric AND on a warped
         metric whose GB scalar is NONZERO (the cancellation is real);
       - D=5: H != 0 (falsifier: the theory is dynamical above four
         dimensions), H symmetric, divergence-free.
  4. Cadabra (skips if not installed): the full variational derivation
     (delta S -> Lanczos H, double IBP + Bianchi/commutator reduction,
     residue zero in general D) and the symbolic-D Lovelock algebra (the p=2
     delta contraction is the GB scalar, the Lovelock field-equation
     contraction equals the documented Lanczos form).
"""

import pytest

from evals.eval5_gauss_bonnet import (
    ELICITATION_ANSWERS,
    MU,
    NU,
    build_npr,
    lanczos_shorthand,
    target_eom,
)
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import WellFormedCheck
from noether.verify.ladder import run_ladder

SPARSE_4D = {"kind": "sparse-diagonal", "seed": 7, "dim": 4, "curved": 3}
WARPED_4D = {"kind": "warped-product-4d"}
SPARSE_5D = {"kind": "sparse-diagonal", "seed": 7, "dim": 5, "curved": 3}


def _component(check: str, metric: dict, expr=None):
    payload = {"check": check, "metric": metric}
    if expr is not None:
        payload["expr"] = expr.model_dump()
    return SympyKernelAdapter().run(
        KernelTask(capability=Capability.COMPONENT_EVAL, description=check, payload=payload)
    )


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):
        with pytest.raises(AmbiguityBlocked):
            build_plan(build_npr(resolved=False))

    def test_documented_answers_unblock(self):
        npr = build_npr(resolved=False)
        for amb in npr.ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
        assert build_plan(npr).task_type == "vary"


class TestStructure:
    def test_lanczos_target_well_formed(self):
        adapters = {"sympy": SympyKernelAdapter()}
        report = run_ladder(target_eom(), [WellFormedCheck(expected_free=[MU, NU])], adapters)
        assert report.all_passed, report.summary()


class TestLovelockProperties:
    def test_d4_identically_zero_sparse(self):
        result = _component("zero", SPARSE_4D, lanczos_shorthand())
        assert result.value["passed"], result.value["detail"]

    def test_d4_identically_zero_with_nonzero_gb_scalar(self):
        """The strongest D=4 check: on this background the GB scalar itself is
        nonzero, so H == 0 is a genuine cancellation between its four
        quadratic-curvature pieces (the dimension-dependent identity)."""
        from noether.kernels.sympy_kernel.geometry import warped_product_4d

        geom = warped_product_4d()
        import sympy as sp

        assert sp.simplify(geom.gauss_bonnet_scalar) != 0
        result = _component("zero", WARPED_4D, lanczos_shorthand())
        assert result.value["passed"], result.value["detail"]

    def test_d5_nonzero_falsifier(self):
        """Above four dimensions the theory is dynamical: H must NOT vanish."""
        result = _component("zero", SPARSE_5D, lanczos_shorthand())
        assert not result.value["passed"], "H vanished in D=5; eval background too degenerate"

    def test_d5_symmetric(self):
        result = _component("symmetric", SPARSE_5D, lanczos_shorthand())
        assert result.value["passed"], result.value["detail"]

    def test_d5_divergence_free(self):
        result = _component("divergence-zero", SPARSE_5D, lanczos_shorthand())
        assert result.value["passed"], result.value["detail"]


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelAlgebra:
    def test_lovelock_delta_algebra_symbolic_d(self):
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.CANONICALIZE,
                description="Lovelock p=2 delta algebra, symbolic D",
                payload={"template": "eval5_gauss_bonnet"},
            )
        )
        checks = result.value["checks"]
        assert checks.get("gb_scalar_zero") == "True", result.raw.stdout
        assert checks.get("lanczos_form_zero") == "True", result.raw.stdout

    def test_variational_derivation_residue_zero(self):
        """delta of int sqrt(-g) GB: Palatini variation, double IBP,
        contracted-Bianchi + commutator reduction, residue against the
        Lanczos H exactly zero in general dimension. The reduction rules
        in the template were sympy-verified on a curved background first."""
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="GB variational derivation",
                payload={"template": "eval5_gauss_bonnet_variation"},
            )
        )
        checks = result.value["checks"]
        assert checks.get("variation_residue_zero") == "True", result.raw.stdout
