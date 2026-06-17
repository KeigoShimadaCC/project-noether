"""Pin the reusable curvature reduction primitives to their defining identities.

Each test builds a tiny Cadabra script that constructs an identity from scratch,
applies one primitive from ``noether.kernels.cadabra.curvature``, and asks the
kernel to confirm the residue against the known right-hand side is zero. This is
the same audit the frozen templates get: the primitive is trusted because the
kernel checks it, not because it looks right. They skip when cadabra2 is absent.

The contracted Bianchi identity is a citable standard result (the Einstein
tensor is divergence-free), so its test only confirms the substitution fires on
the canonical left-hand side; it is not re-derived here.
"""

import pytest

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra import curvature as cv

_BASE_DECL = r"""{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Integer(range=0..3).
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g^{\mu}_{\nu}::KroneckerDelta.
g_{\mu}^{\nu}::KroneckerDelta.
"""


def _script(body: str) -> str:
    return (
        _BASE_DECL
        + cv.CURVATURE_DECL
        + "\n"
        + r"{phi, R, R_{\mu\nu}}::Depends(\nabla{#})."
        + "\n"
        + body
    )


def _run(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="curvature primitive check",
            payload={"script": _script(body)},
        )
    )


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestCurvaturePrimitives:
    def test_commutator_reduces_to_riemann(self):
        body = (
            r"comm := \nabla_{\alpha}{\nabla_{\beta}{\nabla_{\gamma}{phi}}}"
            r" - \nabla_{\beta}{\nabla_{\alpha}{\nabla_{\gamma}{phi}}};"
            "\n" + cv.commute_third_derivative("phi", "comm") + "\n"
            "distribute(comm); canonicalise(comm); rename_dummies(comm);\n"
            r"target := - g^{\delta\lambda} R_{\lambda\gamma\alpha\beta} \nabla_{\delta}{phi};"
            "\n"
            "distribute(target); canonicalise(target); rename_dummies(target);\n"
            "residue := @(comm) - @(target);\n"
            "distribute(residue); canonicalise(residue); rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: commutator_zero=" + str(str(residue) == "0"))'
        )
        result = _run(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("commutator_zero") == "True", result.raw.stdout

    def test_oneway_commutator_matches_difference_form(self):
        # The single-term rule must produce the same Riemann term as the
        # already-verified difference form: applying it to a bare triple and
        # subtracting the swapped triple leaves exactly the difference-form RHS.
        body = (
            r"single := \nabla_{\alpha}{\nabla_{\beta}{\nabla_{\gamma}{phi}}};"
            "\n" + cv.commute_third_derivative_oneway("phi", "single") + "\n"
            r"single := @(single) - \nabla_{\beta}{\nabla_{\alpha}{\nabla_{\gamma}{phi}}};"
            "\n"
            r"diff := \nabla_{\alpha}{\nabla_{\beta}{\nabla_{\gamma}{phi}}}"
            r" - \nabla_{\beta}{\nabla_{\alpha}{\nabla_{\gamma}{phi}}};"
            "\n" + cv.commute_third_derivative("phi", "diff") + "\n"
            "residue := @(single) - @(diff);\n"
            "distribute(residue); canonicalise(residue); rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: oneway_zero=" + str(str(residue) == "0"))'
        )
        result = _run(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("oneway_zero") == "True", result.raw.stdout

    def test_fold_riemann_to_ricci(self):
        body = (
            r"ex := g^{\alpha\gamma} R_{\alpha\beta\gamma\delta};"
            "\n"
            "canonicalise(ex);\n" + cv.fold_ricci("ex") + "\n"
            "target := R_{\\beta\\delta};\n"
            "residue := @(ex) - @(target);\n"
            "canonicalise(residue);\n"
            'print("NOETHER_CHECK: ricci_zero=" + str(str(residue) == "0"))'
        )
        result = _run(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("ricci_zero") == "True", result.raw.stdout

    def test_fold_ricci_trace_to_scalar(self):
        body = (
            r"ex := g^{\beta\delta} g^{\alpha\gamma} R_{\alpha\beta\gamma\delta};"
            "\n"
            "canonicalise(ex);\n" + cv.fold_ricci("ex") + "\n" + cv.fold_scalar("ex") + "\n"
            "canonicalise(ex);\n"
            "residue := @(ex) - R;\n"
            'print("NOETHER_CHECK: scalar_zero=" + str(str(residue) == "0"))'
        )
        result = _run(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("scalar_zero") == "True", result.raw.stdout

    def test_quartic_box_combination_reduces_to_second_order(self):
        # The quartic Horndeski no-Ostrogradski combination
        # 2 G4X (box^2 phi - nabla_a nabla_b nabla^b nabla^a phi) must collapse
        # to a purely second-order curvature coupling: no derivative of phi of
        # order three or higher may survive.
        body = (
            r"ex := 2 G4X g^{\mu\nu} g^{\rho\sigma}"
            r" \nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{\nabla_{\sigma}{phi}}}}"
            r" - 2 G4X g^{\mu\rho} g^{\nu\sigma}"
            r" \nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{\nabla_{\sigma}{phi}}}};"
            "\n" + cv.commute_fourth_cross("phi", "ex") + "\n"
            "distribute(ex); product_rule(ex); distribute(ex);\n"
            r"substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);"
            "\n"
            "distribute(ex); canonicalise(ex); rename_dummies(ex); meld(ex);\n"
            "chk := @(ex);\n"
            r"substitute(chk, $\nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{phi}}} -> 0$);"
            "\n"
            "diff := @(ex) - @(chk);\n"
            "canonicalise(diff); rename_dummies(diff); meld(diff);\n"
            'print("NOETHER_CHECK: second_order=" + str(str(diff) == "0"))'
        )
        result = _run(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("second_order") == "True", result.raw.stdout

    def test_scalar_hessian_symmetrizes(self):
        body = (
            r"ex := \nabla_{\alpha}{\nabla_{\beta}{phi}} - \nabla_{\beta}{\nabla_{\alpha}{phi}};"
            "\n" + cv.hessian_to_symmetric("phi", "ex") + "\n"
            "canonicalise(ex);\n"
            'print("NOETHER_CHECK: hessian_zero=" + str(str(ex) == "0"))'
        )
        result = _run(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("hessian_zero") == "True", result.raw.stdout

    def test_contracted_bianchi_fires(self):
        body = (
            r"ex := g^{\mu\nu} \nabla_{\mu}{R_{\nu\beta}};"
            "\n" + cv.contracted_bianchi("ex") + "\n"
            r"target := 1/2 \nabla_{\beta}{R};"
            "\n"
            "residue := @(ex) - @(target);\n"
            "distribute(residue); canonicalise(residue);\n"
            'print("NOETHER_CHECK: bianchi_zero=" + str(str(residue) == "0"))'
        )
        result = _run(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("bianchi_zero") == "True", result.raw.stdout
