"""Executable eval 1s (docs/04_EVALS.md, eval 1 stretch task): ADM of GR.

Layers:
  1. Elicitation gate: foliation/K-sign/boundary questions must block planning;
     the documented answers unblock an "adm" plan.
  2. Structural V0 on the presented bulk density and constraints.
  3. SymPy component verification on a nondegenerate 1+2 background (single
     shared computation, six named checks): extrinsic curvature equals the
     normal gradient, the Gauss-Codazzi split of sqrt(-g) R, both constraint
     projections of the Einstein tensor, the lapse Euler-Lagrange equation,
     and falsifier hygiene (every structural feature of the background is on).
"""

import pytest

from evals.eval1s_adm import (
    ELICITATION_ANSWERS,
    B,
    build_npr,
    bulk_density,
    hamiltonian_constraint,
    momentum_constraint,
)
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.kernels.sympy_kernel.adm import adm_sample_1p2
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import WellFormedCheck
from noether.verify.ladder import run_ladder


@pytest.fixture(scope="module")
def adm_checks() -> dict[str, tuple[bool, str]]:
    return adm_sample_1p2().run_all()


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):
        with pytest.raises(AmbiguityBlocked):
            build_plan(build_npr(resolved=False))

    def test_documented_answers_unblock_adm_plan(self):
        npr = build_npr(resolved=False)
        for amb in npr.ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
        plan = build_plan(npr)
        assert plan.task_type == "adm"
        assert "constraints-as-normal-projections" in plan.verification


class TestStructure:
    def test_presented_results_well_formed(self):
        adapters = {"sympy": SympyKernelAdapter()}
        for expr, free in (
            (bulk_density(), []),
            (hamiltonian_constraint(), []),
            (momentum_constraint(), [B]),
        ):
            report = run_ladder(expr, [WellFormedCheck(expected_free=free)], adapters)
            assert report.all_passed, report.summary()


class TestComponentVerification:
    def test_background_nondegenerate(self, adm_checks):
        ok, detail = adm_checks["background-nondegenerate"]
        assert ok, detail

    def test_extrinsic_curvature_is_normal_gradient(self, adm_checks):
        ok, detail = adm_checks["extrinsic-curvature-normal-gradient"]
        assert ok, detail

    def test_lagrangian_split(self, adm_checks):
        ok, detail = adm_checks["lagrangian-split"]
        assert ok, detail

    def test_hamiltonian_projection(self, adm_checks):
        ok, detail = adm_checks["hamiltonian-projection"]
        assert ok, detail

    def test_momentum_projection(self, adm_checks):
        ok, detail = adm_checks["momentum-projection"]
        assert ok, detail

    def test_lapse_euler_lagrange(self, adm_checks):
        ok, detail = adm_checks["lapse-euler-lagrange"]
        assert ok, detail
