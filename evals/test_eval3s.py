"""Executable eval 3s (docs/04_EVALS.md, eval 3 stretch task): spectrum.

Layers:
  1. Elicitation gate: background/stationarity/order/gauge questions must
     block planning; the documented answers unblock a "perturb" plan.
  2. Structural V0 on the presented shift, graviton equation, scalar equation.
  3. SymPy component verification (single shared computation, seven named
     checks): the conformal Ricci coefficient, the trace identity, the
     mixing-cancelling shift (with sign falsifier), the scalar kinetic
     diagonalization K_chi = (F0 + 3 F1^2)/F0, the TT null-wave dispersion
     (with non-null falsifier), and both anchors equating the linear
     operators to the eps-derivative of the FULL cadabra-verified eval-3
     equations on concrete fields.
"""

import pytest

from evals.eval3s_spectrum import (
    ELICITATION_ANSWERS,
    MU,
    NU,
    build_npr,
    diagonalizing_shift,
    graviton_eom,
    scalar_eom,
)
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.kernels.sympy_kernel.linearized import spectrum_checks
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import WellFormedCheck
from noether.verify.ladder import run_ladder


@pytest.fixture(scope="module")
def checks() -> dict[str, tuple[bool, str]]:
    return spectrum_checks()


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):
        with pytest.raises(AmbiguityBlocked):
            build_plan(build_npr(resolved=False))

    def test_documented_answers_unblock_perturb_plan(self):
        npr = build_npr(resolved=False)
        for amb in npr.ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
        plan = build_plan(npr)
        assert plan.task_type == "perturb"
        assert "linearization-anchored-to-full-eom" in plan.verification

    def test_action_is_eval3_action(self):
        from evals.eval3_scalar_tensor import build_npr as build_eval3_npr

        assert build_npr().action.lagrangian == build_eval3_npr().action.lagrangian


class TestStructure:
    def test_presented_results_well_formed(self):
        adapters = {"sympy": SympyKernelAdapter()}
        for expr, free in (
            (diagonalizing_shift(), [MU, NU]),
            (graviton_eom(), [MU, NU]),
            (scalar_eom(), []),
        ):
            report = run_ladder(expr, [WellFormedCheck(expected_free=free)], adapters)
            assert report.all_passed, report.summary()


class TestComponentVerification:
    def test_conformal_ricci(self, checks):
        ok, detail = checks["conformal-ricci"]
        assert ok, detail

    def test_trace_identity(self, checks):
        ok, detail = checks["trace-identity"]
        assert ok, detail

    def test_mixing_shift(self, checks):
        ok, detail = checks["mixing-shift"]
        assert ok, detail

    def test_scalar_diagonalization(self, checks):
        ok, detail = checks["scalar-diagonalization"]
        assert ok, detail

    def test_tt_null_wave(self, checks):
        ok, detail = checks["tt-null-wave"]
        assert ok, detail

    def test_full_eom_anchor_metric(self, checks):
        ok, detail = checks["full-eom-anchor-metric"]
        assert ok, detail

    def test_full_eom_anchor_scalar(self, checks):
        ok, detail = checks["full-eom-anchor-scalar"]
        assert ok, detail
