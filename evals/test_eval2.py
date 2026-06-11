"""Executable eval 2 (docs/04_EVALS.md): Palatini acceptance gate.

Layers:
  1. The elicitation gate: the connection-independence question must block
     planning until answered; once answered the plan includes the
     independent-connection step.
  2. SymPy component verification: on explicit random backgrounds, the
     projective family Gamma = LC(g) + delta^lam_nu A_mu leaves the symmetric
     Ricci part untouched, so the Palatini metric equation reduces to
     G_{mu nu}(g) = 0.
  3. Cadabra derivation (skips if not installed): metric variation residue is
     zero against the documented target; the connection equation is solved
     identically by the projective family; Ricci(LC + projective) - Ricci(LC)
     is exactly dA.
"""

import pytest

from evals.eval2_palatini import ELICITATION_ANSWERS, MU, NU, build_npr, target_metric_eom
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.npr.latex import render
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import WellFormedCheck
from noether.verify.ladder import run_ladder


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):
        with pytest.raises(AmbiguityBlocked):
            build_plan(build_npr(resolved=False))

    def test_plan_contains_independent_connection_step(self):
        npr = build_npr(resolved=False)
        for amb in npr.ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
        plan = build_plan(npr)
        assert any(s.capability is Capability.INDEPENDENT_CONNECTION for s in plan.steps)


class TestStructureAndComponents:
    def test_metric_eom_is_well_formed(self):
        adapters = {"sympy": SympyKernelAdapter()}
        report = run_ladder(
            target_metric_eom(), [WellFormedCheck(expected_free=[MU, NU])], adapters
        )
        assert report.all_passed, report.summary()

    def test_presentation_form(self):
        tex = render(target_metric_eom())
        assert "R_{\\mu \\nu}" in tex and "R_{\\nu \\mu}" in tex

    @pytest.mark.parametrize(
        "metric_spec, seed",
        [
            ({"kind": "random-diagonal", "seed": 11, "dim": 3}, 4),
            ({"kind": "random-diagonal", "seed": 23, "dim": 4}, 9),
        ],
    )
    def test_projective_family_is_inert_on_backgrounds(self, metric_spec, seed):
        adapter = SympyKernelAdapter()
        task = KernelTask(
            capability=Capability.COMPONENT_EVAL,
            description="Palatini projective inertness",
            payload={"check": "palatini-projective-inert", "metric": metric_spec, "seed": seed},
        )
        result = adapter.run(task)
        assert result.value["passed"], result.value["detail"]


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelDerivation:
    def test_metric_variation_residue_zero(self):
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="Palatini metric variation",
                payload={"template": "eval2_palatini_metric"},
            )
        )
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout

    def test_connection_equation_solved_by_projective_family(self):
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="Palatini connection variation",
                payload={"template": "eval2_palatini_connection"},
            )
        )
        checks = result.value["checks"]
        assert checks.get("solution_zero") == "True", result.raw.stdout
        assert checks.get("ricci_shift_is_dA") == "True", result.raw.stdout
