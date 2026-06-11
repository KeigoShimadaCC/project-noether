"""Executable eval 4 (docs/04_EVALS.md): Maxwell acceptance gate.

Layers:
  1. Elicitation gate: the metric-role question must block planning; the NPR
     marks g as background (role discipline at the representation level).
  2. SymPy component verification: the Noether identity
     nabla_nu nabla_mu F^{mu nu} = 0 holds for a RANDOM antisymmetric F on
     random curved backgrounds, off shell, exactly as gauge invariance forces.
  3. Cadabra derivation (skips if not installed): the variation residue is
     zero against sg nabla_mu F^{mu nu} dA_nu. Role discipline at kernel
     level: the audited template's only vary() rule touches F through dA.
"""

import pytest

from evals.eval4_maxwell import (
    NU,
    build_npr,
    noether_identity_expr,
    target_eom,
)
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra import templates as cadabra_templates
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import WellFormedCheck
from noether.verify.ladder import run_ladder

BACKGROUNDS = [
    {"metric": {"kind": "random-diagonal", "seed": 7, "dim": 3}, "f_seed": 5},
    {"metric": {"kind": "random-diagonal", "seed": 19, "dim": 4}, "f_seed": 12},
]


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):
        with pytest.raises(AmbiguityBlocked):
            build_plan(build_npr(resolved=False))

    def test_metric_is_background_and_not_varied(self):
        npr = build_npr(resolved=True)
        assert npr.object_named("g").role == "background"
        assert npr.task.with_respect_to == ["A"]
        plan = build_plan(npr)
        vary_steps = [s for s in plan.steps if s.capability is Capability.VARY]
        assert vary_steps and all(s.payload.get("with_respect_to") == ["A"] for s in vary_steps)


class TestStructure:
    def test_eom_well_formed(self):
        adapters = {"sympy": SympyKernelAdapter()}
        report = run_ladder(target_eom(), [WellFormedCheck(expected_free=[NU])], adapters)
        assert report.all_passed, report.summary()


class TestComponentVerification:
    @pytest.mark.parametrize("bg", BACKGROUNDS)
    def test_noether_identity_off_shell(self, bg):
        result = SympyKernelAdapter().run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="Maxwell Noether identity",
                payload={
                    "check": "zero",
                    "expr": noether_identity_expr().model_dump(),
                    "metric": bg["metric"],
                    "fields": {"F": {"kind": "random-antisymmetric", "seed": bg["f_seed"]}},
                },
            )
        )
        assert result.value["passed"], result.value["detail"]


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelDerivation:
    def test_variation_residue_zero(self):
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="Maxwell variation on fixed background",
                payload={"template": "eval4_maxwell"},
            )
        )
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout

    def test_role_discipline_in_template(self):
        """The audited script never varies the metric: its single vary() rule
        rewrites F into derivatives of dA only."""
        src = cadabra_templates.get("eval4_maxwell")
        vary_lines = [ln for ln in src.splitlines() if ln.startswith("vary(")]
        assert len(vary_lines) == 1
        assert "dA" in vary_lines[0]
        assert "g^" not in vary_lines[0] and "g_" not in vary_lines[0] and "sg" not in vary_lines[0]
