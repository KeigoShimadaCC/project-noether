"""Executable eval 1 (docs/04_EVALS.md): acceptance gate for Horizon 1.

Three layers:
  1. The ambiguity gate blocks planning until elicitation completes.
  2. The verification ladder passes on the target result (kernel-computed
     component checks: symmetric, divergence-free, equal to expansion).
  3. The cadabra derivation reproduces the target (skips if not installed;
     required in kernel-equipped CI).
"""

import pytest

from evals.eval1_eh_trace import (
    ELICITATION_ANSWERS,
    MU,
    NU,
    build_npr,
    target_eom,
    target_eom_expanded,
)
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.npr.latex import render
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import (
    DivergenceFreeCheck,
    EqualOnBackgroundCheck,
    SymmetricCheck,
    WellFormedCheck,
)
from noether.verify.ladder import run_ladder

FAST_SPECS = [{"kind": "random-diagonal", "seed": 7, "dim": 4}]


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):
        with pytest.raises(AmbiguityBlocked):
            build_plan(build_npr(resolved=False))

    def test_documented_answers_unblock(self):
        npr = build_npr(resolved=False)
        for amb in npr.ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
        plan = build_plan(npr)
        assert plan.task_type == "vary"


class TestVerificationLadder:
    def test_target_passes_full_ladder(self):
        adapters = {"sympy": SympyKernelAdapter()}
        checks = [
            WellFormedCheck(expected_free=[MU, NU]),
            SymmetricCheck(metric_specs=FAST_SPECS),
            DivergenceFreeCheck(metric_specs=FAST_SPECS),
            EqualOnBackgroundCheck(rhs=target_eom_expanded(), metric_specs=FAST_SPECS),
        ]
        report = run_ladder(target_eom(), checks, adapters)
        assert report.all_passed, report.summary()
        # every mathematical claim was kernel-computed, not asserted
        for r in report.results:
            assert r.computed_by in ("structural", "sympy")

    def test_presentation_form(self):
        assert render(target_eom()) + " = 0" == r"G_{\mu \nu} = 0"


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelDerivation:
    def test_cadabra_derives_the_target(self):
        adapter = CadabraAdapter()
        task = KernelTask(
            capability=Capability.VARY,
            description="EH trace-form metric variation",
            payload={"template": "eval1_eh_trace"},
        )
        result = adapter.run(task)
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout
