"""Executable eval 3 (docs/04_EVALS.md): scalar-tensor acceptance gate.

Layers:
  1. Elicitation gate: the F(phi) coupling question must block planning.
  2. Structural V0 on both targets (the metric EOM contains unknown functions
     F, V, so component checks run on the F = 1, V = 0 limit instead).
  3. SymPy component verification on random backgrounds with a random phi:
     the minimal-limit metric EOM is symmetric, and its divergence equals
     -1/2 box(phi) nabla_nu phi exactly (the generalized Bianchi link, which
     makes the system consistent on the scalar shell).
  4. Cadabra derivation (skips if not installed): metric and scalar variation
     residues are both zero against the documented targets.
"""

import pytest

from evals.eval3_scalar_tensor import (
    ELICITATION_ANSWERS,
    MU,
    NU,
    bianchi_link_lhs,
    bianchi_link_rhs,
    build_npr,
    minimal_limit_metric_eom,
    target_metric_eom,
    target_scalar_eom,
)
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import WellFormedCheck
from noether.verify.ladder import run_ladder

BACKGROUNDS = [
    {"metric": {"kind": "random-diagonal", "seed": 13, "dim": 3}, "phi_seed": 9},
    {"metric": {"kind": "random-diagonal", "seed": 31, "dim": 4}, "phi_seed": 17},
]


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
    def test_metric_eom_well_formed(self):
        adapters = {"sympy": SympyKernelAdapter()}
        report = run_ladder(
            target_metric_eom(), [WellFormedCheck(expected_free=[MU, NU])], adapters
        )
        assert report.all_passed, report.summary()

    def test_scalar_eom_well_formed(self):
        adapters = {"sympy": SympyKernelAdapter()}
        report = run_ladder(target_scalar_eom(), [WellFormedCheck(expected_free=[])], adapters)
        assert report.all_passed, report.summary()


class TestComponentVerification:
    @pytest.mark.parametrize("bg", BACKGROUNDS)
    def test_minimal_limit_eom_is_symmetric(self, bg):
        result = SympyKernelAdapter().run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="minimal-limit metric EOM symmetric",
                payload={
                    "check": "symmetric",
                    "expr": minimal_limit_metric_eom().model_dump(),
                    "metric": bg["metric"],
                    "fields": {"phi": {"kind": "random-scalar", "seed": bg["phi_seed"]}},
                },
            )
        )
        assert result.value["passed"], result.value["detail"]

    @pytest.mark.parametrize("bg", BACKGROUNDS)
    def test_generalized_bianchi_link(self, bg):
        """nabla^mu E_{mu nu} = -1/2 box(phi) nabla_nu phi, exactly, off shell."""
        result = SympyKernelAdapter().run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="generalized Bianchi link",
                payload={
                    "check": "equal",
                    "lhs": bianchi_link_lhs().model_dump(),
                    "rhs": bianchi_link_rhs().model_dump(),
                    "metric": bg["metric"],
                    "fields": {"phi": {"kind": "random-scalar", "seed": bg["phi_seed"]}},
                },
            )
        )
        assert result.value["passed"], result.value["detail"]


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelDerivation:
    def test_metric_variation_residue_zero(self):
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="scalar-tensor metric variation",
                payload={"template": "eval3_scalar_tensor_metric"},
            )
        )
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout

    def test_scalar_variation_residue_zero(self):
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="scalar-tensor scalar variation",
                payload={"template": "eval3_scalar_tensor_scalar"},
            )
        )
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout
