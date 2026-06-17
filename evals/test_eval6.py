"""Executable eval 6 (docs/04_EVALS.md): cubic Galileon scalar acceptance gate.

Layers:
  1. Elicitation gate: the coupling questions must block planning until answered.
  2. Structure: the target scalar EOM is a well-formed scalar (no free indices).
  3. Worked-example routing: the box-phi coupling routes scalar variation to the
     audited eom_cubic_galileon_scalar scaffold, not the plain eval-3 example.
  4. Cadabra derivation (skips if not installed): the scaffold residue is zero,
     and the general path (model stubbed to the audited script) reports the
     scalar field equation verified by the kernel's own residue check.
"""

import pytest

from evals.eval6_cubic_galileon import (
    ELICITATION_ANSWERS,
    LAGRANGIAN_TEX,
    build_npr,
    target_scalar_eom,
)
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter, templates
from noether.kernels.cadabra.blocks import (
    CUBIC,
    assemble_metric_eom_script,
    decompose_metric,
)
from noether.kernels.cadabra.generate import _variation_key
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.llm.base import StubLLMAdapter
from noether.npr.parse import parse_lagrangian
from noether.orchestrator.derive import derive_eom
from noether.orchestrator.ingest import ingest_action
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import WellFormedCheck
from noether.verify.ladder import run_ladder


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
    def test_scalar_eom_well_formed(self):
        report = run_ladder(
            target_scalar_eom(),
            [WellFormedCheck(expected_free=[])],
            {"sympy": SympyKernelAdapter()},
        )
        assert report.all_passed, report.summary()

    def test_action_parses(self):
        # the stored AST matches a fresh parse of the documented tex
        assert build_npr().action.lagrangian == parse_lagrangian(LAGRANGIAN_TEX)


class TestWorkedExampleRouting:
    def test_box_coupling_routes_to_cubic_scaffold(self):
        assert _variation_key(build_npr(resolved=True), "phi") == "vary-scalar-cubic"

    def test_ingested_action_also_routes_to_cubic_scaffold(self):
        npr = ingest_action(r"d^4x \sqrt{-g}", LAGRANGIAN_TEX).npr
        assert _variation_key(npr, "phi") == "vary-scalar-cubic"


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelDerivation:
    def test_scaffold_residue_zero(self):
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="cubic Galileon scalar variation",
                payload={"template": "eom_cubic_galileon_scalar"},
            )
        )
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout

    def test_general_path_scalar_eom_verified(self, tmp_path):
        stub = StubLLMAdapter(reply=templates.get("eom_cubic_galileon_scalar"))
        derivations = derive_eom(
            build_npr(resolved=True),
            stub,
            {"cadabra": CadabraAdapter()},
            session_id="eval6",
            results_root=tmp_path / "results",
        )
        assert [d.wrt for d in derivations] == ["phi"]
        d = derivations[0]
        assert d.verified is True, d.checks
        assert d.result_tex
        assert d.kernel_name == "cadabra"
        assert d.bundle_path

    def test_cubic_galileon_metric_eom_verifies(self):
        # Cubic Galileon coupled to Einstein gravity: the metric EOM composes
        # and the kernel residue is zero. The cubic stress is the kinetic stress
        # of -G_phi, obtained by varying nabla nabla phi directly (Hess block).
        lag = parse_lagrangian(
            r"R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi) + G(\phi)\Box\phi"
        )
        dec = decompose_metric(lag, "phi")
        assert dec.full and any(m.block == CUBIC for m in dec.matches)
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="cubic Galileon metric variation",
                payload={"script": assemble_metric_eom_script(dec.matches)},
            )
        )
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout
