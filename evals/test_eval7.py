"""Executable eval 7 (docs/04_EVALS.md): k-essence / general scalar Horndeski.

Layers:
  1. Elicitation gate: coupling questions block planning until answered.
  2. Structure: the target scalar EOM is a well-formed scalar (no free indices).
  3. Decomposition: the Lagrangian fully decomposes into building blocks
     (k-essence, potential, cubic Galileon); an action with a curvature term
     does not, and is left for refusal rather than guessed.
  4. Cadabra derivation (skips if not installed): the compositional path
     assembles one script for the actual action, the kernel residue is zero,
     and the result renders in collapsed shorthand. The X-dependent k-essence
     block alone also verifies, closing the fidelity-pass gap.
"""

import pytest

from evals.eval7_kessence import (
    ELICITATION_ANSWERS,
    LAGRANGIAN_TEX,
    build_npr,
    target_scalar_eom,
)
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.blocks import (
    KESSENCE,
    assemble_scalar_eom_script,
    decompose_scalar,
)
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.llm.base import StubLLMAdapter
from noether.npr.parse import parse_lagrangian
from noether.orchestrator.derive import derive_eom
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
        assert build_npr().action.lagrangian == parse_lagrangian(LAGRANGIAN_TEX)


class TestDecomposition:
    def test_full_decomposition_into_blocks(self):
        dec = decompose_scalar(build_npr().action.lagrangian, "phi")
        assert dec.full
        kinds = sorted(m.block for m in dec.matches)
        assert kinds == ["cubic", "kessence", "potential"]
        assert any(m.block == KESSENCE and m.coupling == "K" for m in dec.matches)

    def test_higher_horndeski_term_is_not_decomposed(self):
        # an X-dependent curvature coupling G(phi, X) R is Horndeski G4, which
        # has no registered block; the decomposition is left partial so the
        # caller refuses rather than guesses.
        lag = parse_lagrangian(r"K(\phi, X) + G(\phi, X) R")
        dec = decompose_scalar(lag, "phi")
        assert not dec.full
        assert len(dec.unmatched) == 1


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelDerivation:
    def test_general_scalar_eom_verified_compositionally(self, tmp_path):
        # no model is consulted on the compositional path; the stub is inert
        derivations = derive_eom(
            build_npr(resolved=True),
            StubLLMAdapter(reply="unused"),
            {"cadabra": CadabraAdapter()},
            session_id="eval7",
            results_root=tmp_path / "results",
        )
        assert [d.wrt for d in derivations] == ["phi"]
        d = derivations[0]
        assert d.verified is True, d.checks
        assert d.checks.get("residue_zero") == "True"
        assert d.llm_name == "compositional"
        assert d.kernel_name == "cadabra"
        assert d.bundle_path
        # display collapses X and the box back to shorthand
        assert "K_{X}" in d.result_tex and "\\Box\\phi" in d.result_tex

    def test_kessence_block_alone_verifies(self):
        dec = decompose_scalar(parse_lagrangian(r"K(\phi, X)"), "phi")
        assert dec.full and dec.matches[0].block == KESSENCE
        script = assemble_scalar_eom_script(dec.matches, "phi")
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="k-essence G2 scalar variation",
                payload={"script": script},
            )
        )
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout
