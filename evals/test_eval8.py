"""Executable eval 8 (docs/04_EVALS.md): nonminimal scalar-tensor by composition.

Layers:
  1. Elicitation gate: coupling questions block planning until answered.
  2. Decomposition: F(phi) R + kinetic + potential decomposes fully in BOTH
     sectors (scalar and metric); an X-dependent curvature coupling (Horndeski
     G4) does not, and is left for refusal rather than guessed.
  3. Cadabra derivation (skips if not installed): the compositional path
     assembles one script per sector for the actual action, the kernel residue
     is zero for both EOMs, and each result renders in clean shorthand. The
     Einstein-Hilbert block alone (vacuum GR) also verifies as G_{mu nu} = 0.
"""

import pytest

from evals.eval8_nonminimal import ELICITATION_ANSWERS, build_npr
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.blocks import (
    EINSTEIN_HILBERT,
    NONMINIMAL,
    assemble_metric_eom_script,
    decompose_metric,
    decompose_scalar,
)
from noether.llm.base import StubLLMAdapter
from noether.npr.parse import parse_lagrangian
from noether.orchestrator.derive import derive_eom
from noether.orchestrator.planner import AmbiguityBlocked, build_plan


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):
        with pytest.raises(AmbiguityBlocked):
            build_plan(build_npr(resolved=False))

    def test_documented_answers_unblock(self):
        npr = build_npr(resolved=False)
        for amb in npr.ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
        assert build_plan(npr).task_type == "vary"


class TestDecomposition:
    def test_scalar_sector_decomposes_fully(self):
        dec = decompose_scalar(build_npr().action.lagrangian, "phi")
        assert dec.full
        assert any(m.block == NONMINIMAL and m.coupling == "F" for m in dec.matches)

    def test_metric_sector_decomposes_fully(self):
        dec = decompose_metric(build_npr().action.lagrangian, "phi")
        assert dec.full
        assert any(m.block == NONMINIMAL and m.coupling == "F" for m in dec.matches)

    def test_vacuum_einstein_hilbert(self):
        dec = decompose_metric(parse_lagrangian(r"R"), "phi")
        assert dec.full and dec.matches[0].block == EINSTEIN_HILBERT

    def test_g4_density_is_not_decomposed(self):
        lag = parse_lagrangian(r"G(\phi, X) R - V(\phi)")
        assert not decompose_metric(lag, "phi").full
        assert not decompose_scalar(lag, "phi").full


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelDerivation:
    def test_both_eoms_verified_compositionally(self, tmp_path):
        # no model is consulted on the compositional path; the stub is inert
        derivations = derive_eom(
            build_npr(resolved=True),
            StubLLMAdapter(reply="unused"),
            {"cadabra": CadabraAdapter()},
            session_id="eval8",
            results_root=tmp_path / "results",
        )
        by_field = {d.wrt: d for d in derivations}
        assert set(by_field) == {"g", "phi"}
        for d in by_field.values():
            assert d.verified is True, d.checks
            assert d.checks.get("residue_zero") == "True"
            assert d.llm_name == "compositional"
            assert d.kernel_name == "cadabra"
            assert d.bundle_path
        # scalar EOM collapses to shorthand; metric EOM is the modified Einstein eq
        assert "F_{\\phi} R" in by_field["phi"].result_tex
        assert "\\Box\\phi" in by_field["phi"].result_tex
        metric_tex = by_field["g"].result_tex
        assert "F R_{\\mu\\nu}" in metric_tex
        assert "\\nabla_{\\mu}\\nabla_{\\nu} F" in metric_tex

    def test_vacuum_einstein_equation_verifies(self):
        dec = decompose_metric(parse_lagrangian(r"R"), "phi")
        script = assemble_metric_eom_script(dec.matches)
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="vacuum Einstein-Hilbert metric variation",
                payload={"script": script},
            )
        )
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout
