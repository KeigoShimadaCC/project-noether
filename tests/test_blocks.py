"""Unit tests for the compositional building-block decomposition and the
Cadabra script assembler (noether/kernels/cadabra/blocks.py). These run with
no kernel; the residue verification of the assembled scripts is exercised by
evals/test_eval7.py under the cadabra marker."""

from fractions import Fraction

from noether.kernels.cadabra.blocks import (
    CUBIC,
    EINSTEIN_HILBERT,
    KESSENCE,
    KINETIC,
    NONMINIMAL,
    POTENTIAL,
    assemble_metric_eom_script,
    assemble_scalar_eom_script,
    compose_display_tex,
    compose_metric_display_tex,
    coupling_symbols,
    decompose_metric,
    decompose_scalar,
)
from noether.npr.parse import parse_lagrangian

SCALAR_TENSOR = r"F(\phi) R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)"
G4_DENSITY = r"G(\phi, X) R - V(\phi)"

GENERAL = r"K(\phi, X) - V(\phi) + G(\phi)\Box\phi"
CANONICAL = r"- \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)"


class TestDecomposition:
    def test_general_action_blocks(self):
        dec = decompose_scalar(parse_lagrangian(GENERAL), "phi")
        assert dec.full
        by_block = {m.block: m for m in dec.matches}
        assert set(by_block) == {KESSENCE, POTENTIAL, CUBIC}
        assert by_block[KESSENCE].coupling == "K"
        assert by_block[CUBIC].coupling == "G"
        assert by_block[POTENTIAL].coupling == "V"

    def test_canonical_kinetic_coefficient(self):
        dec = decompose_scalar(parse_lagrangian(CANONICAL), "phi")
        kinetic = next(m for m in dec.matches if m.block == KINETIC)
        assert kinetic.coeff == Fraction(-1, 2)
        assert kinetic.coupling is None

    def test_phi_only_function_is_potential_not_kessence(self):
        dec = decompose_scalar(parse_lagrangian(r"K(\phi)"), "phi")
        assert dec.full
        assert dec.matches[0].block == POTENTIAL

    def test_nonminimal_curvature_matched(self):
        # F(phi) R is the nonminimal coupling block; it contributes F_phi R.
        dec = decompose_scalar(parse_lagrangian(SCALAR_TENSOR), "phi")
        assert dec.full
        by_block = {m.block: m for m in dec.matches}
        assert set(by_block) == {NONMINIMAL, KINETIC, POTENTIAL}
        assert by_block[NONMINIMAL].coupling == "F"

    def test_g4_density_unmatched(self):
        # G(phi, X) R is Horndeski G4, held out until it verifies.
        dec = decompose_scalar(parse_lagrangian(G4_DENSITY), "phi")
        assert not dec.full
        assert len(dec.unmatched) == 1

    def test_box_only_function_of_x_refused(self):
        # a pure K(X) (no phi argument) is not the G2 block we register; it is
        # left unmatched rather than mismodeled with a spurious K_phi term.
        dec = decompose_scalar(parse_lagrangian(r"K(X)"), "phi")
        assert not dec.full


class TestCouplingSymbols:
    def test_kessence_symbols(self):
        dec = decompose_scalar(parse_lagrangian(r"K(\phi, X)"), "phi")
        sym = coupling_symbols(dec.matches[0])
        assert sym == {"phi": "Kphi", "X": "KX", "Xphi": "KXphi", "XX": "KXX"}

    def test_cubic_symbols(self):
        dec = decompose_scalar(parse_lagrangian(r"G(\phi)\Box\phi"), "phi")
        sym = coupling_symbols(dec.matches[0])
        assert sym == {"d1": "Gp", "d2": "Gpp"}


class TestAssembler:
    def test_script_has_residue_check_and_two_pass_ibp(self):
        dec = decompose_scalar(parse_lagrangian(GENERAL), "phi")
        src = assemble_scalar_eom_script(dec.matches, "phi")
        # two integration-by-parts passes (outer derivative, then inner)
        assert src.count("integrate_by_parts") == 2
        assert "NOETHER_CHECK: residue_zero=" in src
        # the k-essence chain rule reintroduces phi-derivatives in-kernel
        assert r"\nabla_{\mu}{X} ->" in src
        assert "vary(ex," in src
        # the assembled integrand carries the real couplings
        assert "sg K" in src and "sg V" in src and "sg G" in src

    def test_script_declares_kessence_helpers(self):
        dec = decompose_scalar(parse_lagrangian(r"K(\phi, X)"), "phi")
        src = assemble_scalar_eom_script(dec.matches, "phi")
        for name in ("Kphi", "KX", "KXphi", "KXX", "dX", "X"):
            assert name in src


class TestDisplay:
    def test_canonical_collapses_to_box(self):
        dec = decompose_scalar(parse_lagrangian(CANONICAL), "phi")
        tex = compose_display_tex(dec.matches, "phi")
        assert tex == r"\Box\phi - V_{\phi} = 0"

    def test_general_display_uses_shorthands(self):
        dec = decompose_scalar(parse_lagrangian(GENERAL), "phi")
        tex = compose_display_tex(dec.matches, "phi")
        assert "K_{X}\\Box\\phi" in tex
        assert "K_{X\\phi}(\\nabla\\phi)^2" in tex
        assert "2 G_{\\phi}\\Box\\phi" in tex
        assert tex.endswith("= 0")

    def test_nonminimal_scalar_display(self):
        dec = decompose_scalar(parse_lagrangian(SCALAR_TENSOR), "phi")
        tex = compose_display_tex(dec.matches, "phi")
        assert "F_{\\phi} R" in tex
        assert tex.endswith("= 0")


class TestMetricDecomposition:
    def test_scalar_tensor_metric_blocks(self):
        dec = decompose_metric(parse_lagrangian(SCALAR_TENSOR), "phi")
        assert dec.full
        by_block = {m.block: m for m in dec.matches}
        assert set(by_block) == {NONMINIMAL, KINETIC, POTENTIAL}
        assert by_block[NONMINIMAL].coupling == "F"

    def test_vacuum_einstein_hilbert(self):
        dec = decompose_metric(parse_lagrangian(r"R"), "phi")
        assert dec.full
        assert dec.matches[0].block == EINSTEIN_HILBERT

    def test_g4_density_metric_unmatched(self):
        dec = decompose_metric(parse_lagrangian(G4_DENSITY), "phi")
        assert not dec.full

    def test_metric_display_is_modified_einstein(self):
        dec = decompose_metric(parse_lagrangian(SCALAR_TENSOR), "phi")
        tex = compose_metric_display_tex(dec.matches)
        assert "F R_{\\mu\\nu}" in tex
        assert "\\nabla_{\\mu}\\nabla_{\\nu} F" in tex
        assert tex.endswith("= 0")

    def test_vacuum_display_is_einstein_tensor(self):
        dec = decompose_metric(parse_lagrangian(r"R"), "phi")
        tex = compose_metric_display_tex(dec.matches)
        assert tex == r"R_{\mu\nu} - \tfrac{1}{2} g_{\mu\nu} R = 0"


class TestMetricAssembler:
    def test_metric_script_has_dgamma_and_two_pass_ibp(self):
        dec = decompose_metric(parse_lagrangian(SCALAR_TENSOR), "phi")
        src = assemble_metric_eom_script(dec.matches)
        assert src.count("integrate_by_parts") == 2
        assert "dGamma" in src
        assert "NOETHER_CHECK: residue_zero=" in src
        assert "sg F g^{\\alpha\\beta} R_{\\alpha\\beta}" in src
