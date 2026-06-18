"""Tests for VAL-EOM-004: projective-mode freedom surfaced in the
connection EOM payload of a Palatini Einstein-Hilbert session.

The connection result must explicitly state Gamma is determined only up to
the projective mode (A_mu arbitrary), never present the connection as
uniquely fixed, and carry the genuine solution_zero / ricci_shift_is_dA
checks from the eval2_palatini_connection template.
"""

import pytest

from noether.kernels.base import Capability
from noether.kernels.cadabra import CadabraAdapter
from noether.llm import StubLLMAdapter
from noether.npr import (
    NOETHER_DEFAULT_V1,
    NPR,
    Action,
    ConnectionSpec,
    Geometry,
    ObjectDecl,
    Task,
)
from noether.npr.ast import Func, Sym, down, prod, tensor, up
from noether.orchestrator.derive import FieldDerivation, derive_field

requires_cadabra = pytest.mark.skipif(
    not CadabraAdapter().available(), reason="cadabra2 not installed"
)


def _palatini_eh_npr() -> NPR:
    """Build a well-posed Palatini EH NPR (Einstein-Hilbert with independent
    connection, no other matter fields)."""
    lagrangian = prod(
        tensor("g", up("mu"), up("nu")),
        tensor("R", down("mu"), down("nu"), connection="Gamma"),
    )
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(
            connection=ConnectionSpec(
                type="independent", torsion=True, nonmetricity=True
            )
        ),
        objects=[
            ObjectDecl(
                name="g",
                kind="metric",
                role="dynamical",
                symmetry="symmetric",
                rank=2,
            ),
            ObjectDecl(
                name="Gamma",
                kind="connection",
                role="dynamical",
                rank=3,
            ),
            ObjectDecl(
                name="R",
                kind="shorthand",
                role="shorthand",
                symmetry="none",
                rank=2,
                definition_tex=r"R_{\mu\nu}(\Gamma)",
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian,
            lagrangian_tex=r"g^{\mu\nu} R_{\mu\nu}(\Gamma)",
        ),
        task=Task(type="vary", with_respect_to=["g", "Gamma"]),
        ambiguities=[],
    )


class TestProjectiveFreedomPayload:
    """VAL-EOM-004: The connection EOM payload must state the projective-mode
    freedom and carry the genuine solution_zero / ricci_shift_is_dA checks."""

    @requires_cadabra
    def test_connection_eom_carries_solution_zero_and_ricci_shift_checks(
        self,
    ) -> None:
        """The connection EOM from derive_field carries the genuine
        solution_zero and ricci_shift_is_dA checks from the template."""
        npr = _palatini_eh_npr()
        cadabra = CadabraAdapter()
        llm = StubLLMAdapter(reply="")  # should NOT be called
        adapters = {"cadabra": cadabra, "sympy": None}

        result = derive_field(
            npr,
            "Gamma",
            llm,
            adapters,
            session_id="test-projective",
            results_root=None,
        )
        assert isinstance(result, FieldDerivation)
        assert result.wrt == "Gamma"
        assert result.capability is Capability.INDEPENDENT_CONNECTION
        # The checks must include the genuine template checks
        assert "solution_zero" in result.checks, (
            f"must carry solution_zero check; got checks={result.checks}"
        )
        assert "ricci_shift_is_dA" in result.checks, (
            f"must carry ricci_shift_is_dA check; got checks={result.checks}"
        )
        assert result.checks["solution_zero"] == "True", (
            f"solution_zero must be True; got {result.checks['solution_zero']}"
        )
        assert result.checks["ricci_shift_is_dA"] == "True", (
            f"ricci_shift_is_dA must be True; got {result.checks['ricci_shift_is_dA']}"
        )
        assert result.verified is True, (
            f"connection EOM must be verified when both checks pass; "
            f"got verified={result.verified}"
        )

    @requires_cadabra
    def test_connection_eom_payload_states_projective_freedom(self) -> None:
        """The result_tex / detail of the connection EOM explicitly states
        Gamma = LC(g) + delta^lambda_nu A_mu with A_mu arbitrary."""
        npr = _palatini_eh_npr()
        cadabra = CadabraAdapter()
        llm = StubLLMAdapter(reply="")  # should NOT be called
        adapters = {"cadabra": cadabra, "sympy": None}

        result = derive_field(
            npr,
            "Gamma",
            llm,
            adapters,
            session_id="test-projective",
            results_root=None,
        )

        # The payload must state the projective family
        tex_or_detail = (result.result_tex or "") + (result.detail or "")
        # Check for projective-family markers (delta and A_mu)
        assert "delta" in tex_or_detail.lower() or "projective" in tex_or_detail.lower(), (
            f"payload must mention the projective family or delta^lam_nu A_mu; "
            f"got result_tex={result.result_tex}, detail={result.detail}"
        )
        assert "A" in tex_or_detail, (
            f"payload must mention A_mu (arbitrary projective mode); "
            f"got result_tex={result.result_tex}, detail={result.detail}"
        )

    @requires_cadabra
    def test_connection_eom_does_not_present_connection_as_uniquely_fixed(
        self,
    ) -> None:
        """The detail must never present the connection as uniquely fixed.
        It must state the projective freedom (A_mu arbitrary)."""
        npr = _palatini_eh_npr()
        cadabra = CadabraAdapter()
        llm = StubLLMAdapter(reply="")
        adapters = {"cadabra": cadabra, "sympy": None}

        result = derive_field(
            npr,
            "Gamma",
            llm,
            adapters,
            session_id="test-projective",
            results_root=None,
        )

        # The detail must explicitly state the arbitrariness
        assert result.detail, "connection EOM must have non-empty detail"
        assert "arbitrary" in result.detail.lower() or "up to" in result.detail.lower(), (
            f"detail must state the projective mode is arbitrary; "
            f"got detail={result.detail}"
        )
        # Must NOT say "uniquely" or "uniquely fixed" in a positive sense
        if "uniquely" in result.detail.lower():
            # The word "uniquely" must only appear in a negation
            negated = (
                "not uniquely" in result.detail.lower()
                or "only up to" in result.detail.lower()
            )
            assert negated, (
                f"detail must not present connection as uniquely "
                f"fixed; got detail={result.detail}"
            )

    @requires_cadabra
    def test_connection_eom_uses_template_not_llm(self) -> None:
        """The connection EOM uses the eval2_palatini_connection template,
        not an LLM-generated script."""
        npr = _palatini_eh_npr()
        cadabra = CadabraAdapter()
        llm = StubLLMAdapter(reply="")  # empty reply; if used, would fail
        adapters = {"cadabra": cadabra, "sympy": None}

        result = derive_field(
            npr,
            "Gamma",
            llm,
            adapters,
            session_id="test-projective",
            results_root=None,
        )

        # The script must be the template, not LLM-generated
        assert "eval2_palatini_connection" in result.script or "solution_zero" in result.script, (
            f"script must be the eval2_palatini_connection template; "
            f"got first 200 chars: {result.script[:200]}"
        )
        # The llm_name must NOT be the stub
        assert result.llm_name != "stub", (
            f"should not use LLM for Palatini-EH connection; "
            f"got llm_name={result.llm_name}"
        )


class TestNonPalatiniEhStillUsesGeneralPath:
    """Connection variation for actions that are NOT pure Palatini EH
    should still go through the general (LLM-generated) path."""

    def test_scalar_tensor_connection_uses_llm(self) -> None:
        """Palatini scalar-tensor (F(phi)R) connection variation should
        still use the LLM-generated script path."""

        npr = NPR(
            conventions=NOETHER_DEFAULT_V1,
            geometry=Geometry(
                connection=ConnectionSpec(
                    type="independent", torsion=True, nonmetricity=True
                )
            ),
            objects=[
                ObjectDecl(
                    name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2
                ),
                ObjectDecl(
                    name="Gamma", kind="connection", role="dynamical", rank=3
                ),
                ObjectDecl(
                    name="R",
                    kind="shorthand",
                    role="shorthand",
                    symmetry="none",
                    rank=2,
                ),
                ObjectDecl(
                    name="phi", kind="scalar-field", role="dynamical", rank=0
                ),
                ObjectDecl(
                    name="F",
                    kind="function",
                    role="coupling",
                    rank=0,
                    args=["phi"],
                ),
            ],
            action=Action(
                measure_tex=r"d^4x \sqrt{-g}",
                lagrangian=prod(
                    Func(name="F", args=[Sym(name="phi")]),
                    tensor("g", up("mu"), up("nu")),
                    tensor("R", down("mu"), down("nu"), connection="Gamma"),
                ),
                lagrangian_tex=r"F(\phi) g^{\mu\nu} R_{\mu\nu}(\Gamma)",
            ),
            task=Task(type="vary", with_respect_to=["g", "Gamma", "phi"]),
            ambiguities=[],
        )

        from noether.orchestrator.derive import _is_palatini_eh_connection_variation

        assert not _is_palatini_eh_connection_variation(npr, "Gamma"), (
            "Palatini scalar-tensor should NOT be detected as pure Palatini EH"
        )
