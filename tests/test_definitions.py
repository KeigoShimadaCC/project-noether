"""Readability shorthand proposals: deterministic, pure, human-confirmed.

The contract: propose_definitions derives derivative shorthands from the
declared function couplings without touching the NPR; Session.add_definition
adopts only what is passed to it, as a new immutable version, and never
reopens the ambiguity gate.
"""

import pytest

from noether.orchestrator.definitions import propose_definitions
from noether.orchestrator.ingest import ingest_action
from noether.orchestrator.session import Session

MEASURE = r"d^4x \sqrt{-g}"
SCALAR_TENSOR = r"F(\phi) R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)"


def _npr(lagrangian: str = SCALAR_TENSOR):
    return ingest_action(MEASURE, lagrangian).npr


class TestIngestCapturesFunctionArgs:
    def test_function_args_recorded(self):
        npr = _npr()
        funcs = {o.name: o for o in npr.objects if o.kind == "function"}
        assert funcs["F"].args == ["phi"]
        assert funcs["V"].args == ["phi"]


class TestProposeDefinitions:
    def test_proposes_first_and_second_derivatives(self):
        proposals = propose_definitions(_npr())
        by_symbol = {p.symbol: p for p in proposals}
        assert {"F_phi", "F_phiphi", "V_phi", "V_phiphi"} <= set(by_symbol)
        f_phi = by_symbol["F_phi"]
        assert f_phi.symbol_tex == r"F_{\phi}"
        assert f_phi.meaning_tex == r"\frac{\partial F}{\partial \phi}"
        assert f_phi.as_object().definition_tex == (
            r"F_{\phi} \equiv \frac{\partial F}{\partial \phi}"
        )

    def test_pure_does_not_mutate(self):
        npr = _npr()
        before = len(npr.objects)
        propose_definitions(npr)
        assert len(npr.objects) == before

    def test_constant_coupling_is_skipped(self):
        npr = _npr()
        for amb in npr.ambiguities:
            if amb.id == "amb-coupling-F":
                amb.resolution = "constant"
        symbols = {p.symbol for p in propose_definitions(npr)}
        assert "F_phi" not in symbols and "F_phiphi" not in symbols
        assert "V_phi" in symbols  # V left non-constant

    def test_already_declared_symbols_not_reproposed(self):
        session = Session(session_id="d1")
        session.ingest(_npr())
        first = propose_definitions(session.npr)
        assert any(p.symbol == "F_phi" for p in first)
        session.add_definition("F_phi", r"F_{\phi} \equiv \frac{\partial F}{\partial \phi}")
        again = {p.symbol for p in propose_definitions(session.npr)}
        assert "F_phi" not in again
        assert "F_phiphi" in again  # other proposals remain

    def test_no_functions_means_no_proposals(self):
        npr = _npr(r"R - 2\Lambda")
        assert propose_definitions(npr) == []


class TestSessionAddDefinition:
    def test_adds_shorthand_as_new_version(self):
        session = Session(session_id="d2")
        session.ingest(_npr())
        n_versions = len(session.npr_versions)
        n_objects = len(session.npr.objects)
        session.add_definition("F_phi", r"F_{\phi} \equiv \frac{\partial F}{\partial \phi}")
        assert len(session.npr_versions) == n_versions + 1
        assert len(session.npr.objects) == n_objects + 1
        added = session.npr.object_named("F_phi")
        assert added.kind == "shorthand"
        assert added.definition_tex.startswith("F_{\\phi}")
        # older version untouched
        assert all(o.name != "F_phi" for o in session.npr_versions[0].objects)

    def test_duplicate_symbol_rejected(self):
        session = Session(session_id="d3")
        session.ingest(_npr())
        with pytest.raises(ValueError, match="already declared"):
            session.add_definition("F", "anything")

    def test_adding_notation_keeps_gate_state(self):
        session = Session(session_id="d4")
        session.ingest(_npr())
        open_before = len(session.npr.unresolved_ambiguities())
        session.add_definition("F_phi", r"F_{\phi} \equiv \frac{\partial F}{\partial \phi}")
        assert len(session.npr.unresolved_ambiguities()) == open_before
