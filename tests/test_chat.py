"""Conversational loop: scripted IO, no TTY, no real LLM.

The loop must record only what the human typed (numbered option, free-form
text, or an explicitly accepted model proposal), persist after every answer,
and refuse to plan while questions stay open.
"""

import io

import pytest

from noether.cli.chat import ChatLoop
from noether.llm import CliLLMAdapter, StubLLMAdapter, stub_reply
from noether.orchestrator.elicit import apply_resolutions
from noether.orchestrator.ingest import ingest_action
from noether.orchestrator.store import SessionStore


class ScriptedInput:
    def __init__(self, lines):
        self.lines = list(lines)

    def __call__(self, _prompt: str) -> str:
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


def make_loop(tmp_path, lines, llm=None):
    store = SessionStore(tmp_path / "sessions")
    out = io.StringIO()
    loop = ChatLoop(store=store, llm=llm, input_fn=ScriptedInput(lines), out=out)
    return loop, store, out


def question_count():
    return len(ingest_action(r"d^4x \sqrt{-g}", "R").npr.ambiguities)


def numbered_answers_for(
    lagrangian: str, desired: dict[str, str] | None = None
) -> tuple[list[str], dict[str, int]]:
    desired = desired or {}
    npr = ingest_action(r"d^4x \sqrt{-g}", lagrangian).npr
    confirmations: dict[str, str] = {}
    lines: list[str] = [lagrangian, ""]
    answer_positions: dict[str, int] = {}
    for amb in npr.ambiguities:
        choice = desired.get(amb.id, amb.options[0])
        confirmations[amb.id] = choice
        answer_positions[amb.id] = len(lines)
        lines.append(str(amb.options.index(choice) + 1))
    confirmed = apply_resolutions(npr, confirmations)
    ricci = next(
        (amb for amb in confirmed.unresolved_ambiguities() if amb.id == "amb-ricci-contraction"),
        None,
    )
    if ricci is not None:
        choice = desired.get(ricci.id, ricci.options[0])
        answer_positions[ricci.id] = len(lines)
        lines.append(str(ricci.options.index(choice) + 1))
    return lines, answer_positions


class TestStart:
    def test_answer_all_by_number_reaches_plan(self, tmp_path):
        lines = ["R", ""] + ["1"] * question_count()
        loop, store, out = make_loop(tmp_path, lines)
        assert loop.start() == 0
        text = out.getvalue()
        assert "well posed" in text
        assert "Plan (vary)" in text
        (session_id,) = store.list_ids()
        assert store.get(session_id).npr.is_well_posed()

    def test_free_form_answer_recorded_verbatim(self, tmp_path):
        lines = ["R", "", "my own convention"] + ["1"] * (question_count() - 1)
        loop, store, _ = make_loop(tmp_path, lines)
        assert loop.start() == 0
        (session_id,) = store.list_ids()
        resolutions = {a.id: a.resolution for a in store.get(session_id).npr.ambiguities}
        assert "my own convention" in resolutions.values()

    def test_parse_error_reported(self, tmp_path):
        loop, store, out = make_loop(tmp_path, [r"R_{\mu", ""])
        assert loop.start() == 1
        assert "Could not parse" in out.getvalue()
        assert store.list_ids() == []

    def test_skipping_everything_stops_without_plan(self, tmp_path):
        lines = ["R", ""] + ["skip"] * question_count()
        loop, store, out = make_loop(tmp_path, lines)
        assert loop.start() == 0
        assert "planning would be a guess" in out.getvalue()
        (session_id,) = store.list_ids()
        assert not store.get(session_id).npr.is_well_posed()

    def test_palatini_numbered_answers_reach_independent_connection_plan(self, tmp_path):
        lagrangian = r"g^{\mu\nu} R_{\mu\nu}(\Gamma)"
        lines, _ = numbered_answers_for(
            lagrangian,
            {
                "amb-connection": "independent",
                "amb-torsion": "torsion-allowed",
                "amb-nonmetricity": "nonmetricity-allowed",
                "amb-metric-compatibility": "not-metric-compatible",
                "amb-curvature-free": "curvature-allowed",
                "amb-ricci-contraction": "first-fourth",
            },
        )
        loop, store, out = make_loop(tmp_path, lines)

        assert loop.start() == 0

        text = out.getvalue()
        assert "[amb-connection]" in text
        assert "[amb-torsion]" in text
        assert "[amb-nonmetricity]" in text
        assert "[amb-metric-compatibility]" in text
        assert "1. independent" in text
        assert "[independent-connection]" in text

        (session_id,) = store.list_ids()
        session = store.get(session_id)
        assert session.npr.geometry.connection.type == "independent"
        assert session.npr.geometry.connection.torsion is True
        assert session.npr.geometry.connection.nonmetricity is True
        assert session.npr.geometry.connection.metric_compatible is False

    def test_off_menu_geometry_answer_is_rejected_before_numbered_retry(self, tmp_path):
        lagrangian = r"g^{\mu\nu} R_{\mu\nu}(\Gamma)"
        lines, positions = numbered_answers_for(
            lagrangian,
            {
                "amb-connection": "independent",
                "amb-torsion": "torsion-allowed",
                "amb-nonmetricity": "nonmetricity-free",
                "amb-metric-compatibility": "metric-compatible",
                "amb-curvature-free": "curvature-allowed",
            },
        )
        connection_index = positions["amb-connection"]
        lines = lines[:connection_index] + ["metric-affine"] + lines[connection_index:]

        loop, store, out = make_loop(tmp_path, lines)
        assert loop.start() == 0

        text = out.getvalue()
        assert "not a listed option" in text
        (session_id,) = store.list_ids()
        assert store.get(session_id).npr.geometry.connection.type == "independent"


class TestResume:
    def test_quit_then_resume_completes(self, tmp_path):
        loop, store, _ = make_loop(tmp_path, ["R", "", "quit"])
        assert loop.start() == 0
        (session_id,) = store.list_ids()
        assert not store.get(session_id).npr.is_well_posed()

        lines = ["1"] * question_count()
        resumed, _, out2 = make_loop(tmp_path, lines)
        resumed.store = store
        assert resumed.resume(session_id) == 0
        assert "Plan (vary)" in out2.getvalue()
        assert store.get(session_id).npr.is_well_posed()

    def test_resume_unknown_session(self, tmp_path):
        loop, _, out = make_loop(tmp_path, [])
        assert loop.resume("s-doesnotexist") == 1
        assert "no session" in out.getvalue()


class TestPropose:
    @pytest.fixture()
    def answers(self):
        npr = ingest_action(r"d^4x \sqrt{-g}", "R").npr
        return {a.id: a.options[0] for a in npr.ambiguities}

    def test_proposals_need_explicit_acceptance(self, tmp_path, answers):
        # "propose" then Enter accepts each pending proposal, one per question
        lines = ["R", "", "propose"] + [""] * question_count()
        llm = StubLLMAdapter(stub_reply(answers))
        loop, store, out = make_loop(tmp_path, lines, llm=llm)
        assert loop.start() == 0
        assert "unconfirmed until you accept" in out.getvalue()
        assert "Plan (vary)" in out.getvalue()
        (session_id,) = store.list_ids()
        resolutions = {a.id: a.resolution for a in store.get(session_id).npr.ambiguities}
        assert resolutions == answers

    def test_enter_without_proposal_does_not_resolve(self, tmp_path):
        lines = ["R", "", "", "quit"]
        loop, store, _ = make_loop(tmp_path, lines)
        assert loop.start() == 0
        (session_id,) = store.list_ids()
        assert not store.get(session_id).npr.is_well_posed()

    def test_propose_without_backend_says_so(self, tmp_path):
        offline = CliLLMAdapter(which=lambda _name: None)
        lines = ["R", "", "propose", "quit"]
        loop, _, out = make_loop(tmp_path, lines, llm=offline)
        assert loop.start() == 0
        assert "no agent CLI detected" in out.getvalue()


class TestRationaleDisplay:
    """VAL-GUIDE-019: the proposal's rationale is surfaced alongside the
    on-menu choice during chat elicitation."""

    @pytest.fixture()
    def answers_with_rationale(self):
        npr = ingest_action(r"d^4x \sqrt{-g}", "R").npr
        return {a.id: a.options[0] for a in npr.ambiguities}

    def test_propose_shows_rationale(self, tmp_path, answers_with_rationale):
        """After 'propose', each pending choice shows its rationale."""
        lines = ["R", "", "propose"] + [""] * question_count()
        llm = StubLLMAdapter(stub_reply(answers_with_rationale))
        loop, _, out = make_loop(tmp_path, lines, llm=llm)
        assert loop.start() == 0
        text = out.getvalue()
        # The rationale from the stub reply should appear in the output
        assert "rationale" in text, "rationale should be displayed after propose"

    def test_rationale_shown_alongside_choice(self, tmp_path, answers_with_rationale):
        """The rationale is displayed with the proposed choice."""
        lines = ["R", "", "propose"] + [""] * question_count()
        llm = StubLLMAdapter(stub_reply(answers_with_rationale))
        loop, _, out = make_loop(tmp_path, lines, llm=llm)
        assert loop.start() == 0
        text = out.getvalue()
        # Should show "model proposes: <choice>; rationale: <text>"
        assert "model proposes:" in text
