"""Conversational loop: scripted IO, no TTY, no real LLM.

The loop must record only what the human typed (numbered option, free-form
text, or an explicitly accepted model proposal), persist after every answer,
and refuse to plan while questions stay open.
"""

import io

import pytest

from noether.cli.chat import ChatLoop
from noether.llm import CliLLMAdapter, StubLLMAdapter, stub_reply
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
