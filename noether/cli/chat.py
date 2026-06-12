"""Conversational front-end: ingest -> elicit -> plan in one terminal loop.

The human is the authority. Each question renders its numbered options plus a
free-form answer (a human may answer off-menu; that is recorded verbatim in
the ledger). `propose` asks the configured agent CLI for suggestions, which
are shown as pending defaults and take effect only when the human accepts
them one by one. Sessions persist through the same store as the HTTP and MCP
frontends, so `noether resume <id>` continues exactly where the loop stopped.

IO is injected (input_fn/out) so the loop is unit-testable without a TTY.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable
from typing import TextIO

from noether.llm.base import LLMAdapter, LLMError
from noether.npr.parse import ParseError
from noether.orchestrator.elicit import propose_resolutions
from noether.orchestrator.ingest import ingest_action
from noether.orchestrator.planner import AmbiguityBlocked, Plan
from noether.orchestrator.session import Session
from noether.orchestrator.store import SessionStore

DEFAULT_MEASURE = r"d^4x \sqrt{-g}"

HELP = (
    "  <number>   choose that option\n"
    "  <text>     free-form answer (recorded verbatim; you are the authority)\n"
    "  propose    ask the agent CLI to suggest answers (you still confirm)\n"
    "  skip       leave this question open and move on\n"
    "  quit       save and exit (resume later with: noether resume <id>)"
)


class ChatLoop:
    def __init__(
        self,
        store: SessionStore,
        llm: LLMAdapter | None = None,
        input_fn: Callable[[str], str] = input,
        out: TextIO = sys.stdout,
    ) -> None:
        self.store = store
        self.llm = llm
        self.input_fn = input_fn
        self.out = out
        self._pending: dict[str, str] = {}

    def _say(self, text: str = "") -> None:
        print(text, file=self.out)

    def _ask(self, prompt: str) -> str:
        try:
            return self.input_fn(prompt)
        except EOFError:
            return "quit"

    def _read_action(self) -> str:
        self._say("Paste the Lagrangian density (LaTeX). Finish with an empty line.")
        lines: list[str] = []
        while True:
            line = self._ask("... " if lines else "L = ")
            if line == "quit" and not lines:
                return ""
            if not line.strip():
                break
            lines.append(line)
        return "\n".join(lines)

    def start(self, measure: str = DEFAULT_MEASURE) -> int:
        text = self._read_action()
        if not text.strip():
            self._say("Nothing to ingest.")
            return 1
        try:
            result = ingest_action(measure, text)
        except ParseError as exc:
            self._say(f"Could not parse the action: {exc}")
            return 1
        session = Session(session_id=f"s-{uuid.uuid4().hex[:12]}")
        session.ingest(result.npr)
        self.store.save(session)
        self._say(f"\nSession {session.session_id}")
        self._describe(session)
        return self._run(session)

    def resume(self, session_id: str) -> int:
        try:
            session = self.store.get(session_id)
        except KeyError as exc:
            self._say(str(exc))
            return 1
        self._say(f"Resuming session {session.session_id} (state: {session.state.value})")
        self._describe(session)
        return self._run(session)

    def _describe(self, session: Session) -> None:
        npr = session.npr
        self._say(f"Action: \\int {npr.action.measure_tex} ( {npr.action.lagrangian_tex} )")
        if npr.objects:
            names = ", ".join(f"{o.name} ({o.kind})" for o in npr.objects)
            self._say(f"Objects: {names}")
        open_count = len(npr.unresolved_ambiguities())
        if open_count:
            self._say(f"{open_count} question(s) open. Commands:\n{HELP}")

    def _run(self, session: Session) -> int:
        while True:
            unresolved = session.npr.unresolved_ambiguities()
            if not unresolved:
                return self._finish(session)
            quit_requested, resolved_count = self._question_round(session, unresolved)
            if quit_requested:
                self._say(f"Saved. Resume with: noether resume {session.session_id}")
                return 0
            if resolved_count == 0:
                # a full round without an answer: stop rather than loop forever
                self._say(
                    "Questions remain open; planning would be a guess, so I stop here. "
                    f"Resume with: noether resume {session.session_id}"
                )
                return 0

    def _question_round(self, session: Session, unresolved) -> tuple[bool, int]:
        """One pass over the open questions: (human quit, answers recorded)."""
        resolved_count = 0
        for amb in unresolved:
            self._say(f"\n[{amb.id}] {amb.question}")
            for i, option in enumerate(amb.options, start=1):
                self._say(f"  {i}. {option}")
            if amb.id in self._pending:
                self._say(f"  (model proposes: {self._pending[amb.id]}; Enter accepts)")
            while True:
                answer = self._ask("> ").strip()
                if answer == "quit":
                    return True, resolved_count
                if answer == "skip":
                    break
                if answer == "propose":
                    self._propose(session)
                    if amb.id in self._pending:
                        self._say(f"  (model proposes: {self._pending[amb.id]}; Enter accepts)")
                    continue
                if not answer:
                    if amb.id in self._pending:
                        choice = self._pending.pop(amb.id)
                        session.resolve(amb.id, choice)
                        self.store.save(session)
                        self._say(f"  confirmed: {choice}")
                        resolved_count += 1
                        break
                    continue
                if answer.isdigit() and 1 <= int(answer) <= len(amb.options):
                    choice = amb.options[int(answer) - 1]
                else:
                    choice = answer  # free-form: the human is the authority
                self._pending.pop(amb.id, None)
                session.resolve(amb.id, choice)
                self.store.save(session)
                self._say(f"  recorded: {choice}")
                resolved_count += 1
                break
        return False, resolved_count

    def _propose(self, session: Session) -> None:
        if self.llm is None or not self.llm.available():
            self._say("  no agent CLI detected; answer directly instead")
            return
        try:
            proposal = propose_resolutions(session.npr, self.llm)
        except LLMError as exc:
            self._say(f"  elicitation failed: {exc}")
            return
        self._pending = {
            p.ambiguity_id: p.choice for p in proposal.proposals if p.choice is not None
        }
        self._say(
            f"  proposals from {proposal.llm_name} {proposal.llm_version} "
            "(unconfirmed until you accept each one)"
        )

    def _finish(self, session: Session) -> int:
        try:
            plan = session.plan()
        except AmbiguityBlocked as exc:
            self._say(f"Still blocked: {exc.questions}")
            return 1
        self.store.save(session)
        self._print_plan(plan)
        self._say(
            f"\nSession saved as {session.session_id}. "
            "Derivations for the supported tasks run via the eval commands "
            "(noether eval1 .. eval5, eval1s, eval3s) with full provenance."
        )
        return 0

    def _print_plan(self, plan: Plan) -> None:
        self._say(f"\nProblem is well posed. Plan ({plan.task_type}):")
        for i, step in enumerate(plan.steps, start=1):
            self._say(f"  {i}. [{step.capability.value}] {step.description}")
        if plan.verification:
            self._say(f"Verification: {', '.join(plan.verification)}")
