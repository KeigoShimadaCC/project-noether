"""MCP server over the orchestrator session surface.

The host LLM converses and plans; Noether does the kernel-backed part. The
no-guessing contract survives the protocol: `noether_plan` answers with the
open question list (blocked=true) until every ambiguity is resolved through
`noether_resolve` with an on-menu choice. A host LLM cannot make Noether
guess, and tool results carry refusals as data rather than exceptions so the
host can relay the questions to its human.

Tool logic lives in NoetherTools (plain methods, unit-testable without the
MCP runtime); create_mcp_server wraps it for stdio transport.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.llm.base import LLMAdapter, LLMError
from noether.npr.parse import ParseError
from noether.orchestrator.derive import derive_eom, derive_perturbation
from noether.orchestrator.ingest import ingest_action
from noether.orchestrator.planner import AmbiguityBlocked
from noether.orchestrator.store import DEFAULT_STORE, SessionStore
from noether.orchestrator.view import session_payload

DEFAULT_MEASURE = r"d^4x \sqrt{-g}"


class NoetherTools:
    """Session tools shared by every MCP host. One store, same sessions as
    the HTTP API and the CLI."""

    def __init__(
        self,
        store: SessionStore | None = None,
        llm: LLMAdapter | None = None,
        results_root: Path | None = None,
    ) -> None:
        self.store = store if store is not None else SessionStore(DEFAULT_STORE)
        self.llm = llm
        self.results_root = (
            results_root if results_root is not None else self.store.root.parent / "results"
        )

    def _llm(self) -> LLMAdapter:
        if self.llm is not None:
            return self.llm
        from noether.llm.cli import CliLLMAdapter

        return CliLLMAdapter()

    def kernels(self) -> dict[str, Any]:
        adapters = [SympyKernelAdapter(), CadabraAdapter()]
        return {
            a.name: {
                "available": a.available(),
                "version": a.version() if a.available() else None,
            }
            for a in adapters
        }

    def ingest(self, lagrangian: str, measure: str = DEFAULT_MEASURE) -> dict[str, Any]:
        import uuid

        from noether.orchestrator.session import Session

        try:
            result = ingest_action(measure, lagrangian)
        except ParseError as exc:
            return {"error": f"parse error: {exc}"}
        session = Session(session_id=f"s-{uuid.uuid4().hex[:12]}")
        session.ingest(result.npr)
        self.store.save(session)
        return session_payload(session)

    def sessions(self) -> dict[str, Any]:
        return {"sessions": self.store.list_ids()}

    def session(self, session_id: str) -> dict[str, Any]:
        try:
            return session_payload(self.store.get(session_id))
        except KeyError as exc:
            return {"error": str(exc)}

    def resolve(self, session_id: str, resolutions: dict[str, str]) -> dict[str, Any]:
        try:
            sess = self.store.get(session_id)
        except KeyError as exc:
            return {"error": str(exc)}
        if not resolutions:
            return {"error": "resolutions must not be empty"}
        by_id = {a.id: a for a in sess.npr.ambiguities}
        for amb_id, choice in resolutions.items():
            if amb_id not in by_id:
                return {"error": f"no ambiguity {amb_id!r}"}
            options = by_id[amb_id].options
            if options and choice not in options:
                return {
                    "error": (
                        f"{choice!r} is not a listed option for {amb_id!r}; options: {options}"
                    )
                }
        for amb_id, choice in resolutions.items():
            sess.resolve(amb_id, choice)
        self.store.save(sess)
        return session_payload(sess)

    def propose_definitions(self, session_id: str) -> dict[str, Any]:
        from noether.orchestrator.definitions import propose_definitions

        try:
            sess = self.store.get(session_id)
        except KeyError as exc:
            return {"error": str(exc)}
        return {
            "confirmed": False,
            "proposals": [
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "symbol_tex": p.symbol_tex,
                    "meaning_tex": p.meaning_tex,
                    "definition_tex": p.as_object().definition_tex,
                    "rationale": p.rationale,
                }
                for p in propose_definitions(sess.npr)
            ],
        }

    def adopt_definitions(self, session_id: str, accept: list[str]) -> dict[str, Any]:
        from noether.orchestrator.definitions import propose_definitions

        try:
            sess = self.store.get(session_id)
        except KeyError as exc:
            return {"error": str(exc)}
        if not accept:
            return {"error": "accept must not be empty"}
        by_id = {p.id: p for p in propose_definitions(sess.npr)}
        for def_id in accept:
            if def_id not in by_id:
                return {"error": f"no proposed definition {def_id!r}"}
        for def_id in accept:
            proposal = by_id[def_id]
            sess.add_definition(proposal.symbol, proposal.as_object().definition_tex)
        self.store.save(sess)
        return session_payload(sess)

    def plan(self, session_id: str) -> dict[str, Any]:
        try:
            sess = self.store.get(session_id)
        except KeyError as exc:
            return {"error": str(exc)}
        try:
            built = sess.plan()
        except AmbiguityBlocked as exc:
            return {"blocked": True, "questions": exc.questions}
        self.store.save(sess)
        return {
            "blocked": False,
            "task_type": built.task_type,
            "steps": [
                {"capability": s.capability.value, "description": s.description}
                for s in built.steps
            ],
            "verification": built.verification,
        }

    def derive(
        self,
        session_id: str,
        with_respect_to: list[str] | None = None,
        kind: str = "eom",
    ) -> dict[str, Any]:
        if kind not in ("eom", "perturbation"):
            return {"error": f"unknown derivation kind {kind!r}"}
        try:
            sess = self.store.get(session_id)
        except KeyError as exc:
            return {"error": str(exc)}
        cadabra = CadabraAdapter()
        if not cadabra.available():
            return {"error": "cadabra kernel not installed; cannot derive"}
        llm = self._llm()
        if not llm.available():
            return {"error": "no agent CLI / LLM backend available to parameterize the script"}
        npr = sess.npr
        if with_respect_to:
            declared = {o.name for o in npr.objects}
            for wrt in with_respect_to:
                if wrt not in declared:
                    return {"error": f"{wrt!r} is not a declared object"}
            if kind == "eom":
                npr = npr.model_copy(deep=True)
                npr.task.with_respect_to = with_respect_to
        adapters = {"cadabra": cadabra, "sympy": SympyKernelAdapter()}
        try:
            if kind == "perturbation":
                results = derive_perturbation(
                    npr,
                    llm,
                    adapters,
                    fields=with_respect_to,
                    session_id=session_id,
                    results_root=self.results_root,
                )
            else:
                results = derive_eom(
                    npr,
                    llm,
                    adapters,
                    session_id=session_id,
                    results_root=self.results_root,
                )
        except AmbiguityBlocked as exc:
            return {"blocked": True, "questions": exc.questions}
        except (LLMError, NotImplementedError) as exc:
            return {"error": str(exc)}
        return {
            "session_id": session_id,
            "derivations": [r.model_dump() for r in results],
        }


def create_mcp_server(store: SessionStore | None = None, llm: LLMAdapter | None = None):
    from mcp.server.fastmcp import FastMCP

    tools = NoetherTools(store, llm=llm)
    server = FastMCP(
        "noether",
        instructions=(
            "Symbolic-physics collaborator. Ingest a LaTeX action with "
            "noether_ingest, show the returned questions to your human, "
            "confirm answers with noether_resolve (on-menu choices only), "
            "then noether_plan. While questions are open, plan returns "
            "blocked=true with the question list; relay it, never guess. "
            "Once well posed, noether_derive runs a kernel-checked Cadabra "
            "derivation; it marks each equation verified only if the kernel "
            "confirmed it, and reports unverified results as such."
        ),
    )

    @server.tool()
    def noether_kernels() -> dict[str, Any]:
        """List computer-algebra kernels and their availability."""
        return tools.kernels()

    @server.tool()
    def noether_ingest(lagrangian: str, measure: str = DEFAULT_MEASURE) -> dict[str, Any]:
        """Parse a LaTeX scalar Lagrangian density into a new session.
        Returns the session id and the open clarifying questions, each with
        its listed options. The questions are for the human, not for you."""
        return tools.ingest(lagrangian, measure)

    @server.tool()
    def noether_sessions() -> dict[str, Any]:
        """List stored session ids."""
        return tools.sessions()

    @server.tool()
    def noether_session(session_id: str) -> dict[str, Any]:
        """Show a session: action, objects, questions, resolution state."""
        return tools.session(session_id)

    @server.tool()
    def noether_resolve(session_id: str, resolutions: dict[str, str]) -> dict[str, Any]:
        """Record human-confirmed answers, mapping ambiguity id to one of its
        listed options. Off-menu choices are rejected. Only call this with
        answers the human actually confirmed."""
        return tools.resolve(session_id, resolutions)

    @server.tool()
    def noether_propose_definitions(session_id: str) -> dict[str, Any]:
        """Propose readability shorthands for the derivatives of function
        couplings (e.g. F_phi for partial F / partial phi). These are
        notation, not results; the human chooses which to adopt."""
        return tools.propose_definitions(session_id)

    @server.tool()
    def noether_adopt_definitions(session_id: str, accept: list[str]) -> dict[str, Any]:
        """Adopt human-confirmed notation by proposal id (from
        noether_propose_definitions). Adds shorthands; never reopens
        questions."""
        return tools.adopt_definitions(session_id, accept)

    @server.tool()
    def noether_plan(session_id: str) -> dict[str, Any]:
        """Plan the derivation. If questions remain open this returns
        blocked=true with the question list; relay them to the human."""
        return tools.plan(session_id)

    @server.tool()
    def noether_derive(
        session_id: str,
        with_respect_to: list[str] | None = None,
        kind: str = "eom",
    ) -> dict[str, Any]:
        """Derive a result for a well-posed session. kind="eom" varies the
        action for the equations of motion; kind="perturbation" expands it to
        quadratic order around a background (scalar fields only today). Noether
        parameterizes a Cadabra script, runs it in the kernel, and marks each
        result verified only when the kernel's own residue check confirms it;
        unverified results are returned as such, never as truth. Optionally
        restrict to specific declared fields with with_respect_to. If questions
        remain open this returns blocked=true."""
        return tools.derive(session_id, with_respect_to, kind)

    return server
