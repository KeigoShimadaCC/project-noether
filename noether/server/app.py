"""FastAPI session server over the orchestrator loop.

INGEST (POST /sessions) -> ELICIT (POST /sessions/{id}/elicit proposes,
POST /sessions/{id}/resolve confirms) -> PLAN (GET /sessions/{id}/plan).

The no-guessing contract is enforced server-side exactly as in the library:
/elicit returns UNCONFIRMED model proposals (off-menu suggestions already
discarded); only /resolve, carrying human-confirmed choices validated against
the listed options, mutates the session. Planning while questions remain open
returns 409 with the questions, never a guess.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.llm.base import LLMAdapter, LLMError
from noether.npr.parse import ParseError
from noether.orchestrator.definitions import propose_definitions
from noether.orchestrator.elicit import propose_resolutions
from noether.orchestrator.ingest import ingest_action
from noether.orchestrator.planner import AmbiguityBlocked
from noether.orchestrator.session import Session
from noether.orchestrator.store import DEFAULT_STORE, SessionStore
from noether.orchestrator.view import session_payload as _session_payload

DEFAULT_MEASURE = r"d^4x \sqrt{-g}"


class CreateSessionRequest(BaseModel):
    lagrangian: str
    measure: str = DEFAULT_MEASURE


class ResolveRequest(BaseModel):
    resolutions: dict[str, str] = Field(min_length=1)


class AdoptDefinitionsRequest(BaseModel):
    accept: list[str] = Field(min_length=1)


def create_app(
    store: SessionStore | None = None,
    llm: LLMAdapter | None = None,
) -> FastAPI:
    """Build the app. `llm=None` defers to auto-detecting an agent CLI at
    request time; tests inject a stub instead."""
    app = FastAPI(title="noether", version="0.1.0")
    app.state.store = store if store is not None else SessionStore(DEFAULT_STORE)
    app.state.llm = llm

    def _get_session(session_id: str) -> Session:
        try:
            return app.state.store.get(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _llm() -> LLMAdapter:
        if app.state.llm is not None:
            return app.state.llm
        from noether.llm.cli import CliLLMAdapter

        return CliLLMAdapter()

    @app.get("/health")
    def health() -> dict[str, Any]:
        adapters = [SympyKernelAdapter(), CadabraAdapter()]
        return {
            "status": "ok",
            "kernels": {
                a.name: {
                    "available": a.available(),
                    "version": a.version() if a.available() else None,
                }
                for a in adapters
            },
        }

    @app.get("/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"sessions": app.state.store.list_ids()}

    @app.post("/sessions", status_code=201)
    def create_session(body: CreateSessionRequest) -> dict[str, Any]:
        try:
            result = ingest_action(body.measure, body.lagrangian)
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=f"parse error: {exc}") from exc
        session = Session(session_id=f"s-{uuid.uuid4().hex[:12]}")
        session.ingest(result.npr)
        app.state.store.save(session)
        return _session_payload(session)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return _session_payload(_get_session(session_id))

    @app.post("/sessions/{session_id}/elicit")
    def elicit(session_id: str) -> dict[str, Any]:
        session = _get_session(session_id)
        adapter = _llm()
        if not adapter.available():
            raise HTTPException(
                status_code=503,
                detail="no LLM backend available (no agent CLI detected)",
            )
        try:
            proposal = propose_resolutions(session.npr, adapter)
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "confirmed": False,
            "note": ("model proposals only; confirm through POST /sessions/{id}/resolve"),
            "llm": {"name": proposal.llm_name, "version": proposal.llm_version},
            "proposals": [
                {
                    "ambiguity_id": p.ambiguity_id,
                    "choice": p.choice,
                    "rationale": p.rationale,
                }
                for p in proposal.proposals
            ],
        }

    @app.post("/sessions/{session_id}/resolve")
    def resolve(session_id: str, body: ResolveRequest) -> dict[str, Any]:
        session = _get_session(session_id)
        by_id = {a.id: a for a in session.npr.ambiguities}
        for amb_id, choice in body.resolutions.items():
            if amb_id not in by_id:
                raise HTTPException(status_code=404, detail=f"no ambiguity {amb_id!r}")
            options = by_id[amb_id].options
            if options and choice not in options:
                raise HTTPException(
                    status_code=400,
                    detail=f"{choice!r} is not a listed option for {amb_id!r}: {options}",
                )
        for amb_id, choice in body.resolutions.items():
            session.resolve(amb_id, choice)
        app.state.store.save(session)
        return _session_payload(session)

    @app.get("/sessions/{session_id}/definitions")
    def definitions(session_id: str) -> dict[str, Any]:
        session = _get_session(session_id)
        proposals = propose_definitions(session.npr)
        return {
            "confirmed": False,
            "note": (
                "readability notation only; these are definitions, not results. "
                "Adopt the ones you want through POST /sessions/{id}/definitions"
            ),
            "proposals": [
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "symbol_tex": p.symbol_tex,
                    "meaning_tex": p.meaning_tex,
                    "definition_tex": p.as_object().definition_tex,
                    "rationale": p.rationale,
                }
                for p in proposals
            ],
        }

    @app.post("/sessions/{session_id}/definitions")
    def adopt_definitions(session_id: str, body: AdoptDefinitionsRequest) -> dict[str, Any]:
        session = _get_session(session_id)
        by_id = {p.id: p for p in propose_definitions(session.npr)}
        for def_id in body.accept:
            if def_id not in by_id:
                raise HTTPException(status_code=404, detail=f"no proposed definition {def_id!r}")
        for def_id in body.accept:
            proposal = by_id[def_id]
            session.add_definition(proposal.symbol, proposal.as_object().definition_tex)
        app.state.store.save(session)
        return _session_payload(session)

    @app.get("/sessions/{session_id}/plan")
    def plan(session_id: str) -> dict[str, Any]:
        session = _get_session(session_id)
        try:
            built = session.plan()
        except AmbiguityBlocked as exc:
            raise HTTPException(
                status_code=409,
                detail={"blocked": True, "questions": exc.questions},
            ) from exc
        app.state.store.save(session)
        return {
            "task_type": built.task_type,
            "steps": [
                {"capability": s.capability.value, "description": s.description}
                for s in built.steps
            ],
            "verification": built.verification,
        }

    return app
