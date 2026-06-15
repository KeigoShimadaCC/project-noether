"""Frontend-neutral session view: one dict shape for HTTP, MCP, and CLI.

Presentation only; never mutates the session and never invents physics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noether.orchestrator.session import Session
from noether.provenance.bundle import read_results


def session_payload(session: Session) -> dict[str, Any]:
    npr = session.npr
    return {
        "session_id": session.session_id,
        "state": session.state.value,
        "well_posed": npr.is_well_posed(),
        "action": {
            "measure_tex": npr.action.measure_tex,
            "lagrangian_tex": npr.action.lagrangian_tex,
        },
        "objects": [
            {
                "name": o.name,
                "kind": o.kind,
                "role": o.role,
                "definition_tex": o.definition_tex,
            }
            for o in npr.objects
        ],
        "questions": [
            {
                "id": a.id,
                "question": a.question,
                "kind": a.kind,
                "options": a.options,
                "resolution": a.resolution,
            }
            for a in npr.ambiguities
        ],
        "events": [{"state": e.state.value, "detail": e.detail} for e in session.events],
    }


def results_payload(session: Session, results_root: Path) -> dict[str, Any]:
    """The session's recorded derivations, reloaded from their provenance
    bundles. Presentation only: it reads what the kernel already produced and
    flags which results an assumption change has since made stale."""
    return {
        "session_id": session.session_id,
        "results": read_results(Path(results_root), session.session_id, session.result_ids),
        "stale_result_ids": list(session.stale_result_ids),
    }
