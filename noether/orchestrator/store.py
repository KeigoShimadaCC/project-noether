"""Session persistence: one JSON file per session, atomically replaced.

The Session model is the single source of truth (pydantic, versioned NPRs
inside); this store only serializes it. Used by the API server and by the
conversational CLI for session resume. Physics state never lives client-side.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from noether.orchestrator.session import Session

DEFAULT_STORE = Path.home() / ".noether" / "sessions"


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        if not session_id.replace("-", "").isalnum():
            raise KeyError(f"invalid session id {session_id!r}")
        return self.root / f"{session_id}.json"

    def save(self, session: Session) -> Path:
        path = self._path(session.session_id)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(session.model_dump_json(indent=2))
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return path

    def get(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            raise KeyError(f"no session {session_id!r}")
        return Session.model_validate_json(path.read_text())

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))
