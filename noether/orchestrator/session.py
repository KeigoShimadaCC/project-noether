"""Session state machine: INGEST -> ELICIT -> PLAN -> COMPUTE -> VERIFY -> PRESENT.

Any assumption change moves the machine back to ELICIT and marks dependent
results stale. NPR versions are immutable; results reference the version they
were computed against.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from noether.npr.schema import NPR
from noether.orchestrator.planner import Plan, build_plan


class SessionState(StrEnum):
    INGEST = "ingest"
    ELICIT = "elicit"
    PLAN = "plan"
    COMPUTE = "compute"
    VERIFY = "verify"
    PRESENT = "present"


class SessionEvent(BaseModel):
    state: SessionState
    detail: str


class Session(BaseModel):
    session_id: str
    npr_versions: list[NPR] = Field(default_factory=list)
    events: list[SessionEvent] = Field(default_factory=list)
    stale_result_ids: list[str] = Field(default_factory=list)
    result_ids: list[str] = Field(default_factory=list)

    @property
    def state(self) -> SessionState:
        return self.events[-1].state if self.events else SessionState.INGEST

    @property
    def npr(self) -> NPR:
        if not self.npr_versions:
            raise RuntimeError("session has no NPR yet; ingest first")
        return self.npr_versions[-1]

    def _log(self, state: SessionState, detail: str) -> None:
        self.events.append(SessionEvent(state=state, detail=detail))

    def ingest(self, npr: NPR) -> None:
        self.npr_versions.append(npr)
        self._log(SessionState.INGEST, "action ingested")
        if npr.unresolved_ambiguities():
            self._log(
                SessionState.ELICIT,
                f"{len(npr.unresolved_ambiguities())} ambiguities to resolve",
            )

    def resolve(self, ambiguity_id: str, resolution: str) -> None:
        """Resolving an ambiguity produces a new immutable NPR version."""
        from noether.orchestrator.resolutions import propagate_resolution

        current = self.npr
        updated = current.model_copy(deep=True)
        for amb in updated.ambiguities:
            if amb.id == ambiguity_id:
                amb.resolution = resolution
                propagate_resolution(updated, amb)
                break
        else:
            raise KeyError(f"no ambiguity with id {ambiguity_id!r}")
        self.npr_versions.append(updated)
        self._log(SessionState.ELICIT, f"resolved {ambiguity_id}: {resolution}")

    def confirm_resolutions(self, confirmations: dict[str, str]) -> None:
        """Apply human-confirmed on-menu answers as a new immutable NPR version.

        This funnels menu-based surface answers through the same validation path
        as the HTTP and CLI elicit surfaces: every confirmation must target an
        existing ambiguity and use one of its listed options.
        """
        from noether.orchestrator.elicit import apply_resolutions

        current = self.npr
        known = {amb.id for amb in current.ambiguities}
        missing = [amb_id for amb_id in confirmations if amb_id not in known]
        if missing:
            raise KeyError(f"no ambiguity with id {missing[0]!r}")

        updated = apply_resolutions(current, confirmations)
        self.npr_versions.append(updated)
        for ambiguity_id, resolution in confirmations.items():
            self._log(SessionState.ELICIT, f"resolved {ambiguity_id}: {resolution}")

    def add_definition(self, symbol: str, definition_tex: str) -> None:
        """Adopt a human-confirmed readability shorthand as a new immutable
        NPR version. A shorthand only names an expression; it adds notation,
        not physics, and never reopens the ambiguity gate."""
        from noether.npr.schema import ObjectDecl

        current = self.npr
        if any(obj.name == symbol for obj in current.objects):
            raise ValueError(f"object {symbol!r} already declared")
        updated = current.model_copy(deep=True)
        updated.objects.append(
            ObjectDecl(
                name=symbol,
                kind="shorthand",
                role="shorthand",
                rank=0,
                definition_tex=definition_tex,
            )
        )
        self.npr_versions.append(updated)
        self._log(SessionState.ELICIT, f"adopted notation {symbol}")

    def change_assumption(self, ambiguity_id: str, new_resolution: str) -> None:
        """Mid-session change: new NPR version, downstream results go stale."""
        self.resolve(ambiguity_id, new_resolution)
        self.stale_result_ids.extend(self.result_ids)
        self._log(
            SessionState.ELICIT,
            f"assumption changed ({ambiguity_id}); {len(self.result_ids)} result(s) marked stale",
        )

    def plan(self) -> Plan:
        plan = build_plan(self.npr)  # raises AmbiguityBlocked if not well-posed
        self._log(SessionState.PLAN, f"planned {plan.task_type}: {len(plan.steps)} steps")
        return plan

    def record_result(self, result_id: str) -> None:
        self.result_ids.append(result_id)
        self._log(SessionState.PRESENT, f"result {result_id} presented")

    def mark_results_stale(self, reason: str) -> None:
        """Flag every recorded result as stale after an assumption change.

        Used when a resolution lands after results already exist: the results
        still hold for the NPR version they were computed against, but no
        longer for the current one, so they are surfaced as stale rather than
        silently dropped or trusted.
        """
        newly = [rid for rid in self.result_ids if rid not in self.stale_result_ids]
        if not newly:
            return
        self.stale_result_ids.extend(newly)
        self._log(SessionState.ELICIT, f"{len(newly)} result(s) marked stale: {reason}")
