"""Session state machine and the ambiguity gate."""

import pytest

from evals.eval1_eh_trace import ELICITATION_ANSWERS, build_npr
from noether.kernels.base import Capability
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.orchestrator.session import Session, SessionState


class TestAmbiguityGate:
    def test_planning_blocked_while_ambiguous(self):
        npr = build_npr(resolved=False)
        with pytest.raises(AmbiguityBlocked) as exc:
            build_plan(npr)
        assert len(exc.value.questions) == 3

    def test_planning_allowed_when_resolved(self):
        plan = build_plan(build_npr(resolved=True))
        caps = [step.capability for step in plan.steps]
        assert Capability.VARY in caps
        assert Capability.IBP in caps
        assert Capability.CANONICALIZE in caps
        assert "divergence-free" in plan.verification


class TestSession:
    def test_elicit_then_plan(self):
        session = Session(session_id="t1")
        session.ingest(build_npr(resolved=False))
        assert session.state is SessionState.ELICIT
        for amb in list(session.npr.unresolved_ambiguities()):
            session.resolve(amb.id, ELICITATION_ANSWERS[amb.id])
        plan = session.plan()
        assert session.state is SessionState.PLAN
        assert plan.task_type == "vary"

    def test_npr_versions_immutable(self):
        session = Session(session_id="t2")
        session.ingest(build_npr(resolved=False))
        n_before = len(session.npr_versions)
        session.resolve("amb-conventions", "noether-default-v1")
        assert len(session.npr_versions) == n_before + 1
        # the older version still has the ambiguity open
        old = session.npr_versions[0]
        assert any(a.id == "amb-conventions" and not a.resolved for a in old.ambiguities)

    def test_assumption_change_marks_results_stale(self):
        session = Session(session_id="t3")
        session.ingest(build_npr(resolved=True))
        session.record_result("eom-1")
        session.change_assumption("amb-conventions", "custom")
        assert "eom-1" in session.stale_result_ids
        assert session.state is SessionState.ELICIT


class TestResolutionPropagation:
    """A confirmed vary-wrt answer must reach task.with_respect_to; the
    ledger entry alone is not what the planner reads."""

    @staticmethod
    def _ingested_session() -> Session:
        from noether.orchestrator.ingest import ingest_action

        npr = ingest_action(
            r"d^4x \sqrt{-g}",
            r"F(\phi) R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)",
        ).npr
        session = Session(session_id="prop")
        session.ingest(npr)
        return session

    def test_choice_narrows_task(self):
        session = self._ingested_session()
        assert session.npr.task.with_respect_to == ["g", "phi"]
        session.resolve("amb-vary-wrt", "g")
        assert session.npr.task.with_respect_to == ["g"]
        # earlier NPR version stays untouched
        assert session.npr_versions[0].task.with_respect_to == ["g", "phi"]

    def test_eval_style_answer_parses(self):
        session = self._ingested_session()
        session.resolve("amb-vary-wrt", "g and phi")
        assert session.npr.task.with_respect_to == ["g", "phi"]

    def test_free_form_without_declared_fields_changes_nothing(self):
        session = self._ingested_session()
        session.resolve("amb-vary-wrt", "everything dynamical please")
        assert session.npr.task.with_respect_to == ["g", "phi"]

    def test_plan_reflects_choice(self):
        session = self._ingested_session()
        for amb in list(session.npr.unresolved_ambiguities()):
            session.resolve(amb.id, amb.options[0] if amb.id != "amb-vary-wrt" else "phi")
        plan = session.plan()
        vary_steps = [s for s in plan.steps if s.payload.get("with_respect_to")]
        assert vary_steps and vary_steps[0].payload["with_respect_to"] == ["phi"]
