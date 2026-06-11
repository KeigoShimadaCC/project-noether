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
