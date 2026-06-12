"""HTTP session API: the no-guessing contract enforced over the wire.

Skips cleanly when the [server] extra is not installed. No test reaches a
real LLM: /elicit is exercised with the in-process stub, and the
"no backend" path with a CLI adapter whose detection is forced empty.
"""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from noether.llm import CliLLMAdapter, StubLLMAdapter, stub_reply  # noqa: E402
from noether.orchestrator.ingest import ingest_action  # noqa: E402
from noether.orchestrator.store import SessionStore  # noqa: E402
from noether.server import create_app  # noqa: E402

MEASURE = r"d^4x \sqrt{-g}"


@pytest.fixture()
def store(tmp_path):
    return SessionStore(tmp_path / "sessions")


@pytest.fixture()
def client(store):
    return TestClient(create_app(store=store))


def _create(client, lagrangian="R"):
    response = client.post("/sessions", json={"lagrangian": lagrangian})
    assert response.status_code == 201, response.text
    return response.json()


class TestHealthAndCreate:
    def test_health_reports_kernels(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["kernels"]["sympy"]["available"] is True

    def test_create_session_ingests_and_blocks(self, client):
        body = _create(client)
        assert body["well_posed"] is False
        assert body["questions"] and all(q["resolution"] is None for q in body["questions"])
        assert body["state"] == "elicit"

    def test_parse_error_is_422(self, client):
        response = client.post("/sessions", json={"lagrangian": r"R_{\mu"})
        assert response.status_code == 422

    def test_unknown_session_is_404(self, client):
        assert client.get("/sessions/s-doesnotexist").status_code == 404


class TestResolveAndPlan:
    def test_plan_blocked_while_questions_open(self, client):
        body = _create(client)
        response = client.get(f"/sessions/{body['session_id']}/plan")
        assert response.status_code == 409
        assert response.json()["detail"]["questions"]

    def test_off_menu_resolution_rejected(self, client):
        body = _create(client)
        question = body["questions"][0]
        response = client.post(
            f"/sessions/{body['session_id']}/resolve",
            json={"resolutions": {question["id"]: "not-an-option"}},
        )
        assert response.status_code == 400

    def test_unknown_ambiguity_rejected(self, client):
        body = _create(client)
        response = client.post(
            f"/sessions/{body['session_id']}/resolve",
            json={"resolutions": {"amb-nope": "x"}},
        )
        assert response.status_code == 404

    def test_confirmed_resolutions_unblock_plan(self, client):
        body = _create(client)
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        response = client.post(
            f"/sessions/{body['session_id']}/resolve", json={"resolutions": resolutions}
        )
        assert response.status_code == 200
        assert response.json()["well_posed"] is True
        plan = client.get(f"/sessions/{body['session_id']}/plan")
        assert plan.status_code == 200
        assert plan.json()["task_type"] == "vary"

    def test_sessions_persist_across_app_instances(self, store):
        first = TestClient(create_app(store=store))
        body = _create(first)
        second = TestClient(create_app(store=store))
        reread = second.get(f"/sessions/{body['session_id']}")
        assert reread.status_code == 200
        assert reread.json()["questions"] == body["questions"]
        assert body["session_id"] in second.get("/sessions").json()["sessions"]


class TestElicit:
    def test_no_backend_is_503(self, store):
        offline = CliLLMAdapter(which=lambda _name: None)
        client = TestClient(create_app(store=store, llm=offline))
        body = _create(client)
        response = client.post(f"/sessions/{body['session_id']}/elicit")
        assert response.status_code == 503

    def test_stub_proposals_are_unconfirmed_and_do_not_mutate(self, store):
        npr = ingest_action(MEASURE, "R").npr
        answers = {a.id: a.options[0] for a in npr.ambiguities}
        client = TestClient(create_app(store=store, llm=StubLLMAdapter(stub_reply(answers))))
        body = _create(client)
        response = client.post(f"/sessions/{body['session_id']}/elicit")
        assert response.status_code == 200
        payload = response.json()
        assert payload["confirmed"] is False
        assert all(p["choice"] == answers[p["ambiguity_id"]] for p in payload["proposals"])
        # proposals must not have resolved anything server-side
        after = client.get(f"/sessions/{body['session_id']}").json()
        assert after["well_posed"] is False
        assert all(q["resolution"] is None for q in after["questions"])
