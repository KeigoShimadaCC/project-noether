"""HTTP session API: the no-guessing contract enforced over the wire.

Skips cleanly when the [server] extra is not installed. No test reaches a
real LLM: /elicit is exercised with the in-process stub, and the
"no backend" path with a CLI adapter whose detection is forced empty.
"""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from noether.kernels.cadabra import CadabraAdapter, templates  # noqa: E402
from noether.llm import CliLLMAdapter, StubLLMAdapter, stub_reply  # noqa: E402
from noether.orchestrator.ingest import ingest_action  # noqa: E402
from noether.orchestrator.store import SessionStore  # noqa: E402
from noether.server import create_app  # noqa: E402

MEASURE = r"d^4x \sqrt{-g}"

requires_cadabra = pytest.mark.skipif(
    not CadabraAdapter().available(), reason="cadabra2 not installed"
)


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


SCALAR_TENSOR = r"F(\phi) R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)"


class TestDefinitions:
    def test_proposals_are_notation_not_results(self, client):
        body = _create(client, SCALAR_TENSOR)
        payload = client.get(f"/sessions/{body['session_id']}/definitions").json()
        assert payload["confirmed"] is False
        symbols = {p["symbol"] for p in payload["proposals"]}
        assert {"F_phi", "F_phiphi", "V_phi"} <= symbols
        f_phi = next(p for p in payload["proposals"] if p["symbol"] == "F_phi")
        assert f_phi["meaning_tex"] == r"\frac{\partial F}{\partial \phi}"

    def test_no_proposals_without_function_coupling(self, client):
        body = _create(client, "R")
        payload = client.get(f"/sessions/{body['session_id']}/definitions").json()
        assert payload["proposals"] == []

    def test_adopt_adds_shorthand_object(self, client):
        body = _create(client, SCALAR_TENSOR)
        sid = body["session_id"]
        response = client.post(f"/sessions/{sid}/definitions", json={"accept": ["def-F-phi"]})
        assert response.status_code == 200
        objects = {o["name"]: o for o in response.json()["objects"]}
        assert "F_phi" in objects
        assert objects["F_phi"]["definition_tex"].startswith("F_{\\phi}")
        # it disappears from the remaining proposals (idempotent)
        again = client.get(f"/sessions/{sid}/definitions").json()
        assert "F_phi" not in {p["symbol"] for p in again["proposals"]}

    def test_unknown_definition_rejected(self, client):
        body = _create(client, SCALAR_TENSOR)
        response = client.post(
            f"/sessions/{body['session_id']}/definitions", json={"accept": ["def-nope"]}
        )
        assert response.status_code == 404

    def test_adopting_notation_does_not_unblock_or_block_plan(self, client):
        body = _create(client, SCALAR_TENSOR)
        sid = body["session_id"]
        before = client.get(f"/sessions/{sid}").json()["well_posed"]
        client.post(f"/sessions/{sid}/definitions", json={"accept": ["def-V-phi"]})
        after = client.get(f"/sessions/{sid}").json()["well_posed"]
        assert before is False and after is False


def _well_posed_scalar_tensor(client) -> str:
    """Create a scalar-tensor session and resolve every question (vary wrt g)."""
    body = _create(client, SCALAR_TENSOR)
    resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
    response = client.post(
        f"/sessions/{body['session_id']}/resolve", json={"resolutions": resolutions}
    )
    assert response.json()["well_posed"] is True, response.text
    return body["session_id"]


@requires_cadabra
class TestDerive:
    def _client(self, store, tmp_path, reply):
        return TestClient(
            create_app(store=store, llm=StubLLMAdapter(reply=reply), results_root=tmp_path)
        )

    def test_derive_returns_verified_eom(self, store, tmp_path):
        client = self._client(store, tmp_path, templates.get("eval3_scalar_tensor_metric"))
        sid = _well_posed_scalar_tensor(client)
        response = client.post(f"/sessions/{sid}/derive")
        assert response.status_code == 200, response.text
        derivations = response.json()["derivations"]
        assert [d["wrt"] for d in derivations] == ["g"]  # narrowed by vary-wrt=g
        g = derivations[0]
        assert g["verified"] is True, g["checks"]
        assert g["result_tex"]
        assert g["kernel_name"] == "cadabra"
        assert g["bundle_path"]

    def test_derive_blocked_when_questions_open(self, store, tmp_path):
        client = self._client(store, tmp_path, templates.get("eval3_scalar_tensor_metric"))
        body = _create(client, SCALAR_TENSOR)  # unresolved
        response = client.post(f"/sessions/{body['session_id']}/derive")
        assert response.status_code == 409
        assert response.json()["detail"]["questions"]

    def test_derive_rejects_undeclared_field(self, store, tmp_path):
        client = self._client(store, tmp_path, templates.get("eval3_scalar_tensor_metric"))
        sid = _well_posed_scalar_tensor(client)
        response = client.post(f"/sessions/{sid}/derive", json={"with_respect_to": ["not_a_field"]})
        assert response.status_code == 400

    def test_unknown_kind_is_422(self, store, tmp_path):
        client = self._client(store, tmp_path, templates.get("eval3_scalar_tensor_metric"))
        sid = _well_posed_scalar_tensor(client)
        response = client.post(f"/sessions/{sid}/derive", json={"kind": "bogus"})
        assert response.status_code == 422

    def test_perturbation_returns_verified_quadratic_action(self, store, tmp_path):
        client = self._client(store, tmp_path, templates.get("pert_scalar_quadratic"))
        sid = _well_posed_scalar_tensor(client)
        response = client.post(f"/sessions/{sid}/derive", json={"kind": "perturbation"})
        assert response.status_code == 200, response.text
        derivations = response.json()["derivations"]
        assert [d["wrt"] for d in derivations] == ["phi"]
        phi = derivations[0]
        assert phi["kind"] == "perturbation"
        assert phi["verified"] is True, phi["checks"]
        assert phi["result_tex"]

    def test_perturbation_refuses_nonscalar(self, store, tmp_path):
        client = self._client(store, tmp_path, templates.get("pert_scalar_quadratic"))
        sid = _well_posed_scalar_tensor(client)
        response = client.post(
            f"/sessions/{sid}/derive", json={"kind": "perturbation", "with_respect_to": ["g"]}
        )
        assert response.status_code == 422
