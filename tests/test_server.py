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

    def test_metric_curvature_session_lists_full_geometry_questionnaire(self, client):
        body = _create(client, "R")
        questions = {q["id"]: q for q in body["questions"]}

        assert {
            "amb-connection",
            "amb-torsion",
            "amb-nonmetricity",
            "amb-metric-compatibility",
            "amb-curvature-free",
        } <= questions.keys()
        assert {"levi-civita", "independent"} <= set(questions["amb-connection"]["options"])
        assert questions["amb-connection"]["kind"] == "inferable"
        assert questions["amb-torsion"]["kind"] == "inferable"
        assert questions["amb-nonmetricity"]["kind"] == "inferable"
        assert questions["amb-metric-compatibility"]["kind"] == "inferable"
        assert questions["amb-curvature-free"]["kind"] == "inferable"


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
        after = client.get(f"/sessions/{body['session_id']}").json()
        unresolved = {q["id"]: q["resolution"] for q in after["questions"]}
        assert unresolved[question["id"]] is None

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
        assert all(
            step["capability"] != "independent-connection" for step in plan.json()["steps"]
        )

    def test_independent_connection_live_path_surfaces_ricci_and_then_plans(self, client, store):
        body = _create(client, "R")
        sid = body["session_id"]

        response = client.post(
            f"/sessions/{sid}/resolve",
            json={
                "resolutions": {
                    "amb-conventions": "noether-default-v1",
                    "amb-vary-wrt": "g",
                    "amb-connection": "independent",
                    "amb-torsion": "torsion-allowed",
                    "amb-nonmetricity": "nonmetricity-free",
                    "amb-metric-compatibility": "metric-compatible",
                    "amb-curvature-free": "curvature-allowed",
                }
            },
        )
        assert response.status_code == 200
        assert response.json()["well_posed"] is False

        session = store.get(sid)
        assert session.npr.geometry.connection.type == "independent"

        ricci = next(q for q in response.json()["questions"] if q["id"] == "amb-ricci-contraction")
        assert ricci["resolution"] is None
        assert len(ricci["options"]) > 1

        blocked = client.get(f"/sessions/{sid}/plan")
        assert blocked.status_code == 409
        assert any("Ricci" in question for question in blocked.json()["detail"]["questions"])

        response = client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": {"amb-ricci-contraction": "first-fourth"}},
        )
        assert response.status_code == 200
        assert response.json()["well_posed"] is True

        plan = client.get(f"/sessions/{sid}/plan")
        assert plan.status_code == 200
        independent = next(
            step for step in plan.json()["steps"] if step["capability"] == "independent-connection"
        )
        assert "torsion=True, nonmetricity=False" in independent["description"]

    def test_reresolving_to_levi_civita_resets_connection_flags(self, client, store):
        body = _create(client, "R")
        sid = body["session_id"]

        response = client.post(
            f"/sessions/{sid}/resolve",
            json={
                "resolutions": {
                    "amb-connection": "independent",
                    "amb-torsion": "torsion-allowed",
                    "amb-nonmetricity": "nonmetricity-allowed",
                    "amb-metric-compatibility": "not-metric-compatible",
                    "amb-curvature-free": "curvature-allowed",
                }
            },
        )
        assert response.status_code == 200
        assert any(q["id"] == "amb-ricci-contraction" for q in response.json()["questions"])

        response = client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": {"amb-connection": "levi-civita"}},
        )
        assert response.status_code == 200

        session = store.get(sid)
        connection = session.npr.geometry.connection
        assert connection.type == "levi-civita"
        assert connection.torsion is False
        assert connection.nonmetricity is False
        assert connection.metric_compatible is True
        assert all(q["id"] != "amb-ricci-contraction" for q in response.json()["questions"])

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
MAXWELL = r"-\tfrac14 F_{\mu\nu} F^{\mu\nu}"


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
        response = client.post(
            f"/sessions/{sid}/derive", json={"kind": "perturbation", "with_respect_to": ["phi"]}
        )
        assert response.status_code == 200, response.text
        derivations = response.json()["derivations"]
        assert [d["wrt"] for d in derivations] == ["phi"]
        phi = derivations[0]
        assert phi["kind"] == "perturbation"
        assert phi["verified"] is True, phi["checks"]
        assert phi["result_tex"]

    def test_perturbation_metric_returns_verified_quadratic_action(self, store, tmp_path):
        client = self._client(store, tmp_path, templates.get("pert_metric_quadratic"))
        sid = _well_posed_scalar_tensor(client)
        response = client.post(
            f"/sessions/{sid}/derive", json={"kind": "perturbation", "with_respect_to": ["g"]}
        )
        assert response.status_code == 200, response.text
        g = response.json()["derivations"][0]
        assert g["wrt"] == "g"
        assert g["kind"] == "perturbation"
        assert g["verified"] is True, g["checks"]

    def test_perturbation_refuses_unsupported_field(self, store, tmp_path):
        client = self._client(store, tmp_path, templates.get("pert_scalar_quadratic"))
        body = _create(client, MAXWELL)
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        assert (
            client.post(
                f"/sessions/{body['session_id']}/resolve", json={"resolutions": resolutions}
            ).json()["well_posed"]
            is True
        )
        response = client.post(
            f"/sessions/{body['session_id']}/derive",
            json={"kind": "perturbation", "with_respect_to": ["F"]},
        )
        assert response.status_code == 422


class TestDeriveAdm:
    """ADM is verified by the SymPy component kernel, so it needs neither the
    Cadabra kernel nor an LLM backend on the server."""

    def _well_posed(self, client, lagrangian) -> str:
        body = _create(client, lagrangian)
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        resolved = client.post(
            f"/sessions/{body['session_id']}/resolve", json={"resolutions": resolutions}
        )
        assert resolved.json()["well_posed"] is True, resolved.text
        return body["session_id"]

    def test_adm_returns_verified_decomposition(self, store, tmp_path):
        client = TestClient(create_app(store=store, results_root=tmp_path))
        sid = self._well_posed(client, "R")
        response = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert response.status_code == 200, response.text
        derivations = response.json()["derivations"]
        assert len(derivations) == 3
        assert all(d["kind"] == "adm" and d["verified"] for d in derivations)
        assert derivations[0]["checks"]["lagrangian-split"] == "True"
        assert derivations[0]["kernel_name"] == "sympy"

    def test_unknown_kind_still_rejected(self, store, tmp_path):
        client = TestClient(create_app(store=store, results_root=tmp_path))
        sid = self._well_posed(client, "R")
        response = client.post(f"/sessions/{sid}/derive", json={"kind": "bogus"})
        assert response.status_code == 422


class TestResultHistory:
    """Derivations are recorded into the session and reloadable, verified by
    the SymPy ADM path so the test needs neither Cadabra nor an LLM."""

    def _well_posed(self, client, lagrangian="R") -> str:
        body = _create(client, lagrangian)
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        resolved = client.post(
            f"/sessions/{body['session_id']}/resolve", json={"resolutions": resolutions}
        )
        assert resolved.json()["well_posed"] is True, resolved.text
        return body["session_id"]

    def test_results_empty_before_any_derivation(self, store, tmp_path):
        client = TestClient(create_app(store=store, results_root=tmp_path))
        sid = self._well_posed(client)
        payload = client.get(f"/sessions/{sid}/results").json()
        assert payload == {"session_id": sid, "results": [], "stale_result_ids": []}

    def test_derivation_is_recorded_and_reloadable(self, store, tmp_path):
        client = TestClient(create_app(store=store, results_root=tmp_path))
        sid = self._well_posed(client)
        client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        # a fresh app over the same store proves the history persisted to disk
        reread = TestClient(create_app(store=store, results_root=tmp_path))
        payload = reread.get(f"/sessions/{sid}/results").json()
        assert len(payload["results"]) == 3
        assert all(d["result_id"].startswith("adm-") for d in payload["results"])
        assert payload["stale_result_ids"] == []

    def test_repeat_derivation_does_not_duplicate_history(self, store, tmp_path):
        client = TestClient(create_app(store=store, results_root=tmp_path))
        sid = self._well_posed(client)
        client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        payload = client.get(f"/sessions/{sid}/results").json()
        assert len(payload["results"]) == 3

    def test_resolution_after_results_marks_them_stale(self, store, tmp_path):
        client = TestClient(create_app(store=store, results_root=tmp_path))
        body = _create(client, "R")
        sid = body["session_id"]
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        client.post(f"/sessions/{sid}/resolve", json={"resolutions": resolutions})
        client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        before = client.get(f"/sessions/{sid}/results").json()
        # re-confirm one assumption now that a result exists
        first = body["questions"][0]
        client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": {first["id"]: first["options"][0]}},
        )
        after = client.get(f"/sessions/{sid}/results").json()
        assert before["stale_result_ids"] == []
        assert after["stale_result_ids"] == [after["results"][0]["result_id"]]


# ---------------------------------------------------------------------------
# VAL-EOM-008/017/018: Palatini session reachability, gate, and gated detail
# over HTTP
# ---------------------------------------------------------------------------

PALATINI_LAGRANGIAN = r"g^{\mu\nu} R_{\mu\nu}(\Gamma)"


def _resolve_all_palatini(client, body, *, connection="independent"):
    """Resolve all ambiguities for a Palatini session in two passes
    (geometry first, then any newly-opened ambiguities)."""
    sid = body["session_id"]
    resolutions = {}
    for q in body["questions"]:
        if q["id"] == "amb-connection":
            resolutions[q["id"]] = connection
        elif q["id"] == "amb-torsion":
            resolutions[q["id"]] = (
                "torsion-allowed" if connection == "independent"
                else "torsion-free"
            )
        elif q["id"] == "amb-nonmetricity":
            resolutions[q["id"]] = (
                "nonmetricity-allowed" if connection == "independent"
                else "nonmetricity-free"
            )
        elif q["id"] == "amb-metric-compatibility":
            resolutions[q["id"]] = (
                "not-metric-compatible" if connection == "independent"
                else "metric-compatible"
            )
        elif q["id"] == "amb-conventions":
            resolutions[q["id"]] = "noether-default-v1"
        elif q["id"] == "amb-vary-wrt":
            if "g and Gamma" in q["options"]:
                resolutions[q["id"]] = "g and Gamma"
            else:
                resolutions[q["id"]] = q["options"][0]
    result = client.post(
        f"/sessions/{sid}/resolve", json={"resolutions": resolutions}
    ).json()

    # Second pass for any newly-opened ambiguities
    remaining = {}
    for q in result.get("questions", []):
        if q.get("resolution") is None:
            if q["id"] == "amb-ricci-contraction":
                remaining[q["id"]] = "first-third"
            else:
                remaining[q["id"]] = q["options"][0]
    if remaining:
        result = client.post(
            f"/sessions/{sid}/resolve", json={"resolutions": remaining}
        ).json()
    return result


class TestPalatiniHttpReachability:
    """VAL-EOM-008/018: HTTP Palatini session reachability and elicitation
    gate."""

    def test_palatini_ingest_adds_gamma_and_connection_question(self, client):
        body = _create(client, PALATINI_LAGRANGIAN)
        object_names = {o["name"] for o in body["objects"]}
        assert "Gamma" in object_names
        gamma = next(o for o in body["objects"] if o["name"] == "Gamma")
        assert gamma["kind"] == "connection"
        q_ids = {q["id"] for q in body["questions"]}
        assert "amb-connection" in q_ids
        conn_q = next(q for q in body["questions"] if q["id"] == "amb-connection")
        assert "independent" in conn_q["options"]

    def test_palatini_plan_blocked_while_connection_open(self, client):
        """VAL-EOM-018: while the connection question is open, GET /plan
        returns 409."""
        body = _create(client, PALATINI_LAGRANGIAN)
        response = client.get(f"/sessions/{body['session_id']}/plan")
        assert response.status_code == 409

    def test_palatini_derive_blocked_while_connection_open(self, client):
        """VAL-EOM-008: while the connection question is open, POST /derive
        returns 409 (not a guess)."""
        body = _create(client, PALATINI_LAGRANGIAN)
        response = client.post(
            f"/sessions/{body['session_id']}/derive",
            json={"with_respect_to": ["g", "Gamma"]},
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail.get("blocked") is True
        assert detail.get("questions")

    def test_palatini_resolve_independent_enables_connection_step(self, client):
        """VAL-EOM-018: resolving connection=independent enables the
        independent-connection plan step."""
        body = _create(client, PALATINI_LAGRANGIAN)
        _resolve_all_palatini(client, body, connection="independent")
        plan = client.get(f"/sessions/{body['session_id']}/plan")
        assert plan.status_code == 200
        capabilities = [s["capability"] for s in plan.json()["steps"]]
        assert "independent-connection" in capabilities

    def test_palatini_resolve_levi_civita_no_connection_step(self, client):
        """Resolving to levi-civita must not produce an
        independent-connection step."""
        body = _create(client, PALATINI_LAGRANGIAN)
        _resolve_all_palatini(client, body, connection="levi-civita")
        plan = client.get(f"/sessions/{body['session_id']}/plan")
        assert plan.status_code == 200
        capabilities = [s["capability"] for s in plan.json()["steps"]]
        assert "independent-connection" not in capabilities

    @requires_cadabra
    def test_palatini_derive_resolved_returns_both_eoms(self, store, tmp_path):
        """VAL-EOM-008: on a resolved Palatini session, POST /derive with
        with_respect_to=['g','Gamma'] returns both EOMs."""
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
                results_root=tmp_path,
            )
        )
        body = _create(client, PALATINI_LAGRANGIAN)
        _resolve_all_palatini(client, body)
        sid = body["session_id"]
        response = client.post(
            f"/sessions/{sid}/derive",
            json={"with_respect_to": ["g", "Gamma"]},
        )
        assert response.status_code == 200
        derivations = response.json()["derivations"]
        wrt_set = {d["wrt"] for d in derivations}
        assert "g" in wrt_set
        assert "Gamma" in wrt_set

    @requires_cadabra
    def test_palatini_gated_eom_has_nonempty_detail(self, store, tmp_path):
        """VAL-EOM-017: a gated EOM derivation has verified==false and a
        non-empty detail."""
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(
                    reply=(
                        'print("NOETHER_RESULT: x");\n'
                        'print("NOETHER_CHECK: residue_zero=False");\n'
                    )
                ),
                results_root=tmp_path,
            )
        )
        body = _create(client, PALATINI_LAGRANGIAN)
        _resolve_all_palatini(client, body)
        sid = body["session_id"]
        response = client.post(
            f"/sessions/{sid}/derive",
            json={"with_respect_to": ["g"]},
        )
        assert response.status_code == 200
        d = response.json()["derivations"][0]
        assert d["verified"] is False
        assert d["detail"], "gated derivation must have non-empty detail"
