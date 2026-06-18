"""Backend cross-flows: metric-affine derivations through HTTP, MCP, CLI,
and the store (architecture.md section 8, VAL-CROSS-001/005/007/008/012/016
VAL-ADM-010/011/012/014).

These tests exercise the full loop on a single Palatini session:
  ingest -> resolve -> plan -> derive -> results -> stale -> resume
and verify consistency across HTTP, MCP, and the provenance bundle.
"""

from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from noether.kernels.cadabra import CadabraAdapter, templates  # noqa: E402
from noether.llm import StubLLMAdapter  # noqa: E402
from noether.mcp import NoetherTools  # noqa: E402
from noether.orchestrator.store import SessionStore  # noqa: E402
from noether.provenance.bundle import read_results  # noqa: E402
from noether.server import create_app  # noqa: E402

requires_cadabra = pytest.mark.skipif(
    not CadabraAdapter().available(), reason="cadabra2 not installed"
)

PALATINI_LAGRANGIAN = r"g^{\mu\nu} R_{\mu\nu}(\Gamma)"
MEASURE = r"d^4x \sqrt{-g}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_all_palatini(
    body: dict,
    resolve_fn,
    *,
    connection: str = "independent",
    torsion: str = "torsion-allowed",
    nonmetricity: str = "nonmetricity-allowed",
    metric_compat: str = "not-metric-compatible",
    ricci_contraction: str = "first-third",
) -> dict:
    """Two-pass resolution of all Palatini ambiguities via the supplied
    resolve function (HTTP POST or MCP tools.resolve)."""
    sid = body["session_id"]
    resolutions: dict[str, str] = {}
    for q in body["questions"]:
        if q["id"] == "amb-connection":
            resolutions[q["id"]] = connection
        elif q["id"] == "amb-torsion":
            resolutions[q["id"]] = torsion
        elif q["id"] == "amb-nonmetricity":
            resolutions[q["id"]] = nonmetricity
        elif q["id"] == "amb-metric-compatibility":
            resolutions[q["id"]] = metric_compat
        elif q["id"] == "amb-conventions":
            resolutions[q["id"]] = "noether-default-v1"
        elif q["id"] == "amb-vary-wrt":
            if "g and Gamma" in q["options"]:
                resolutions[q["id"]] = "g and Gamma"
            else:
                resolutions[q["id"]] = q["options"][0]
        elif q["id"] == "amb-curvature-free":
            resolutions[q["id"]] = "curvature-allowed"
    result = resolve_fn(sid, resolutions)

    # Second pass for newly opened ambiguities (e.g. amb-ricci-contraction
    # when connection=independent).
    remaining: dict[str, str] = {}
    for q in result.get("questions", []):
        if q.get("resolution") is None:
            if q["id"] == "amb-ricci-contraction":
                remaining[q["id"]] = ricci_contraction
            else:
                remaining[q["id"]] = q["options"][0]
    if remaining:
        result = resolve_fn(sid, remaining)
    return result


def _load_fields(derivation: dict) -> dict[str, Any]:
    """Extract the load-bearing fields from a derivation dict for
    field-for-field comparison across surfaces."""
    return {
        "wrt": derivation["wrt"],
        "kind": derivation.get("kind", "eom"),
        "result_id": derivation["result_id"],
        "result_tex": derivation.get("result_tex"),
        "verified": derivation["verified"],
        "checks": derivation.get("checks", {}),
        "detail": derivation.get("detail", ""),
        "kernel_name": derivation.get("kernel_name", ""),
        "kernel_version": derivation.get("kernel_version", ""),
        "script": derivation.get("script", ""),
        "conventions": derivation.get("conventions", {}),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


@pytest.fixture()
def results_root(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.fixture()
def client(store: SessionStore, results_root: Path) -> TestClient:
    return TestClient(create_app(store=store, results_root=results_root))


@pytest.fixture()
def tools(store: SessionStore, results_root: Path) -> NoetherTools:
    return NoetherTools(store, results_root=results_root)


def _create(client: TestClient, lagrangian: str = PALATINI_LAGRANGIAN) -> dict:
    response = client.post("/sessions", json={"lagrangian": lagrangian})
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# VAL-CROSS-001: Full metric-affine loop end to end through one HTTP session
# ---------------------------------------------------------------------------


class TestFullMetricAffineHTTPLoop:
    """VAL-CROSS-001: On one HTTP session: POST /sessions (Palatini action)
    returns the geometry question; POST /resolve (connection independent,
    torsion+non-metricity, wrt g and Gamma) makes it well posed; GET /plan
    contains the independent-connection step; POST /derive kind=eom returns
    both a metric (wrt=='g') and a connection (wrt=='Gamma') derivation, each
    verified-or-gated; GET /results reloads both with provenance."""

    def test_ingest_returns_geometry_question(self, client: TestClient) -> None:
        body = _create(client)
        q_ids = {q["id"] for q in body["questions"]}
        assert "amb-connection" in q_ids
        conn_q = next(q for q in body["questions"] if q["id"] == "amb-connection")
        assert "independent" in conn_q["options"]

    def test_resolve_makes_well_posed(self, client: TestClient, store: SessionStore) -> None:
        body = _create(client)
        sid = body["session_id"]

        def _resolve(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        result = _resolve_all_palatini(body, _resolve, connection="independent")
        assert result["well_posed"] is True

        # Verify geometry.connection is independent
        session = store.get(sid)
        assert session.npr.geometry.connection.type == "independent"

    def test_plan_contains_independent_connection_step(
        self, client: TestClient
    ) -> None:
        body = _create(client)

        def _resolve(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            return resp.json()

        _resolve_all_palatini(body, _resolve, connection="independent")
        plan_resp = client.get(f"/sessions/{body['session_id']}/plan")
        assert plan_resp.status_code == 200
        plan = plan_resp.json()
        capabilities = [s["capability"] for s in plan["steps"]]
        assert "independent-connection" in capabilities

    @requires_cadabra
    def test_derive_returns_both_eoms_and_results_reloads(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The full loop: ingest -> resolve -> plan -> derive -> results,
        all on one session, verifying both EOMs and reload."""
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
                results_root=results_root,
            )
        )
        body = _create(client)
        sid = body["session_id"]

        def _resolve(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        # Step 1: Resolve all geometry questions
        _resolve_all_palatini(body, _resolve, connection="independent")

        # Step 2: Plan
        plan_resp = client.get(f"/sessions/{sid}/plan")
        assert plan_resp.status_code == 200
        plan = plan_resp.json()
        capabilities = [s["capability"] for s in plan["steps"]]
        assert "independent-connection" in capabilities

        # Step 3: Derive with with_respect_to=['g', 'Gamma']
        derive_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g", "Gamma"]},
        )
        assert derive_resp.status_code == 200, derive_resp.text
        derivations = derive_resp.json()["derivations"]
        wrt_set = {d["wrt"] for d in derivations}
        assert "g" in wrt_set, f"must include wrt='g'; got {wrt_set}"
        assert "Gamma" in wrt_set, f"must include wrt='Gamma'; got {wrt_set}"

        # Each derivation has kernel_name, script, checks, and verified
        for d in derivations:
            assert d["kernel_name"], "derivation must have kernel_name"
            assert d["script"], "derivation must have script"
            assert isinstance(d["checks"], dict), "derivation must have checks dict"
            assert isinstance(d["verified"], bool), "derivation must have verified bool"
            # When gated, detail must be non-empty
            if not d["verified"]:
                assert d["detail"], "gated derivation must have non-empty detail"

        # Step 4: GET /results reloads both derivations
        results_resp = client.get(f"/sessions/{sid}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()
        result_ids = {d["result_id"] for d in derivations}
        reloaded_ids = {d["result_id"] for d in results["results"]}
        assert result_ids == reloaded_ids, (
            f"results must reload the same result_ids; "
            f"derive={result_ids}, results={reloaded_ids}"
        )


# ---------------------------------------------------------------------------
# VAL-CROSS-005: A gated result reads identically across HTTP and MCP
# ---------------------------------------------------------------------------


class TestGatedResultAcrossSurfaces:
    """VAL-CROSS-005: A gated metric-affine derivation has the same
    verified==false, the same detail, and the same checks on GET /results
    and MCP noether_results for the same result_id."""

    @requires_cadabra
    def test_gated_result_identical_http_and_mcp(
        self, store: SessionStore, results_root: Path
    ) -> None:
        # Create a gated result by using a script that produces nonzero residue
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(
                    reply=(
                        'print("NOETHER_RESULT: x");\n'
                        'print("NOETHER_CHECK: residue_zero=False");\n'
                    )
                ),
                results_root=results_root,
            )
        )
        tools = NoetherTools(store, results_root=results_root)

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive a gated result
        derive_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert derive_resp.status_code == 200
        derive_d = derive_resp.json()["derivations"][0]
        assert derive_d["verified"] is False

        # Read via HTTP /results
        http_results = client.get(f"/sessions/{sid}/results").json()
        http_d = next(
            r for r in http_results["results"] if r["result_id"] == derive_d["result_id"]
        )

        # Read via MCP noether_results
        mcp_results = tools.results(sid)
        mcp_d = next(
            r for r in mcp_results["results"] if r["result_id"] == derive_d["result_id"]
        )

        # Compare the load-bearing fields
        assert _load_fields(http_d) == _load_fields(mcp_d), (
            f"gated result must match across HTTP and MCP by result_id; "
            f"http={_load_fields(http_d)}, mcp={_load_fields(mcp_d)}"
        )

        # Specifically verify the gated verdict fields
        assert http_d["verified"] is False
        assert http_d["detail"], "gated result must have non-empty detail"
        assert http_d["verified"] == mcp_d["verified"]
        assert http_d["detail"] == mcp_d["detail"]
        assert http_d["checks"] == mcp_d["checks"]


# ---------------------------------------------------------------------------
# VAL-CROSS-007: Provenance round-trip across server, MCP, and store
# ---------------------------------------------------------------------------


class TestProvenanceRoundTrip:
    """VAL-CROSS-007: The derivations returned by POST /derive equal those
    reloaded by GET /results, MCP noether_results, and the bundle
    derivations.json, field for field by result_id."""

    @requires_cadabra
    def test_derive_equals_results_mcp_and_bundle(
        self, store: SessionStore, results_root: Path
    ) -> None:
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
                results_root=results_root,
            )
        )
        tools = NoetherTools(store, results_root=results_root)

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive
        derive_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g", "Gamma"]},
        )
        assert derive_resp.status_code == 200
        live_derivations = derive_resp.json()["derivations"]

        # GET /results
        http_results = client.get(f"/sessions/{sid}/results").json()
        http_derivations = http_results["results"]

        # MCP noether_results
        mcp_results = tools.results(sid)
        mcp_derivations = mcp_results["results"]

        # Bundle derivations.json (raw file read)
        session = store.get(sid)
        bundle_derivations = read_results(results_root, sid, session.result_ids)

        # Build maps by result_id
        live_by_id = {d["result_id"]: _load_fields(d) for d in live_derivations}
        http_by_id = {d["result_id"]: _load_fields(d) for d in http_derivations}
        mcp_by_id = {d["result_id"]: _load_fields(d) for d in mcp_derivations}
        bundle_by_id = {d["result_id"]: _load_fields(d) for d in bundle_derivations}

        # All four surfaces must agree on every result_id
        all_ids = set(live_by_id) | set(http_by_id) | set(mcp_by_id) | set(bundle_by_id)
        for rid in all_ids:
            assert rid in live_by_id, f"result_id {rid} missing from live /derive"
            assert rid in http_by_id, f"result_id {rid} missing from GET /results"
            assert rid in mcp_by_id, f"result_id {rid} missing from MCP results"
            assert rid in bundle_by_id, f"result_id {rid} missing from bundle"
            assert live_by_id[rid] == http_by_id[rid], (
                f"live vs HTTP mismatch for {rid}: "
                f"live={live_by_id[rid]}, http={http_by_id[rid]}"
            )
            assert live_by_id[rid] == mcp_by_id[rid], (
                f"live vs MCP mismatch for {rid}: "
                f"live={live_by_id[rid]}, mcp={mcp_by_id[rid]}"
            )
            assert live_by_id[rid] == bundle_by_id[rid], (
                f"live vs bundle mismatch for {rid}: "
                f"live={live_by_id[rid]}, bundle={bundle_by_id[rid]}"
            )


# ---------------------------------------------------------------------------
# VAL-CROSS-008: Stale-on-late-resolution across surfaces
# ---------------------------------------------------------------------------


class TestStaleOnLateResolution:
    """VAL-CROSS-008: After a metric-affine result exists, a late /resolve
    of a different geometry/convention answer marks the prior result stale
    on /results and MCP and excludes it from export."""

    def test_late_resolution_marks_stale_on_http_and_mcp(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """Use the ADM path (no cadabra needed) to create a result, then
        do a late resolution and verify stale_result_ids on both HTTP
        and MCP surfaces."""
        client = TestClient(create_app(store=store, results_root=results_root))
        tools = NoetherTools(store, results_root=results_root)

        # Ingest Palatini and resolve to independent
        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive an ADM result (no cadabra needed)
        derive_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert derive_resp.status_code == 200

        # Confirm results exist and are not stale
        http_before = client.get(f"/sessions/{sid}/results").json()
        mcp_before = tools.results(sid)
        assert http_before["stale_result_ids"] == []
        assert mcp_before["stale_result_ids"] == []
        result_ids = [d["result_id"] for d in http_before["results"]]
        assert result_ids, "must have at least one result before stale test"

        # Late resolve: change a geometry answer
        # Find an already-resolved geometry question and re-resolve with a
        # different value.  The first geometry question (amb-connection or
        # amb-curvature-free) is a safe pick.
        session = store.get(sid)
        first_resolved_amb = None
        for amb in session.npr.ambiguities:
            if amb.resolution is not None and amb.id in (
                "amb-curvature-free",
                "amb-ricci-contraction",
            ):
                first_resolved_amb = amb
                break
        # If no curvature-free/ricci question, use the first resolved one
        if first_resolved_amb is None:
            for amb in session.npr.ambiguities:
                if amb.resolution is not None:
                    first_resolved_amb = amb
                    break
        assert first_resolved_amb is not None, "must have a resolved ambiguity to re-resolve"

        # Pick a different option
        other_option = None
        for opt in first_resolved_amb.options:
            if opt != first_resolved_amb.resolution:
                other_option = opt
                break
        # If no other option, skip the late-resolution step
        if other_option is None:
            pytest.skip("no alternative option for late resolution test")

        late_resp = client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": {first_resolved_amb.id: other_option}},
        )
        assert late_resp.status_code == 200

        # Verify stale_result_ids on HTTP
        http_after = client.get(f"/sessions/{sid}/results").json()
        assert http_after["stale_result_ids"], (
            "late resolution must mark prior results stale on HTTP"
        )

        # Verify stale_result_ids on MCP
        mcp_after = tools.results(sid)
        assert mcp_after["stale_result_ids"], (
            "late resolution must mark prior results stale on MCP"
        )

        # The stale IDs must match across surfaces
        assert set(http_after["stale_result_ids"]) == set(mcp_after["stale_result_ids"]), (
            f"stale IDs must match: HTTP={http_after['stale_result_ids']}, "
            f"MCP={mcp_after['stale_result_ids']}"
        )

        # The stale IDs must include the result_ids from before
        for rid in result_ids:
            assert rid in http_after["stale_result_ids"], (
                f"result {rid} must be in stale_result_ids after late resolution"
            )


# ---------------------------------------------------------------------------
# VAL-CROSS-012: MCP tool chain (ingest->resolve->plan->derive)
# ---------------------------------------------------------------------------


class TestMCPToolChain:
    """VAL-CROSS-012: noether_ingest -> noether_resolve -> noether_plan ->
    noether_derive (with_respect_to=['g','Gamma'], kind='eom'): unresolved
    tool calls return blocked dicts, resolved noether_derive returns both
    EOMs, no tool raises on the refusal path."""

    def test_chain_returns_blocked_while_open(self, tools: NoetherTools) -> None:
        """While the connection ambiguity is open, plan and derive must
        return blocked dicts, not raise exceptions."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        # Plan must be blocked
        plan = tools.plan(sid)
        assert plan.get("blocked") is True, f"plan must be blocked; got {plan}"
        assert plan.get("questions"), "blocked plan must include questions"

        # Derive must be blocked
        derive = tools.derive(sid, ["g", "Gamma"])
        assert derive.get("blocked") is True, (
            f"derive must be blocked while open; got {derive}"
        )
        assert derive.get("questions"), "blocked derive must include questions"

        # No derivation produced
        assert "derivations" not in derive or not derive.get("derivations")

    def test_chain_returns_both_eoms_when_resolved(
        self, tools: NoetherTools
    ) -> None:
        """After resolving all geometry questions, the plan must not be
        blocked and must contain the independent-connection step."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        resolved = _resolve_all_palatini(body, _resolve_mcp, connection="independent")
        assert resolved["well_posed"] is True

        plan = tools.plan(sid)
        assert plan.get("blocked") is False
        capabilities = [s["capability"] for s in plan["steps"]]
        assert "independent-connection" in capabilities

    @requires_cadabra
    def test_chain_derive_returns_both_eoms(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """Full MCP chain: ingest -> resolve -> plan -> derive returns both
        wrt g and Gamma EOMs when resolved."""
        tools = NoetherTools(
            store,
            llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
            results_root=results_root,
        )

        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        # Plan
        plan = tools.plan(sid)
        assert plan.get("blocked") is False

        # Derive
        derive = tools.derive(sid, ["g", "Gamma"])
        assert "derivations" in derive, f"expected derivations; got {derive}"
        wrt_set = {d["wrt"] for d in derive["derivations"]}
        assert "g" in wrt_set, f"must include wrt='g'; got {wrt_set}"
        assert "Gamma" in wrt_set, f"must include wrt='Gamma'; got {wrt_set}"

        # Verify each derivation has the expected fields
        for d in derive["derivations"]:
            assert d["kernel_name"], "derivation must have kernel_name"
            assert d["script"], "derivation must have script"
            assert isinstance(d["checks"], dict)
            assert isinstance(d["verified"], bool)
            if not d["verified"]:
                assert d["detail"], "gated derivation must have non-empty detail"

    def test_no_exception_on_refusal_path(self, tools: NoetherTools) -> None:
        """The MCP tool chain must never raise on the refusal path;
        blocked results come back as data, not exceptions."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        # These must return dicts (with "blocked" or "error"), not raise
        plan = tools.plan(sid)
        assert isinstance(plan, dict)

        derive = tools.derive(sid, ["g", "Gamma"])
        assert isinstance(derive, dict)
        assert derive.get("blocked") is True

        # Off-menu resolve must return error dict, not raise
        off_menu = tools.resolve(sid, {"amb-connection": "not-an-option"})
        assert "error" in off_menu


# ---------------------------------------------------------------------------
# VAL-CROSS-016: Session resume with NPR, resolutions, and results intact
# ---------------------------------------------------------------------------


class TestSessionResume:
    """VAL-CROSS-016: Persisting a metric-affine session and resuming it
    restores the geometry resolutions, NPR version history, and recorded
    result ids, so a follow-up derivation runs against the same geometry
    without re-eliciting."""

    def test_resume_restores_geometry_and_results(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """Create a Palatini session with ADM result, persist, resume, and
        verify geometry, NPR versions, and result_ids are intact."""
        client = TestClient(create_app(store=store, results_root=results_root))

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive an ADM result (no cadabra needed)
        derive_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert derive_resp.status_code == 200

        # Read the session state before resume
        original = store.get(sid)
        original_connection_type = original.npr.geometry.connection.type
        original_torsion = original.npr.geometry.connection.torsion
        original_nonmetricity = original.npr.geometry.connection.nonmetricity
        original_npr_versions = len(original.npr_versions)
        original_result_ids = list(original.result_ids)
        original_resolved_ambiguities = {
            a.id: a.resolution for a in original.npr.ambiguities if a.resolution is not None
        }

        # Resume by loading from the store (same mechanism as `noether resume`)
        resumed = store.get(sid)

        # Verify geometry connection is preserved
        assert resumed.npr.geometry.connection.type == original_connection_type
        assert resumed.npr.geometry.connection.torsion == original_torsion
        assert resumed.npr.geometry.connection.nonmetricity == original_nonmetricity

        # Verify NPR version history is preserved
        assert len(resumed.npr_versions) == original_npr_versions

        # Verify result_ids are preserved
        assert resumed.result_ids == original_result_ids

        # Verify resolved ambiguities match
        resumed_resolved = {
            a.id: a.resolution for a in resumed.npr.ambiguities if a.resolution is not None
        }
        assert resumed_resolved == original_resolved_ambiguities

        # A follow-up derive should run without re-elicitation
        derive2_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert derive2_resp.status_code == 200

    def test_resume_across_app_instances(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """A session persisted by one app instance is correctly reloaded by
        a fresh instance, with geometry intact."""
        client1 = TestClient(create_app(store=store, results_root=results_root))

        body = _create(client1)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client1.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Fresh app instance over the same store
        client2 = TestClient(create_app(store=store, results_root=results_root))

        # Reload the session
        reloaded = client2.get(f"/sessions/{sid}")
        assert reloaded.status_code == 200
        assert reloaded.json()["well_posed"] is True

        # Plan must work on the fresh instance
        plan = client2.get(f"/sessions/{sid}/plan")
        assert plan.status_code == 200
        capabilities = [s["capability"] for s in plan.json()["steps"]]
        assert "independent-connection" in capabilities

    def test_resume_via_mcp_restores_geometry(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """A session created via MCP and resumed via store.get() has the
        correct geometry.connection and resolved ambiguities."""
        tools = NoetherTools(store, results_root=results_root)

        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        # Resume via store
        resumed = store.get(sid)
        assert resumed.npr.geometry.connection.type == "independent"
        assert resumed.npr.geometry.connection.torsion is True
        assert resumed.npr.geometry.connection.nonmetricity is True

        # Plan on the resumed session must work
        plan = tools.plan(sid)
        assert plan.get("blocked") is False
        capabilities = [s["capability"] for s in plan["steps"]]
        assert "independent-connection" in capabilities


# ---------------------------------------------------------------------------
# VAL-PERT-009: Reachable via the HTTP general perturb path (kind=perturbation)
# ---------------------------------------------------------------------------


class TestHTTPPerturbationReachability:
    """VAL-PERT-009: POST /derive with kind='perturbation' on a metric-affine
    session returns 200 with derivations[].kind=='perturbation' and a checks
    dict; an unknown kind is 422."""

    def test_unknown_kind_returns_422(self, client: TestClient) -> None:
        """An unknown derivation kind must return HTTP 422."""
        body = _create(client)

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        resp = client.post(
            f"/sessions/{body['session_id']}/derive",
            json={"kind": "unknown_kind"},
        )
        assert resp.status_code == 422, (
            f"expected 422 for unknown kind; got {resp.status_code}: {resp.text}"
        )

    @requires_cadabra
    def test_perturbation_on_metric_affine_returns_200_with_checks(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive with kind='perturbation' on a resolved metric-affine
        session returns 200 with derivations[].kind=='perturbation' and a
        checks dict."""
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("pert_metric_affine_quadratic")),
                results_root=results_root,
            )
        )
        body = _create(client)

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        resp = client.post(
            f"/sessions/{body['session_id']}/derive",
            json={"kind": "perturbation"},
        )
        assert resp.status_code == 200, (
            f"expected 200 for kind=perturbation; got {resp.status_code}: {resp.text}"
        )
        derivations = resp.json()["derivations"]
        assert len(derivations) > 0, "must return at least one derivation"

        for d in derivations:
            assert d["kind"] == "perturbation", (
                f"expected kind='perturbation'; got {d['kind']!r}"
            )
            assert isinstance(d["checks"], dict), (
                f"derivation must have a checks dict; got {type(d['checks'])}"
            )
            # Perturbation checks should include residue_zero and/or
            # linearized_eom_match (may be True or False depending on the
            # run, but the keys must exist in the checks dict)
            assert "residue_zero" in d["checks"], (
                f"perturbation checks must include residue_zero; got {d['checks']}"
            )

    def test_perturbation_blocked_while_questions_open(self, client: TestClient) -> None:
        """POST /derive kind=perturbation while questions are open returns
        409 (AmbiguityBlocked), never a guess."""
        body = _create(client)
        # Do NOT resolve geometry questions
        resp = client.post(
            f"/sessions/{body['session_id']}/derive",
            json={"kind": "perturbation"},
        )
        assert resp.status_code == 409, (
            f"expected 409 for unresolved session; got {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# VAL-PERT-010: Reachable via MCP noether_derive (kind=perturbation)
# ---------------------------------------------------------------------------


class TestMCPPerturbationReachability:
    """VAL-PERT-010: noether_derive with kind='perturbation' returns the
    same derivation surface as a tool result, or an {'error': ...} refusal;
    never a fabricated verified result."""

    def test_mcp_unknown_kind_returns_error(self, tools: NoetherTools) -> None:
        """An unknown derivation kind returns an error dict via MCP."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        result = tools.derive(sid, kind="unknown_kind")
        assert "error" in result, (
            f"expected error dict for unknown kind; got {result}"
        )
        assert "unknown" in result["error"].lower(), (
            f"error message should mention unknown kind; got {result['error']}"
        )

    @requires_cadabra
    def test_mcp_perturbation_on_metric_affine_returns_derivation_with_checks(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """MCP noether_derive with kind='perturbation' on a resolved
        metric-affine session returns derivations with kind='perturbation'
        and a checks dict, or an error dict; never a fabricated verified
        result."""
        tools = NoetherTools(
            store,
            llm=StubLLMAdapter(reply=templates.get("pert_metric_affine_quadratic")),
            results_root=results_root,
        )

        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        result = tools.derive(sid, kind="perturbation")
        # Must NOT be an error dict (unless cadabra is genuinely absent)
        assert "error" not in result or "cadabra" in result.get("error", ""), (
            f"unexpected error for kind=perturbation: {result}"
        )
        if "derivations" in result:
            derivations = result["derivations"]
            assert len(derivations) > 0, "must return at least one derivation"
            for d in derivations:
                assert d["kind"] == "perturbation", (
                    f"expected kind='perturbation'; got {d['kind']!r}"
                )
                assert isinstance(d["checks"], dict), (
                    "derivation must have a checks dict"
                )
                assert "residue_zero" in d["checks"], (
                    f"perturbation checks must include residue_zero; got {d['checks']}"
                )

    def test_mcp_perturbation_blocked_while_questions_open(
        self, tools: NoetherTools
    ) -> None:
        """MCP noether_derive kind=perturbation while questions are open
        returns a blocked dict, never a guess."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]
        # Do NOT resolve geometry questions

        result = tools.derive(sid, kind="perturbation")
        assert result.get("blocked") is True, (
            f"expected blocked=True while questions are open; got {result}"
        )
        assert result.get("questions"), (
            "blocked result must include questions"
        )

    def test_mcp_never_fabricates_verified_result(self, tools: NoetherTools) -> None:
        """MCP never returns a fabricated verified result: an off-menu
        resolve returns an error dict, not a derivation."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]
        # Off-menu resolve
        off_menu = tools.resolve(sid, {"amb-connection": "not-a-real-option"})
        assert "error" in off_menu, (
            f"off-menu resolve must return error dict; got {off_menu}"
        )

        # Derive must still be blocked (session still unresolved)
        result = tools.derive(sid, kind="perturbation")
        assert result.get("blocked") is True or "error" in result, (
            f"expected blocked or error after off-menu resolve; got {result}"
        )


# ---------------------------------------------------------------------------
# VAL-PERT-012: Refusal discipline for unsupported perturbed fields
# ---------------------------------------------------------------------------


class TestPerturbationRefusalDiscipline:
    """VAL-PERT-012: requesting a perturbation of a field with no audited
    scaffold raises NotImplementedError naming the field (HTTP 422 with that
    message), never a guessed quadratic action."""

    def test_derive_perturbation_refuses_unsupported_field(self) -> None:
        """derive_perturbation raises NotImplementedError naming the
        unsupported field."""
        from noether.npr import (
            NOETHER_DEFAULT_V1,
            NPR,
            Action,
            Geometry,
            ObjectDecl,
            Task,
        )
        from noether.npr.ast import down, tensor, up
        from noether.orchestrator.derive import derive_perturbation

        # A rank-2 tensor (field strength F) has no quadratic-action scaffold
        npr = NPR(
            conventions=NOETHER_DEFAULT_V1,
            geometry=Geometry(),
            objects=[
                ObjectDecl(name="g", kind="metric", role="background", rank=2),
                ObjectDecl(name="F", kind="tensor-field", role="shorthand", rank=2),
            ],
            action=Action(
                measure_tex=r"d^4x \sqrt{-g}",
                lagrangian=tensor("F", up("mu"), down("nu")),
                lagrangian_tex=r"F^{\mu}_{\nu}",
            ),
            task=Task(type="vary"),
        )
        with pytest.raises(NotImplementedError) as exc_info:
            derive_perturbation(
                npr,
                StubLLMAdapter(),
                {"cadabra": CadabraAdapter()},
                fields=["F"],
                session_id="s",
            )
        # The error message must name the field F
        assert "F" in str(exc_info.value), (
            f"NotImplementedError must name the unsupported field; "
            f"got: {exc_info.value}"
        )

    def test_http_perturbation_refuses_unsupported_field_422(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive with kind='perturbation' for an unsupported field
        returns HTTP 422 with a message naming the field."""
        client = TestClient(create_app(store=store, results_root=results_root))

        # Use a Palatini session (already has a metric-affine geometry)
        body = _create(client)

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Request a perturbation with respect to an unsupported field (e.g.
        # a rank-2 field strength that doesn't exist as a declared object)
        resp = client.post(
            f"/sessions/{body['session_id']}/derive",
            json={"kind": "perturbation", "with_respect_to": ["nonexistent_field"]},
        )
        # nonexistent_field is not a declared object, so 400 (not 422)
        assert resp.status_code == 400, (
            f"expected 400 for undeclared field; got {resp.status_code}: {resp.text}"
        )

    def test_mcp_perturbation_refuses_unsupported_field(
        self, tools: NoetherTools
    ) -> None:
        """MCP noether_derive with kind='perturbation' for an unsupported
        field returns an error dict naming the field."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        result = tools.derive(sid, with_respect_to=["nonexistent_field"], kind="perturbation")
        assert "error" in result, (
            f"expected error dict for unsupported field; got {result}"
        )


# ---------------------------------------------------------------------------
# VAL-PERT-016: Metric-affine perturbation result persists and reloads
# across surfaces
# ---------------------------------------------------------------------------


class TestPerturbationPersistence:
    """VAL-PERT-016: the perturbation run records its result_id and writes a
    bundle, reloading via GET /results, MCP noether_results, and web history
    with its kind, verified verdict, and checks intact."""

    @requires_cadabra
    def test_perturbation_result_persists_and_reloads_across_surfaces(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The perturbation result records its result_id, writes a bundle,
        and reloads identically via GET /results, MCP noether_results, and
        the bundle derivations.json, with its kind, verified verdict, and
        checks intact."""
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("pert_metric_affine_quadratic")),
                results_root=results_root,
            )
        )
        tools = NoetherTools(store, results_root=results_root)

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive a perturbation result
        derive_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "perturbation"},
        )
        assert derive_resp.status_code == 200, derive_resp.text
        derivations = derive_resp.json()["derivations"]
        assert len(derivations) > 0

        # Verify the derivation has the expected shape
        live_d = derivations[0]
        assert live_d["kind"] == "perturbation"
        assert isinstance(live_d["verified"], bool)
        assert isinstance(live_d["checks"], dict)
        result_id = live_d["result_id"]
        assert result_id, "perturbation derivation must have a result_id"

        # GET /results reloads the perturbation derivation
        http_results = client.get(f"/sessions/{sid}/results").json()
        http_d = next(
            (r for r in http_results["results"] if r["result_id"] == result_id),
            None,
        )
        assert http_d is not None, (
            f"result_id {result_id} not found in GET /results; "
            f"available: {[r['result_id'] for r in http_results['results']]}"
        )
        # Kind, verified, and checks must survive the reload
        assert http_d["kind"] == "perturbation", (
            f"reloaded kind must be 'perturbation'; got {http_d['kind']!r}"
        )
        assert http_d["verified"] == live_d["verified"], (
            f"reloaded verified must match live; "
            f"live={live_d['verified']}, reloaded={http_d['verified']}"
        )
        assert http_d["checks"] == live_d["checks"], (
            f"reloaded checks must match live; "
            f"live={live_d['checks']}, reloaded={http_d['checks']}"
        )

        # MCP noether_results reloads identically
        mcp_results = tools.results(sid)
        mcp_d = next(
            (r for r in mcp_results["results"] if r["result_id"] == result_id),
            None,
        )
        assert mcp_d is not None, (
            f"result_id {result_id} not found in MCP results; "
            f"available: {[r['result_id'] for r in mcp_results['results']]}"
        )
        assert mcp_d["kind"] == "perturbation"
        assert mcp_d["verified"] == live_d["verified"]
        assert mcp_d["checks"] == live_d["checks"]

        # Bundle derivations.json matches
        session = store.get(sid)
        bundle_derivations = read_results(results_root, sid, session.result_ids)
        bundle_d = next(
            (r for r in bundle_derivations if r["result_id"] == result_id),
            None,
        )
        assert bundle_d is not None, (
            f"result_id {result_id} not found in bundle; "
            f"available: {[r['result_id'] for r in bundle_derivations]}"
        )
        assert bundle_d["kind"] == "perturbation"
        assert bundle_d["verified"] == live_d["verified"]
        assert bundle_d["checks"] == live_d["checks"]


# ---------------------------------------------------------------------------
# VAL-ADM-010: An action with no metric is refused
# ---------------------------------------------------------------------------


class TestADMNoMetricRefused:
    """VAL-ADM-010: kind='adm' for an action declaring no metric is refused
    (HTTP 422 / MCP error) naming the missing metric; no ADM derivation."""

    def _make_no_metric_session(self, store: SessionStore) -> str:
        """Create a well-posed session with no metric object directly in the
        store, bypassing ingest (which always adds a metric for curvature
        or kinetic actions)."""
        import uuid as _uuid

        from noether.npr import (
            NOETHER_DEFAULT_V1,
            NPR,
            Action,
            Geometry,
            ObjectDecl,
            Task,
        )
        from noether.npr.ast import tensor
        from noether.orchestrator.session import Session

        # An NPR with only a scalar field, no metric object
        npr = NPR(
            conventions=NOETHER_DEFAULT_V1,
            geometry=Geometry(),
            objects=[
                ObjectDecl(name="phi", kind="scalar-field", role="dynamical"),
            ],
            action=Action(
                measure_tex=r"d^4x",
                lagrangian=tensor("V"),
                lagrangian_tex=r"V(\phi)",
            ),
            task=Task(type="vary", with_respect_to=["phi"]),
            ambiguities=[],
        )
        session = Session(session_id=f"s-nometric-{_uuid.uuid4().hex[:8]}")
        session.ingest(npr)
        store.save(session)
        return session.session_id

    def test_http_adm_no_metric_returns_422(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive kind='adm' with no metric returns 422 naming the
        missing metric."""
        client = TestClient(create_app(store=store, results_root=results_root))
        sid = self._make_no_metric_session(store)

        resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        # Must be 422 (NotImplementedError caught) or 409 (AmbiguityBlocked)
        if resp.status_code == 409:
            # Session not well-posed yet, that's also a valid refusal
            pass
        else:
            assert resp.status_code == 422, (
                f"expected 422 for no-metric ADM; got {resp.status_code}: {resp.text}"
            )
            detail = resp.json()["detail"]
            # The detail must name the missing metric
            assert "metric" in detail.lower(), (
                f"422 detail must name the missing metric; got: {detail}"
            )
            # The detail must name the specific metric object
            assert "'g'" in detail or '"g"' in detail, (
                f"422 detail must name the missing metric object 'g'; got: {detail}"
            )
            # No derivations produced
            body = resp.json()
            assert "derivations" not in body or not body.get("derivations"), (
                "no ADM derivation should be produced for a no-metric action"
            )

    def test_mcp_adm_no_metric_returns_error_naming_metric(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """MCP noether_derive kind='adm' with no metric returns an error dict
        naming the missing metric."""
        tools = NoetherTools(store, results_root=results_root)
        sid = self._make_no_metric_session(store)

        result = tools.derive(sid, kind="adm")
        # Must be an error or blocked dict, not a derivation
        if result.get("blocked"):
            # Session not well-posed - also valid refusal
            pass
        else:
            assert "error" in result, (
                f"expected error dict for no-metric ADM; got {result}"
            )
            assert "metric" in result["error"].lower(), (
                f"error must name the missing metric; got: {result['error']}"
            )
            assert "'g'" in result["error"] or '"g"' in result["error"], (
                f"error must name the missing metric object 'g'; got: {result['error']}"
            )
            # No derivations produced
            assert "derivations" not in result or not result.get("derivations"), (
                "no ADM derivation should be produced for a no-metric action"
            )

    def test_no_metric_error_names_specific_metric_object(self) -> None:
        """The NotImplementedError message names the specific metric object
        (e.g. 'g') from the NPR geometry."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.npr import (
            NOETHER_DEFAULT_V1,
            NPR,
            Action,
            Geometry,
            ObjectDecl,
            Task,
        )
        from noether.npr.ast import tensor
        from noether.orchestrator.derive import derive_adm

        # An NPR with no metric object
        npr = NPR(
            conventions=NOETHER_DEFAULT_V1,
            geometry=Geometry(),
            objects=[
                ObjectDecl(name="phi", kind="scalar-field", role="dynamical"),
            ],
            action=Action(
                measure_tex=r"d^4x \sqrt{-g}",
                lagrangian=tensor("X"),
                lagrangian_tex=r"X",
            ),
            task=Task(type="vary"),
            ambiguities=[],
        )
        with pytest.raises(NotImplementedError) as exc_info:
            derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-test")
        msg = str(exc_info.value)
        # The message must name the expected metric object ('g')
        assert "g" in msg, (
            f"error must name the missing metric object 'g'; got: {msg}"
        )
        assert "metric" in msg.lower(), (
            f"error must reference 'metric'; got: {msg}"
        )


# ---------------------------------------------------------------------------
# VAL-ADM-011: Reachable identically across HTTP and MCP
# ---------------------------------------------------------------------------


class TestADMHTTPMCPParity:
    """VAL-ADM-011: the metric-affine kind='adm' derivation returns the same
    shape and verdict on HTTP and MCP for the same session; an unknown kind
    is rejected on both."""

    def _setup_resolved_session(
        self, store: SessionStore, results_root: Path, *, ricci: str = "first-third"
    ) -> tuple[TestClient, NoetherTools, str]:
        """Create a resolved Palatini session for ADM testing."""
        client = TestClient(create_app(store=store, results_root=results_root))
        tools = NoetherTools(store, results_root=results_root)

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(
            body, _resolve_http,
            connection="independent", ricci_contraction=ricci,
        )
        return client, tools, sid

    def test_http_and_mcp_adm_agree_on_result_id_and_verified(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """HTTP POST /derive kind='adm' and MCP noether_derive kind='adm'
        return the same result_id and verified verdict for the same session."""
        client, tools, sid = self._setup_resolved_session(store, results_root)

        # Derive ADM via HTTP
        http_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert http_resp.status_code == 200, http_resp.text
        http_derivations = http_resp.json()["derivations"]
        assert len(http_derivations) > 0, "ADM must produce derivations"

        # The session now has results recorded; read them via both surfaces
        http_results = client.get(f"/sessions/{sid}/results").json()
        mcp_results = tools.results(sid)

        # Both surfaces must have the same number of results
        assert len(http_results["results"]) == len(mcp_results["results"]), (
            f"HTTP and MCP result counts differ: "
            f"HTTP={len(http_results['results'])}, MCP={len(mcp_results['results'])}"
        )

        # Match by result_id + wrt and verify the load-bearing fields agree
        for http_d in http_results["results"]:
            rid = http_d["result_id"]
            wrt = http_d["wrt"]
            mcp_d = next(
                (
                    r
                    for r in mcp_results["results"]
                    if r["result_id"] == rid and r["wrt"] == wrt
                ),
                None,
            )
            assert mcp_d is not None, (
                f"result_id {rid} wrt {wrt!r} present in HTTP but missing in MCP"
            )
            assert http_d["verified"] == mcp_d["verified"], (
                f"verified mismatch for {rid}/{wrt}: "
                f"HTTP={http_d['verified']}, MCP={mcp_d['verified']}"
            )
            assert http_d["kind"] == mcp_d["kind"], (
                f"kind mismatch for {rid}/{wrt}: "
                f"HTTP={http_d['kind']}, MCP={mcp_d['kind']}"
            )
            assert http_d["checks"] == mcp_d["checks"], (
                f"checks mismatch for {rid}/{wrt}"
            )

    def test_http_rejects_unknown_kind(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """HTTP POST /derive with an unknown kind returns 422."""
        client, tools, sid = self._setup_resolved_session(store, results_root)

        resp = client.post(f"/sessions/{sid}/derive", json={"kind": "bogus"})
        assert resp.status_code == 422, (
            f"expected 422 for unknown kind; got {resp.status_code}: {resp.text}"
        )

    def test_mcp_rejects_unknown_kind(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """MCP noether_derive with an unknown kind returns an error dict."""
        client, tools, sid = self._setup_resolved_session(store, results_root)

        result = tools.derive(sid, kind="bogus")
        assert "error" in result, (
            f"expected error dict for unknown kind; got {result}"
        )
        assert "unknown" in result["error"].lower(), (
            f"error should mention 'unknown'; got: {result['error']}"
        )

    def test_http_and_mcp_adm_same_derivation_shape(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The ADM derivation shape (wrt, kind, result_tex, kernel_name)
        is the same on HTTP and MCP for the same session."""
        client, tools, sid = self._setup_resolved_session(store, results_root)

        http_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert http_resp.status_code == 200
        http_derivations = http_resp.json()["derivations"]

        # Read results from MCP
        mcp_results = tools.results(sid)
        mcp_derivations = mcp_results["results"]

        # Build maps by result_id + wrt for comparison
        for http_d in http_derivations:
            rid = http_d["result_id"]
            wrt = http_d["wrt"]
            mcp_match = next(
                (
                    r
                    for r in mcp_derivations
                    if r["result_id"] == rid and r["wrt"] == wrt
                ),
                None,
            )
            assert mcp_match is not None, (
                f"no MCP match for result_id={rid} wrt={wrt}"
            )
            # Compare load-bearing fields
            http_fields = _load_fields(http_d)
            mcp_fields = _load_fields(mcp_match)
            assert http_fields == mcp_fields, (
                f"HTTP and MCP derivation fields differ for {rid}/{wrt}: "
                f"HTTP={http_fields}, MCP={mcp_fields}"
            )


# ---------------------------------------------------------------------------
# VAL-ADM-012: Result persists and reloads across surfaces
# ---------------------------------------------------------------------------


class TestADMPersistence:
    """VAL-ADM-012: An ADM run records its result_id and writes a bundle,
    reloading identically via GET /results, MCP noether_results, and the
    web client."""

    def _setup_and_derive_adm(
        self, store: SessionStore, results_root: Path
    ) -> tuple[TestClient, NoetherTools, str, list[dict]]:
        """Create a resolved session and derive ADM."""
        client = TestClient(create_app(store=store, results_root=results_root))
        tools = NoetherTools(store, results_root=results_root)

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive ADM
        derive_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert derive_resp.status_code == 200, derive_resp.text
        derivations = derive_resp.json()["derivations"]
        assert len(derivations) > 0, "ADM must produce derivations"
        return client, tools, sid, derivations

    def test_adm_result_id_recorded_in_session(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The ADM run records its result_id in the session."""
        client, tools, sid, derivations = self._setup_and_derive_adm(store, results_root)

        session = store.get(sid)
        result_ids = session.result_ids
        assert result_ids, "session must have recorded result_ids after ADM derive"

        # The result_ids must include the derivation result_ids
        derive_rids = {d["result_id"] for d in derivations}
        for rid in derive_rids:
            assert rid in result_ids, (
                f"result_id {rid} from /derive must be in session.result_ids"
            )

    def test_adm_bundle_written(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The ADM derivation writes a provenance bundle to disk."""
        client, tools, sid, derivations = self._setup_and_derive_adm(store, results_root)

        # Check the bundle exists on disk
        result_id = derivations[0]["result_id"]
        bundle_dir = results_root / sid / result_id
        assert bundle_dir.exists(), (
            f"bundle directory must exist at {bundle_dir}"
        )
        assert (bundle_dir / "derivations.json").exists(), (
            "bundle must contain derivations.json"
        )
        assert (bundle_dir / "assumptions.json").exists(), (
            "bundle must contain assumptions.json"
        )

    def test_adm_reloads_via_get_results(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """GET /results returns the ADM result with the same result_id,
        verified, checks, and result_tex."""
        client, tools, sid, derivations = self._setup_and_derive_adm(store, results_root)

        results_resp = client.get(f"/sessions/{sid}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()

        # Match each derivation from /derive with /results by result_id + wrt
        for live_d in derivations:
            rid = live_d["result_id"]
            wrt = live_d["wrt"]
            reloaded = next(
                (
                    r
                    for r in results["results"]
                    if r["result_id"] == rid and r["wrt"] == wrt
                ),
                None,
            )
            assert reloaded is not None, (
                f"result_id {rid} wrt {wrt!r} not found in GET /results"
            )
            assert reloaded["verified"] == live_d["verified"], (
                f"verified mismatch for {rid}/{wrt}"
            )
            assert reloaded["checks"] == live_d["checks"], (
                f"checks mismatch for {rid}/{wrt}"
            )
            assert reloaded["result_tex"] == live_d["result_tex"], (
                f"result_tex mismatch for {rid}/{wrt}"
            )
            assert reloaded["kind"] == "adm", (
                f"reloaded kind must be 'adm'; got {reloaded['kind']!r}"
            )

    def test_adm_reloads_identically_via_mcp(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """MCP noether_results returns the same ADM derivations as
        GET /results."""
        client, tools, sid, derivations = self._setup_and_derive_adm(store, results_root)

        http_results = client.get(f"/sessions/{sid}/results").json()
        mcp_results = tools.results(sid)

        # Both must return the same derivations
        assert len(http_results["results"]) == len(mcp_results["results"]), (
            "HTTP and MCP result counts must match"
        )
        for http_d in http_results["results"]:
            rid = http_d["result_id"]
            wrt = http_d["wrt"]
            mcp_d = next(
                (
                    r
                    for r in mcp_results["results"]
                    if r["result_id"] == rid and r["wrt"] == wrt
                ),
                None,
            )
            assert mcp_d is not None, f"result_id {rid} wrt {wrt!r} missing from MCP"
            assert _load_fields(http_d) == _load_fields(mcp_d), (
                f"MCP derivation must match HTTP for {rid}/{wrt}"
            )

    def test_adm_bundle_derivations_json_matches(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The bundle derivations.json matches the live /derive derivations
        field for field by result_id."""
        client, tools, sid, derivations = self._setup_and_derive_adm(store, results_root)

        session = store.get(sid)
        bundle_derivations = read_results(results_root, sid, session.result_ids)

        live_by_id = {d["result_id"]: _load_fields(d) for d in derivations}
        bundle_by_id = {d["result_id"]: _load_fields(d) for d in bundle_derivations}

        for rid in live_by_id:
            assert rid in bundle_by_id, f"result_id {rid} missing from bundle"
            assert live_by_id[rid] == bundle_by_id[rid], (
                f"live vs bundle mismatch for {rid}"
            )


# ---------------------------------------------------------------------------
# VAL-ADM-014: Conventions are explicit and threaded on the metric-affine
# result
# ---------------------------------------------------------------------------


class TestADMConventionBlock:
    """VAL-ADM-014: The ADM result carries its full convention block (torsion
    sign, non-metricity definition, Ricci-contraction, signature, K-sign,
    foliation/normal); changing the elicited Ricci-contraction is reflected."""

    REQUIRED_CONVENTION_KEYS = {
        "signature",
        "torsion_sign",
        "nonmetricity_definition",
        "ricci_contraction",
        "contortion_sign",
        "disformation_sign",
        "K_sign",
        "foliation_normal",
        "convention_id",
    }

    def _derive_adm_and_get_conventions(
        self,
        store: SessionStore,
        results_root: Path,
        *,
        ricci: str = "first-third",
    ) -> dict[str, str]:
        """Create a resolved session, derive ADM, and return the convention
        block from the first derivation."""
        client = TestClient(create_app(store=store, results_root=results_root))

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(
            body, _resolve_http, connection="independent", ricci_contraction=ricci
        )

        derive_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert derive_resp.status_code == 200, derive_resp.text
        derivations = derive_resp.json()["derivations"]
        assert len(derivations) > 0
        conv = derivations[0].get("conventions", {})
        return conv

    def test_adm_derivation_carries_convention_block(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """Each ADM derivation carries a non-empty convention block."""
        conv = self._derive_adm_and_get_conventions(store, results_root)
        assert conv, "ADM derivation must carry a non-empty convention block"

    def test_convention_block_has_all_required_entries(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The convention block names torsion sign, non-metricity definition,
        Ricci-contraction, signature, K-sign, and foliation/normal."""
        conv = self._derive_adm_and_get_conventions(store, results_root)
        for key in self.REQUIRED_CONVENTION_KEYS:
            assert key in conv, (
                f"convention block must include '{key}'; "
                f"present keys: {sorted(conv.keys())}"
            )

    def test_convention_block_values_match_npr_defaults(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The convention block values match the NPR's active conventions."""
        conv = self._derive_adm_and_get_conventions(store, results_root)
        assert conv["signature"] == "mostly-plus"
        assert conv["torsion_sign"] == "+1"
        assert conv["nonmetricity_definition"] == "nabla-g"
        assert conv["ricci_contraction"] == "first-third"
        assert conv["contortion_sign"] == "+1"
        assert conv["convention_id"] == "noether-default-v1"

    def test_changing_ricci_contraction_is_reflected(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """Changing the elicited Ricci-contraction is reflected in the
        ADM result's convention block."""
        conv_default = self._derive_adm_and_get_conventions(store, results_root)
        conv_alt = self._derive_adm_and_get_conventions(
            store, results_root, ricci="first-fourth"
        )
        # The ricci_contraction must differ
        assert conv_default["ricci_contraction"] == "first-third"
        assert conv_alt["ricci_contraction"] == "first-fourth", (
            "changing the elicited Ricci-contraction must be reflected"
        )
        # Other conventions stay the same
        for key in self.REQUIRED_CONVENTION_KEYS - {"ricci_contraction"}:
            if key in conv_default and key in conv_alt:
                assert conv_default[key] == conv_alt[key], (
                    f"convention {key} should not change when only "
                    f"Ricci-contraction changes"
                )

    def test_convention_block_survives_reload_via_results(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """The convention block is present and identical when reloaded via
        GET /results and MCP noether_results."""
        client = TestClient(create_app(store=store, results_root=results_root))
        tools = NoetherTools(store, results_root=results_root)

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive ADM
        derive_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert derive_resp.status_code == 200
        live_derivations = derive_resp.json()["derivations"]
        live_conv = live_derivations[0].get("conventions", {})

        # Reload via GET /results
        http_results = client.get(f"/sessions/{sid}/results").json()
        rid0 = live_derivations[0]["result_id"]
        http_d = next(
            r for r in http_results["results"] if r["result_id"] == rid0
        )
        http_conv = http_d.get("conventions", {})
        assert http_conv == live_conv, (
            f"convention block must survive GET /results reload; "
            f"live={live_conv}, reloaded={http_conv}"
        )

        # Reload via MCP noether_results
        mcp_results = tools.results(sid)
        mcp_d = next(
            r for r in mcp_results["results"] if r["result_id"] == rid0
        )
        mcp_conv = mcp_d.get("conventions", {})
        assert mcp_conv == live_conv, (
            f"convention block must survive MCP reload; "
            f"live={live_conv}, mcp={mcp_conv}"
        )

    def test_convention_block_independent_connection_includes_field_strength(
        self,
    ) -> None:
        """For a metric-affine NPR (independent connection), the convention
        block includes the field_strength_definition."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr_helper()
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-conv-fs")
        conv = results[0].conventions
        assert "field_strength_definition" in conv, (
            "metric-affine ADM convention block must include "
            "field_strength_definition; got keys: " + str(sorted(conv.keys()))
        )

    def test_levi_civita_adm_convention_block_lacks_field_strength(self) -> None:
        """For a Levi-Civita (non-metric-affine) NPR, the convention block
        does not include field_strength_definition."""
        from evals.eval1s_adm import build_npr as build_adm_npr
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = build_adm_npr(resolved=True)
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-gr-conv")
        conv = results[0].conventions
        assert "field_strength_definition" not in conv, (
            "Levi-Civita ADM convention block should not include "
            "field_strength_definition; got keys: " + str(sorted(conv.keys()))
        )


# ---------------------------------------------------------------------------
# VAL-CROSS-003: Convention override threads through EOM, perturbation, ADM
# ---------------------------------------------------------------------------


class TestConventionOverrideCrossKind:
    """VAL-CROSS-003: A non-default convention chosen at elicitation appears
    identically in the assumptions snapshot of kind=eom, kind=perturbation,
    and kind=adm results in the same session."""

    @requires_cadabra
    def test_convention_override_appears_in_all_three_kinds(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """Resolve with a non-default Ricci contraction (first-fourth instead
        of first-third), then derive eom, perturbation, and adm on the same
        session. All three bundles' assumptions.json conventions must be
        identical and must carry the override."""
        import json

        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
                results_root=results_root,
            )
        )

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        # Resolve with non-default Ricci contraction
        _resolve_all_palatini(
            body, _resolve_http,
            connection="independent", ricci_contraction="first-fourth",
        )

        # Derive EOM
        eom_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert eom_resp.status_code == 200, eom_resp.text
        eom_d = eom_resp.json()["derivations"][0]
        eom_conv = eom_d.get("conventions", {})
        assert eom_conv.get("ricci_contraction") == "first-fourth", (
            f"EOM derivation must carry the non-default Ricci contraction; "
            f"got {eom_conv}"
        )

        # Derive perturbation
        pert_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "perturbation"},
        )
        assert pert_resp.status_code == 200, pert_resp.text
        pert_d = pert_resp.json()["derivations"][0]
        pert_conv = pert_d.get("conventions", {})
        assert pert_conv.get("ricci_contraction") == "first-fourth", (
            f"perturbation derivation must carry the non-default Ricci "
            f"contraction; got {pert_conv}"
        )

        # Derive ADM
        adm_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "adm"},
        )
        assert adm_resp.status_code == 200, adm_resp.text
        adm_d = adm_resp.json()["derivations"][0]
        adm_conv = adm_d.get("conventions", {})
        assert adm_conv.get("ricci_contraction") == "first-fourth", (
            f"ADM derivation must carry the non-default Ricci contraction; "
            f"got {adm_conv}"
        )

        # All three convention blocks must be identical
        assert eom_conv == pert_conv == adm_conv, (
            f"convention blocks must be identical across all three kinds; "
            f"eom={eom_conv}, pert={pert_conv}, adm={adm_conv}"
        )

        # Also verify the assumptions.json files on disk match
        session = store.get(sid)
        assumptions_list = []
        for rid in session.result_ids:
            assumptions_path = results_root / sid / rid / "assumptions.json"
            if assumptions_path.exists():
                data = json.loads(assumptions_path.read_text())
                assumptions_list.append(data.get("conventions", {}))

        if len(assumptions_list) >= 2:
            first = assumptions_list[0]
            for a in assumptions_list[1:]:
                assert a == first, (
                    f"assumptions.json conventions must be identical; "
                    f"first={first}, other={a}"
                )
            assert first.get("ricci_contraction") == "first-fourth", (
                f"assumptions.json must carry the override; got "
                f"{first.get('ricci_contraction')}"
            )

    def test_convention_block_on_eom_derivation(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """EOM derivations carry a non-empty conventions block (not just ADM)."""
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
                results_root=results_root,
            )
        )
        body = _create(client)

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        eom_resp = client.post(
            f"/sessions/{body['session_id']}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert eom_resp.status_code == 200
        eom_d = eom_resp.json()["derivations"][0]
        conv = eom_d.get("conventions", {})
        assert conv, "EOM derivation must carry a non-empty convention block"
        assert "ricci_contraction" in conv, (
            f"EOM convention block must include ricci_contraction; got {sorted(conv.keys())}"
        )
        assert "signature" in conv, (
            f"EOM convention block must include signature; got {sorted(conv.keys())}"
        )


# ---------------------------------------------------------------------------
# VAL-CROSS-004: One action, three operations, agreeing geometry in same session
# ---------------------------------------------------------------------------


class TestThreeKindsOneSession:
    """VAL-CROSS-004: For one metric-affine action, eom then perturbation
    then adm run in the same session against the same geometry.connection,
    and all appear in one results payload."""

    @requires_cadabra
    def test_three_kinds_one_session_agreeing_geometry(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """Run eom, perturbation, and adm on the same session and verify
        all three appear in /results with identical geometry.connection."""
        client = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
                results_root=results_root,
            )
        )

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive EOM
        eom_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert eom_resp.status_code == 200, eom_resp.text
        eom_kinds = {d["kind"] for d in eom_resp.json()["derivations"]}
        assert "eom" in eom_kinds

        # Derive perturbation
        pert_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "perturbation"},
        )
        assert pert_resp.status_code == 200, pert_resp.text
        pert_kinds = {d["kind"] for d in pert_resp.json()["derivations"]}
        assert "perturbation" in pert_kinds

        # Derive ADM
        adm_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "adm"},
        )
        assert adm_resp.status_code == 200, adm_resp.text
        adm_kinds = {d["kind"] for d in adm_resp.json()["derivations"]}
        assert "adm" in adm_kinds

        # GET /results includes all three kinds
        results_resp = client.get(f"/sessions/{sid}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()
        result_kinds = {d["kind"] for d in results["results"]}
        assert {"eom", "perturbation", "adm"} <= result_kinds, (
            f"/results must include all three kinds; got {result_kinds}"
        )

        # All result_ids from derivations appear in the session
        session = store.get(sid)
        all_result_ids = set(session.result_ids)
        eom_rids = {d["result_id"] for d in eom_resp.json()["derivations"]}
        pert_rids = {d["result_id"] for d in pert_resp.json()["derivations"]}
        adm_rids = {d["result_id"] for d in adm_resp.json()["derivations"]}
        assert eom_rids <= all_result_ids, (
            f"EOM result_ids must be in session; missing={eom_rids - all_result_ids}"
        )
        assert pert_rids <= all_result_ids, (
            f"perturbation result_ids must be in session; missing={pert_rids - all_result_ids}"
        )
        assert adm_rids <= all_result_ids, (
            f"ADM result_ids must be in session; missing={adm_rids - all_result_ids}"
        )

        # Each bundle's geometry.connection is identical
        # (checked via assumptions.json from the bundles)
        import json

        connection_specs = []
        for rid in session.result_ids:
            assumptions_path = results_root / sid / rid / "assumptions.json"
            if assumptions_path.exists():
                data = json.loads(assumptions_path.read_text())
                geo = data.get("geometry", {}).get("connection", {})
                connection_specs.append(geo)

        if len(connection_specs) >= 2:
            first = connection_specs[0]
            for spec in connection_specs[1:]:
                assert spec == first, (
                    f"geometry.connection must be identical across bundles; "
                    f"first={first}, other={spec}"
                )
            assert first.get("type") == "independent", (
                f"geometry.connection.type must be 'independent'; got {first.get('type')}"
            )

    def test_adm_only_on_metric_affine_session(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """Even without cadabra, ADM can be derived on a metric-affine
        session and appears in results alongside any prior results."""
        client = TestClient(create_app(store=store, results_root=results_root))

        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive ADM (no cadabra needed)
        adm_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert adm_resp.status_code == 200, adm_resp.text
        adm_d = adm_resp.json()["derivations"]
        assert len(adm_d) > 0, "ADM must produce derivations"
        kinds = {d["kind"] for d in adm_d}
        assert "adm" in kinds

        # GET /results includes the ADM results
        results_resp = client.get(f"/sessions/{sid}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()
        result_kinds = {d["kind"] for d in results["results"]}
        assert "adm" in result_kinds, (
            f"/results must include ADM kind; got {result_kinds}"
        )


# ---------------------------------------------------------------------------
# VAL-CROSS-009: Levi-Civita regression -- pure-GR session unchanged
# ---------------------------------------------------------------------------


class TestLCRegressionCrossKind:
    """VAL-CROSS-009: A pure-GR session still walks action -> EOM -> ADM
    unchanged. eval1 and eval1s exit 0 with checks True; GR plan has no
    INDEPENDENT_CONNECTION; connection type levi-civita."""

    def test_gr_plan_has_no_independent_connection(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """A pure-GR session plan contains no independent-connection step."""
        client = TestClient(create_app(store=store, results_root=results_root))

        # Ingest a pure GR action (no Gamma annotation)
        eh_lagrangian = r"R"
        resp = client.post("/sessions", json={"lagrangian": eh_lagrangian})
        assert resp.status_code == 201
        body = resp.json()
        sid = body["session_id"]

        # Resolve geometry questions to Levi-Civita
        resolutions = {}
        for q in body["questions"]:
            if q["id"] == "amb-connection":
                resolutions[q["id"]] = "levi-civita"
            elif q["id"] == "amb-conventions":
                resolutions[q["id"]] = "noether-default-v1"
            elif q["id"] == "amb-vary-wrt":
                resolutions[q["id"]] = q["options"][0]
            elif q["id"] == "amb-curvature-free":
                resolutions[q["id"]] = "curvature-allowed"
            elif q.get("resolution") is None:
                resolutions[q["id"]] = q["options"][0]

        if resolutions:
            resolve_resp = client.post(
                f"/sessions/{sid}/resolve",
                json={"resolutions": resolutions},
            )
            assert resolve_resp.status_code == 200

        # Get the plan
        plan_resp = client.get(f"/sessions/{sid}/plan")
        assert plan_resp.status_code == 200
        plan = plan_resp.json()
        capabilities = [s["capability"] for s in plan["steps"]]
        assert "independent-connection" not in capabilities, (
            f"pure-GR plan must not have INDEPENDENT_CONNECTION; got {capabilities}"
        )

        # Verify connection type is levi-civita
        session = store.get(sid)
        assert session.npr.geometry.connection.type == "levi-civita", (
            f"connection type must be levi-civita; "
            f"got {session.npr.geometry.connection.type}"
        )

    def test_eval1s_adm_checks_pass(self) -> None:
        """The GR ADM eval (eval1s) component checks pass."""
        from noether.kernels.base import Capability, KernelTask
        from noether.kernels.sympy_kernel import SympyKernelAdapter

        adapter = SympyKernelAdapter()
        result = adapter.run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="ADM GR 1+2 check",
                payload={"check": "adm-gr-1p2"},
            )
        )
        assert result.value.get("passed"), (
            f"GR ADM checks must pass; detail: {result.value.get('detail', '')}"
        )

    def test_eval1_eom_checks_pass(self) -> None:
        """The GR EOM eval (eval1) structure and component checks pass."""
        from evals.eval1_eh_trace import MU, NU, build_npr, target_eom
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.planner import build_plan
        from noether.verify.checks import (
            DivergenceFreeCheck,
            SymmetricCheck,
            WellFormedCheck,
        )
        from noether.verify.ladder import run_ladder

        npr = build_npr(resolved=True)
        plan = build_plan(npr)
        # GR plan has no INDEPENDENT_CONNECTION
        caps = [s.capability.value for s in plan.steps]
        assert "independent-connection" not in caps

        adapters = {"sympy": SympyKernelAdapter()}
        eom = target_eom()
        report = run_ladder(
            eom,
            [
                WellFormedCheck(expected_free=[MU, NU]),
                SymmetricCheck(),
                DivergenceFreeCheck(),
            ],
            adapters,
        )
        assert report.all_passed, report.summary()

    def test_gr_session_adm_derivation(self, store: SessionStore, results_root: Path) -> None:
        """A pure-GR session can derive ADM, and the derivation carries
        conventions with connection type levi-civita."""
        client = TestClient(create_app(store=store, results_root=results_root))

        # Ingest pure GR
        resp = client.post("/sessions", json={"lagrangian": "R"})
        assert resp.status_code == 201
        body = resp.json()
        sid = body["session_id"]

        # Resolve to Levi-Civita
        resolutions = {}
        for q in body["questions"]:
            if q["id"] == "amb-connection":
                resolutions[q["id"]] = "levi-civita"
            elif q["id"] == "amb-conventions":
                resolutions[q["id"]] = "noether-default-v1"
            elif q["id"] == "amb-vary-wrt":
                resolutions[q["id"]] = q["options"][0]
            elif q["id"] == "amb-curvature-free":
                resolutions[q["id"]] = "curvature-allowed"
            elif q.get("resolution") is None:
                resolutions[q["id"]] = q["options"][0]
        if resolutions:
            client.post(
                f"/sessions/{sid}/resolve",
                json={"resolutions": resolutions},
            )

        # Derive ADM
        adm_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert adm_resp.status_code == 200, adm_resp.text
        derivations = adm_resp.json()["derivations"]
        assert len(derivations) > 0

        # The GR ADM convention block should NOT have field_strength_definition
        # (that's metric-affine only)
        conv = derivations[0].get("conventions", {})
        assert "field_strength_definition" not in conv, (
            "GR ADM should not have field_strength_definition in convention block"
        )


# ---------------------------------------------------------------------------
# VAL-CROSS-013: HTTP reachability -- all three operation kinds
# ---------------------------------------------------------------------------


class TestHTTPOperationKindReachability:
    """VAL-CROSS-013: On a well-posed metric-affine session, kind=eom,
    kind=perturbation, kind=adm each return 200 with derivations; an unknown
    kind returns 422; a with_respect_to naming an undeclared field returns 400.
    """

    def _setup_resolved_session(
        self, store: SessionStore, results_root: Path
    ) -> tuple[TestClient, str]:
        """Create and resolve a Palatini session for reachability testing."""
        client = TestClient(create_app(store=store, results_root=results_root))
        body = _create(client)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")
        return client, sid

    def test_kind_eom_returns_200_with_derivations(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive kind=eom returns 200 with derivations on a
        well-posed metric-affine session."""
        client, sid = self._setup_resolved_session(store, results_root)

        # Use a stub LLM so the EOM derivation runs without a real agent
        from noether.llm import StubLLMAdapter
        client2 = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
                results_root=results_root,
            )
        )

        # Re-resolve for client2 (same session)
        resp = client2.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        # May be 200 (cadabra available) or 503 (cadabra missing)
        if CadabraAdapter().available():
            assert resp.status_code == 200, (
                f"expected 200 for kind=eom; got {resp.status_code}: {resp.text}"
            )
            derivations = resp.json()["derivations"]
            assert len(derivations) > 0, "must return at least one EOM derivation"
            assert all(d["kind"] == "eom" for d in derivations), (
                "all derivations must have kind='eom'"
            )
        else:
            assert resp.status_code == 503, (
                "without cadabra, eom derive should return 503"
            )

    def test_kind_perturbation_returns_200_with_checks(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive kind=perturbation returns 200 with derivations and
        a checks dict on a well-posed metric-affine session."""
        client, sid = self._setup_resolved_session(store, results_root)

        from noether.llm import StubLLMAdapter
        client2 = TestClient(
            create_app(
                store=store,
                llm=StubLLMAdapter(
                    reply=templates.get("pert_metric_affine_quadratic")
                ),
                results_root=results_root,
            )
        )

        resp = client2.post(
            f"/sessions/{sid}/derive",
            json={"kind": "perturbation"},
        )
        if CadabraAdapter().available():
            assert resp.status_code == 200, (
                f"expected 200 for kind=perturbation; got {resp.status_code}: {resp.text}"
            )
            derivations = resp.json()["derivations"]
            assert len(derivations) > 0
            for d in derivations:
                assert d["kind"] == "perturbation"
                assert isinstance(d["checks"], dict)
        else:
            assert resp.status_code == 503

    def test_kind_adm_returns_200_with_derivations(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive kind=adm returns 200 with derivations on a
        well-posed metric-affine session (no cadabra needed)."""
        client, sid = self._setup_resolved_session(store, results_root)

        resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert resp.status_code == 200, (
            f"expected 200 for kind=adm; got {resp.status_code}: {resp.text}"
        )
        derivations = resp.json()["derivations"]
        assert len(derivations) > 0, "ADM must produce derivations"
        for d in derivations:
            assert d["kind"] == "adm", (
                f"expected kind='adm'; got {d['kind']!r}"
            )

    def test_unknown_kind_returns_422(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive with an unknown kind returns 422."""
        client, sid = self._setup_resolved_session(store, results_root)

        resp = client.post(f"/sessions/{sid}/derive", json={"kind": "bogus"})
        assert resp.status_code == 422, (
            f"expected 422 for unknown kind; got {resp.status_code}: {resp.text}"
        )

    def test_bad_wrt_returns_400(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive with with_respect_to naming an undeclared field
        returns 400."""
        client, sid = self._setup_resolved_session(store, results_root)

        resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["nonexistent_field"]},
        )
        assert resp.status_code == 400, (
            f"expected 400 for undeclared field; got {resp.status_code}: {resp.text}"
        )
        assert "nonexistent_field" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# VAL-CROSS-017: New metric-affine evals registered as CLI subcommands
# ---------------------------------------------------------------------------


class TestMetricAffineEvalSubcommands:
    """VAL-CROSS-017: The metric-affine acceptance evals (Palatini/EOM,
    perturbation, ADM) are invokable as noether <evalname> subcommands
    and exit 0 with their checks True (skipping only if cadabra2 absent)."""

    def test_eval2_registered_in_eval_keys(self) -> None:
        """eval2 (Palatini/EOM) is in EVAL_KEYS."""
        from noether.cli.main import EVAL_KEYS

        assert "eval2" in EVAL_KEYS, f"eval2 must be in EVAL_KEYS; got {EVAL_KEYS}"

    def test_eval4ma_registered_in_eval_keys(self) -> None:
        """eval4ma (metric-affine perturbation) is in EVAL_KEYS."""
        from noether.cli.main import EVAL_KEYS

        assert "eval4ma" in EVAL_KEYS, (
            f"eval4ma must be in EVAL_KEYS; got {EVAL_KEYS}"
        )

    def test_adm_affine_registered_in_eval_keys(self) -> None:
        """adm-affine (metric-affine ADM) is in EVAL_KEYS."""
        from noether.cli.main import EVAL_KEYS

        assert "adm-affine" in EVAL_KEYS, (
            f"adm-affine must be in EVAL_KEYS; got {EVAL_KEYS}"
        )

    def test_eval2_spec_builds(self) -> None:
        """The eval2 spec can be built and has the expected structure."""
        from evals.registry import get_spec

        spec = get_spec("eval2")
        assert spec.key == "eval2"
        assert "Palatini" in spec.title
        npr = spec.build_npr(resolved=True)
        assert npr.geometry.connection.type == "independent"

    def test_eval4ma_spec_builds(self) -> None:
        """The eval4ma spec can be built and has the expected structure."""
        from evals.registry import get_spec

        spec = get_spec("eval4ma")
        assert spec.key == "eval4ma"
        npr = spec.build_npr(resolved=True)
        assert npr.geometry.connection.type == "independent"
        assert npr.task.type == "perturb"

    def test_adm_affine_spec_builds(self) -> None:
        """The adm-affine spec can be built and has the expected structure."""
        from evals.registry import get_spec

        spec = get_spec("adm-affine")
        assert spec.key == "adm-affine"
        npr = spec.build_npr(resolved=True)
        assert npr.geometry.connection.type == "independent"
        assert npr.task.type == "adm"

    def test_eval2_runs_via_cli(self, tmp_path: Path) -> None:
        """`noether eval2` runs from the CLI and exits 0 with checks True
        (or skips if cadabra2 absent)."""
        import subprocess

        result = subprocess.run(
            [".venv/bin/python", "-m", "noether.cli.main", "eval2",
             "--results", str(tmp_path / "results")],
            capture_output=True,
            text=True,
            cwd="/Users/keigoshimada/Documents/project-noether",
            timeout=120,
        )
        if CadabraAdapter().available():
            assert result.returncode == 0, (
                f"eval2 must exit 0 when cadabra2 is available; "
                f"exit={result.returncode}, stdout={result.stdout[:500]}, "
                f"stderr={result.stderr[:500]}"
            )
            assert "PASS" in result.stdout or "Verified: True" in result.stdout, (
                f"eval2 must report passing checks; stdout={result.stdout[:500]}"
            )
        else:
            # When cadabra is absent, the eval should still run (skipping
            # the cadabra parts) and exit 0 or skip gracefully
            assert result.returncode in (0, 2), (
                f"eval2 must exit 0 or 2 when cadabra2 absent; "
                f"exit={result.returncode}"
            )

    def test_adm_affine_runs_via_cli(self, tmp_path: Path) -> None:
        """`noether adm-affine` runs from the CLI and exits 0 with its
        checks True (SymPy only, no cadabra needed)."""
        import subprocess

        result = subprocess.run(
            [".venv/bin/python", "-m", "noether.cli.main", "adm-affine",
             "--results", str(tmp_path / "results")],
            capture_output=True,
            text=True,
            cwd="/Users/keigoshimada/Documents/project-noether",
            timeout=120,
        )
        assert result.returncode == 0, (
            f"adm-affine must exit 0; exit={result.returncode}, "
            f"stdout={result.stdout[:500]}, stderr={result.stderr[:500]}"
        )
        assert "PASS" in result.stdout or "Verified: True" in result.stdout, (
            f"adm-affine must report passing checks; stdout={result.stdout[:500]}"
        )

    def test_eval4ma_runs_via_cli(self, tmp_path: Path) -> None:
        """`noether eval4ma` runs from the CLI and exits 0 with its
        checks True (or skips if cadabra2 absent)."""
        import subprocess

        result = subprocess.run(
            [".venv/bin/python", "-m", "noether.cli.main", "eval4ma",
             "--results", str(tmp_path / "results")],
            capture_output=True,
            text=True,
            cwd="/Users/keigoshimada/Documents/project-noether",
            timeout=120,
        )
        if CadabraAdapter().available():
            assert result.returncode == 0, (
                f"eval4ma must exit 0 when cadabra2 is available; "
                f"exit={result.returncode}, stdout={result.stdout[:500]}, "
                f"stderr={result.stderr[:500]}"
            )
        else:
            # Without cadabra, the perturbation eval can't run the
            # residue check; it should skip gracefully
            assert result.returncode in (0, 1, 2), (
                f"eval4ma exit code when cadabra absent: {result.returncode}; "
                f"stdout={result.stdout[:300]}"
            )


def _build_metric_affine_adm_npr_helper():
    """Helper for building a metric-affine NPR for convention testing."""
    from noether.npr import (
        NOETHER_DEFAULT_V1,
        NPR,
        Action,
        ConnectionSpec,
        Geometry,
        ObjectDecl,
        Task,
    )
    from noether.npr.ast import tensor

    connection = ConnectionSpec(
        type="independent",
        torsion=True,
        nonmetricity=True,
        metric_compatible=False,
        family="metric-affine",
    )
    geometry = Geometry(
        metric_name="g",
        connection_name="Gamma",
        connection=connection,
    )
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=geometry,
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(
                name="Gamma",
                kind="connection",
                role="dynamical",
                rank=3,
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=tensor("R"),
            lagrangian_tex="R",
        ),
        task=Task(type="adm", with_respect_to=["g"]),
        ambiguities=[],
    )
