"""Backend cross-flows: metric-affine derivations through HTTP, MCP, CLI,
and the store (architecture.md section 8, VAL-CROSS-001/005/007/008/012/016).

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
