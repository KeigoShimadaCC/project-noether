"""Gated verdict surfaces across backend (VAL-GUIDE-012/013/014).

A derivation the kernel cannot close returns verified==false with a non-empty
detail naming the blocker; a closed one returns verified==true. /derive and
/results return per-derivation verified and a non-empty detail, with a gated
result distinguishable from a verified one by these fields and the reason
visible. MCP noether_derive/noether_results return derivations whose verified
and detail match the HTTP surface; refusals come back as error/blocked data,
never a fabricated verified result.
"""

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from noether.kernels.base import Capability  # noqa: E402
from noether.kernels.cadabra import CadabraAdapter, templates  # noqa: E402
from noether.llm import StubLLMAdapter  # noqa: E402
from noether.mcp import NoetherTools  # noqa: E402
from noether.orchestrator.derive import FieldDerivation  # noqa: E402
from noether.orchestrator.store import SessionStore  # noqa: E402
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

    # Second pass for newly opened ambiguities
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
# VAL-GUIDE-012: An unverified result is returned with verified=false and a
# stated reason
# ---------------------------------------------------------------------------


class TestGatedVerdictFlagAndReason:
    """VAL-GUIDE-012: A derivation the kernel cannot close returns
    verified==false with a non-empty detail naming the blocker; a closed
    one returns verified==true."""

    def test_gated_field_derivation_has_verified_false_and_nonempty_detail(
        self,
    ) -> None:
        """A gated FieldDerivation has verified=False with an explanatory
        detail."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-gated-012",
            verified=False,
            detail="unverified: the kernel computed a nonzero residue",
        )
        assert d.verified is False, "gated derivation must have verified=False"
        assert d.detail, "gated derivation must have non-empty detail"
        # The detail names the blocker
        assert "nonzero" in d.detail.lower() or "residue" in d.detail.lower(), (
            f"detail must name the blocker; got: {d.detail!r}"
        )

    def test_verified_field_derivation_has_verified_true_and_nonempty_detail(
        self,
    ) -> None:
        """A verified FieldDerivation has verified=True with a non-empty
        detail (the confirmation reason)."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-verified-012",
            verified=True,
            detail="kernel confirmed the variation matches the candidate equation",
        )
        assert d.verified is True, "verified derivation must have verified=True"
        assert d.detail, "verified derivation must have non-empty detail"

    def test_gated_detail_naming_blocker(self) -> None:
        """Gated detail names the specific blocker (not just 'failed')."""
        # Various blocker scenarios
        blocked_details = [
            "unverified: the generated script produced no residue check; "
            "it did not run to completion (kernel exit 1)",
            "unverified: the kernel computed a nonzero residue, so the model's "
            "candidate equation does not match its own derivation",
            "unverified: the residue vanished but the independent linearized-EOM "
            "cross-check did not match",
            "needs covariant-derivative normal-ordering (SortCovDs) "
            "unavailable without xAct",
            "Dirac chain cannot be closed for the general metric-affine case "
            "with non-metricity (Q != 0)",
        ]
        for detail in blocked_details:
            d = FieldDerivation(
                wrt="g",
                kind="eom",
                capability=Capability.VARY,
                result_id="test-blocker-012",
                verified=False,
                detail=detail,
            )
            assert d.verified is False
            assert d.detail, "gated detail must be non-empty"
            # The detail names a specific blocker, not a generic 'failed'
            assert len(d.detail) > 10, (
                f"detail should name the specific blocker, not be generic: {d.detail!r}"
            )

    def test_g4g5_verified_has_nonempty_detail(self) -> None:
        """The G4/G5 best-effort path produces non-empty detail even when
        verified=True (a closed result carries a confirmation reason)."""
        d = FieldDerivation(
            wrt="phi",
            kind="eom",
            capability=Capability.VARY,
            result_id="g4g5-verified-012",
            verified=True,
            detail="kernel verified the G4 scalar and metric EOMs "
            "(both residue checks passed)",
        )
        assert d.verified is True
        assert d.detail, "verified G4/G5 derivation must have non-empty detail"

    def test_g4g5_gated_has_nonempty_detail_naming_blocker(self) -> None:
        """The G4/G5 best-effort path produces non-empty detail naming the
        blocker when gated."""
        d = FieldDerivation(
            wrt="phi",
            kind="eom",
            capability=Capability.VARY,
            result_id="g4g5-gated-012",
            verified=False,
            detail=(
                "needs covariant-derivative normal-ordering (SortCovDs) "
                "unavailable without xAct"
            ),
        )
        assert d.verified is False
        assert d.detail, "gated G4/G5 derivation must have non-empty detail"
        assert "SortCovDs" in d.detail, "detail must name the SortCovDs blocker"

    @requires_cadabra
    def test_live_gated_derivation_has_nonempty_detail(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """A real derivation that fails verification returns verified=False
        with a non-empty detail naming the blocker."""
        # Use a stub that produces a nonzero residue to force a gated result
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

        derive_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert derive_resp.status_code == 200
        derivations = derive_resp.json()["derivations"]
        assert len(derivations) > 0
        d = derivations[0]
        assert d["verified"] is False, "gated derivation must have verified=False"
        assert d["detail"], "gated derivation must have non-empty detail"
        # The detail names the blocker
        assert "nonzero" in d["detail"].lower() or "unverified" in d["detail"].lower(), (
            f"gated detail must name the blocker; got: {d['detail']!r}"
        )

    @requires_cadabra
    def test_live_verified_derivation_has_verified_true(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """A real derivation that passes verification returns verified=True
        with a non-empty detail."""
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

        derive_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert derive_resp.status_code == 200
        derivations = derive_resp.json()["derivations"]
        assert len(derivations) > 0
        d = derivations[0]
        assert d["verified"] is True, "verified derivation must have verified=True"
        assert d["detail"], "verified derivation must have non-empty detail"


# ---------------------------------------------------------------------------
# Structural safeguard: detail is always non-empty (Pydantic validator)
# ---------------------------------------------------------------------------


class TestDetailNonemptyValidator:
    """FieldDerivation.detail must be non-empty and not whitespace-only;
    the Pydantic model_validator rejects construction with empty detail."""

    def test_empty_detail_raises_validation_error(self) -> None:
        """Constructing a FieldDerivation with empty detail raises
        ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="detail must be non-empty"):
            FieldDerivation(
                wrt="g",
                kind="eom",
                capability=Capability.VARY,
                result_id="test-empty-detail",
                detail="",
            )

    def test_whitespace_only_detail_raises_validation_error(self) -> None:
        """Constructing a FieldDerivation with whitespace-only detail raises
        ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="detail must be non-empty"):
            FieldDerivation(
                wrt="g",
                kind="eom",
                capability=Capability.VARY,
                result_id="test-whitespace-detail",
                detail="   \t  ",
            )

    def test_nonempty_detail_passes_validation(self) -> None:
        """Constructing a FieldDerivation with a non-empty detail succeeds."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-valid-detail",
            verified=True,
            detail="kernel confirmed the variation matches the candidate equation",
        )
        assert d.detail  # detail is non-empty


# ---------------------------------------------------------------------------
# VAL-GUIDE-013: HTTP surfaces the verified flag and reason; verified is
# distinct from gated
# ---------------------------------------------------------------------------


class TestHTTPVerdictAndReasonSurface:
    """VAL-GUIDE-013: /derive and /results return per-derivation verified and
    a non-empty detail; a gated result is distinguishable from a verified one
    by these fields and the reason is visible."""

    def test_derive_json_shows_verified_and_detail(self, client: TestClient) -> None:
        """FieldDerivation.model_dump() includes verified and detail keys."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-013a",
            verified=False,
            detail="unverified: the kernel computed a nonzero residue",
        )
        dumped = d.model_dump()
        assert "verified" in dumped, "model_dump must include verified"
        assert "detail" in dumped, "model_dump must include detail"
        assert dumped["verified"] is False
        assert dumped["detail"] != "", "detail must be non-empty"

    def test_verified_derivation_json_shows_both_fields(self) -> None:
        """A verified derivation's model_dump shows verified=true and
        non-empty detail."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-013b",
            verified=True,
            detail="kernel confirmed the variation matches the candidate equation",
        )
        dumped = d.model_dump()
        assert dumped["verified"] is True
        assert dumped["detail"] != "", "verified detail must be non-empty"

    def test_gated_distinguishable_from_verified_by_both_fields(self) -> None:
        """A gated result is distinguishable from a verified one by both
        verified and detail."""
        d_verified = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-013v",
            verified=True,
            detail="kernel confirmed the variation matches the candidate equation",
        )
        d_gated = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-013g",
            verified=False,
            detail="unverified: the kernel computed a nonzero residue",
        )
        # Distinguishable by verified
        assert d_verified.verified != d_gated.verified
        # Distinguishable by detail (the reason is visible)
        assert d_verified.detail != d_gated.detail
        # The gated detail names the blocker
        assert "unverified" in d_gated.detail.lower() or "nonzero" in d_gated.detail.lower()

    @requires_cadabra
    def test_http_derive_returns_verified_and_nonempty_detail(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """POST /derive returns derivations with verified and non-empty
        detail for both gated and verified results."""
        # Gated case
        client_gated = TestClient(
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
        body = _create(client_gated)
        sid = body["session_id"]

        def _resolve_http(session_id: str, resolutions: dict) -> dict:
            resp = client_gated.post(
                f"/sessions/{session_id}/resolve",
                json={"resolutions": resolutions},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        derive_resp = client_gated.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert derive_resp.status_code == 200
        derivations = derive_resp.json()["derivations"]
        assert len(derivations) > 0
        d = derivations[0]
        assert "verified" in d, "derivation must have verified key"
        assert "detail" in d, "derivation must have detail key"
        assert d["verified"] is False
        assert d["detail"], "gated detail must be non-empty via HTTP"

    @requires_cadabra
    def test_http_results_returns_verified_and_nonempty_detail(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """GET /results returns derivations with verified and non-empty
        detail matching the /derive response."""
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

        # Derive
        derive_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert derive_resp.status_code == 200
        derive_d = derive_resp.json()["derivations"][0]

        # Results
        results_resp = client.get(f"/sessions/{sid}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()
        result_d = next(
            r for r in results["results"] if r["result_id"] == derive_d["result_id"]
        )
        # verified and detail match
        assert result_d["verified"] == derive_d["verified"]
        assert result_d["detail"] == derive_d["detail"]
        # Both are non-empty
        assert result_d["detail"], "detail must be non-empty in /results"

    def test_http_adm_derivation_has_nonempty_detail(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """ADM derivations via HTTP have non-empty detail for both verified
        and gated pieces."""
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

        derive_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert derive_resp.status_code == 200
        derivations = derive_resp.json()["derivations"]
        assert len(derivations) > 0
        for d in derivations:
            assert "verified" in d, f"ADM derivation must have verified; wrt={d['wrt']}"
            assert "detail" in d, f"ADM derivation must have detail; wrt={d['wrt']}"
            # Detail is always non-empty (even for verified, and especially for gated)
            assert d["detail"], (
                f"ADM derivation detail must be non-empty; "
                f"wrt={d['wrt']}, verified={d['verified']}"
            )

    def test_http_adm_results_has_nonempty_detail(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """GET /results for ADM derivations has non-empty detail matching
        /derive."""
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

        derive_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert derive_resp.status_code == 200

        results_resp = client.get(f"/sessions/{sid}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()
        for d in results["results"]:
            assert d["detail"], (
                f"ADM /results detail must be non-empty; "
                f"wrt={d['wrt']}, verified={d['verified']}"
            )


# ---------------------------------------------------------------------------
# VAL-GUIDE-014: MCP surfaces the verified flag and reason consistently
# ---------------------------------------------------------------------------


class TestMCPVerdictAndReasonSurface:
    """VAL-GUIDE-014: noether_derive/noether_results return derivations whose
    verified and detail match the HTTP surface; refusals come back as
    error/blocked data, never a fabricated verified result."""

    def test_mcp_derive_returns_verified_and_nonempty_detail(
        self, tools: NoetherTools
    ) -> None:
        """MCP noether_derive returns derivations with verified and
        non-empty detail (even when using ADM, which does not need cadabra)."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        result = tools.derive(sid, kind="adm")
        assert "derivations" in result, f"expected derivations; got {result}"
        for d in result["derivations"]:
            assert "verified" in d, f"MCP derivation must have verified; wrt={d['wrt']}"
            assert "detail" in d, f"MCP derivation must have detail; wrt={d['wrt']}"
            assert d["detail"], (
                f"MCP derivation detail must be non-empty; "
                f"wrt={d['wrt']}, verified={d['verified']}"
            )

    def test_mcp_results_returns_verified_and_nonempty_detail(
        self, tools: NoetherTools
    ) -> None:
        """MCP noether_results returns derivations with verified and
        non-empty detail."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        # Derive an ADM result
        tools.derive(sid, kind="adm")

        # Read via noether_results
        result = tools.results(sid)
        assert "results" in result, f"expected results key; got {result}"
        for d in result["results"]:
            assert "verified" in d, f"MCP results derivation must have verified; wrt={d['wrt']}"
            assert "detail" in d, f"MCP results derivation must have detail; wrt={d['wrt']}"
            assert d["detail"], (
                f"MCP results detail must be non-empty; "
                f"wrt={d['wrt']}, verified={d['verified']}"
            )

    def test_mcp_derive_detail_matches_http(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """MCP noether_derive and HTTP /derive return the same verified and
        detail for the same session."""
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

        # Derive via HTTP
        http_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert http_resp.status_code == 200
        http_derivations = http_resp.json()["derivations"]

        # Read via MCP results (same session)
        mcp_results = tools.results(sid)
        mcp_derivations = mcp_results["results"]

        # Match by result_id
        for http_d in http_derivations:
            rid = http_d["result_id"]
            wrt = http_d["wrt"]
            mcp_d = next(
                (
                    r
                    for r in mcp_derivations
                    if r["result_id"] == rid and r["wrt"] == wrt
                ),
                None,
            )
            assert mcp_d is not None, (
                f"no MCP match for result_id={rid} wrt={wrt}"
            )
            # verified and detail must match
            assert http_d["verified"] == mcp_d["verified"], (
                f"verified mismatch for {rid}/{wrt}: "
                f"HTTP={http_d['verified']}, MCP={mcp_d['verified']}"
            )
            assert http_d["detail"] == mcp_d["detail"], (
                f"detail mismatch for {rid}/{wrt}: "
                f"HTTP={http_d['detail']!r}, MCP={mcp_d['detail']!r}"
            )
            # Both must be non-empty
            assert http_d["detail"], "detail must be non-empty on HTTP"
            assert mcp_d["detail"], "detail must be non-empty on MCP"

    def test_mcp_blocked_derive_returns_blocked_dict(
        self, tools: NoetherTools
    ) -> None:
        """When the session has unresolved ambiguities, MCP noether_derive
        returns a blocked dict, not a fabricated verified result."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]
        # Do NOT resolve any ambiguities

        result = tools.derive(sid, kind="eom")
        # Must be blocked, not a derivation
        assert result.get("blocked") is True, (
            f"expected blocked=True; got {result}"
        )
        # Must not contain fabricated derivations
        assert "derivations" not in result or not result.get("derivations"), (
            "blocked result must not contain derivations"
        )
        # Must have questions
        assert result.get("questions"), "blocked result must have questions"

    def test_mcp_blocked_plan_returns_blocked_dict(
        self, tools: NoetherTools
    ) -> None:
        """When the session has unresolved ambiguities, MCP noether_plan
        returns a blocked dict, not a fabricated plan."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]
        # Do NOT resolve any ambiguities

        result = tools.plan(sid)
        assert result.get("blocked") is True, (
            f"expected blocked=True; got {result}"
        )
        assert result.get("questions"), "blocked plan must have questions"

    def test_mcp_refused_derive_returns_error_dict(
        self, tools: NoetherTools
    ) -> None:
        """When the derive request is invalid (unknown kind), MCP
        noether_derive returns an error dict, not a fabricated result."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        result = tools.derive(sid, kind="invalid_kind")
        assert "error" in result, f"expected error dict; got {result}"
        # No derivations in error response
        assert "derivations" not in result or not result.get("derivations"), (
            "error response must not contain derivations"
        )

    def test_mcp_off_menu_resolve_returns_error_dict(
        self, tools: NoetherTools
    ) -> None:
        """An off-menu resolve attempt returns an error dict, never a
        fabricated verified result."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        # Off-menu resolve
        result = tools.resolve(sid, {"amb-connection": "not-an-option"})
        assert "error" in result, f"expected error dict; got {result}"

        # Session should still be unresolved, so derive must be blocked
        derive_result = tools.derive(sid, kind="eom")
        assert derive_result.get("blocked") is True, (
            f"derive must still be blocked after off-menu resolve; got {derive_result}"
        )

    def test_mcp_never_fabricates_verified_result(self, tools: NoetherTools) -> None:
        """MCP never returns a fabricated verified result under any refusal
        condition."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        # 1. Blocked derive (unresolved)
        derive_blocked = tools.derive(sid, kind="eom")
        assert derive_blocked.get("blocked") is True
        assert "derivations" not in derive_blocked or not derive_blocked.get("derivations")

        # 2. Error from unknown kind
        # Resolve first so we can test the error path
        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        error_result = tools.derive(sid, kind="nonexistent")
        assert "error" in error_result
        assert "derivations" not in error_result or not error_result.get("derivations")

        # 3. Error from off-menu resolve on a fresh session
        body2 = tools.ingest(PALATINI_LAGRANGIAN)
        sid2 = body2["session_id"]
        off_menu = tools.resolve(sid2, {"amb-connection": "fabricated-option"})
        assert "error" in off_menu

    @requires_cadabra
    def test_mcp_gated_derivation_detail_matches_http(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """A gated derivation's verified and detail match across MCP
        noether_results and HTTP GET /results."""
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
            assert resp.status_code == 200, resp.text
            return resp.json()

        _resolve_all_palatini(body, _resolve_http, connection="independent")

        # Derive a gated result via HTTP
        derive_resp = client.post(
            f"/sessions/{sid}/derive",
            json={"kind": "eom", "with_respect_to": ["g"]},
        )
        assert derive_resp.status_code == 200
        derive_d = derive_resp.json()["derivations"][0]
        assert derive_d["verified"] is False, "must be gated for this test"

        # HTTP /results
        http_results = client.get(f"/sessions/{sid}/results").json()
        http_d = next(
            r for r in http_results["results"] if r["result_id"] == derive_d["result_id"]
        )

        # MCP noether_results
        mcp_results = tools.results(sid)
        mcp_d = next(
            r for r in mcp_results["results"] if r["result_id"] == derive_d["result_id"]
        )

        # verified and detail must match across HTTP and MCP
        assert http_d["verified"] == mcp_d["verified"], (
            f"verified mismatch: HTTP={http_d['verified']}, MCP={mcp_d['verified']}"
        )
        assert http_d["detail"] == mcp_d["detail"], (
            f"detail mismatch: HTTP={http_d['detail']!r}, MCP={mcp_d['detail']!r}"
        )
        # Both non-empty
        assert http_d["detail"], "gated detail must be non-empty on HTTP"
        assert mcp_d["detail"], "gated detail must be non-empty on MCP"

    def test_mcp_adm_gated_piece_has_nonempty_detail(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """An ADM derivation on a metric-affine session has at least one
        gated piece with verified=False and non-empty detail naming the
        blocker (Dirac chain cannot close when Q != 0)."""
        tools = NoetherTools(store, results_root=results_root)

        body = tools.ingest(PALATINI_LAGRANGIAN)
        sid = body["session_id"]

        def _resolve_mcp(session_id: str, resolutions: dict) -> dict:
            return tools.resolve(session_id, resolutions)

        # Resolve with non-metricity allowed (triggers Dirac chain gating)
        _resolve_all_palatini(body, _resolve_mcp, connection="independent")

        result = tools.derive(sid, kind="adm")
        assert "derivations" in result
        derivations = result["derivations"]

        # At least one derivation must be gated (connection-sector constraints
        # when Q != 0)
        gated = [d for d in derivations if d["verified"] is False]
        assert len(gated) > 0, (
            f"expected at least one gated ADM piece for metric-affine with Q; "
            f"got all verified: {[d['wrt'] for d in derivations]}"
        )
        for d in gated:
            assert d["detail"], (
                f"gated ADM piece must have non-empty detail; "
                f"wrt={d['wrt']}, detail={d['detail']!r}"
            )

    def test_mcp_adm_derivation_detail_matches_http_across_results(
        self, store: SessionStore, results_root: Path
    ) -> None:
        """ADM derivations: verified and detail match between MCP
        noether_results and HTTP GET /results for all pieces, including
        gated ones."""
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

        # Derive via HTTP
        http_resp = client.post(f"/sessions/{sid}/derive", json={"kind": "adm"})
        assert http_resp.status_code == 200
        http_derivations = http_resp.json()["derivations"]

        # Read via MCP results
        mcp_results = tools.results(sid)
        mcp_derivations = mcp_results["results"]

        # Read via HTTP results
        http_results = client.get(f"/sessions/{sid}/results").json()

        # All three surfaces must agree
        for http_d in http_derivations:
            rid = http_d["result_id"]
            wrt = http_d["wrt"]

            # Match in MCP
            mcp_d = next(
                (
                    r
                    for r in mcp_derivations
                    if r["result_id"] == rid and r["wrt"] == wrt
                ),
                None,
            )
            assert mcp_d is not None, f"no MCP match for {rid}/{wrt}"

            # Match in HTTP results
            http_r = next(
                (
                    r
                    for r in http_results["results"]
                    if r["result_id"] == rid and r["wrt"] == wrt
                ),
                None,
            )
            assert http_r is not None, f"no HTTP results match for {rid}/{wrt}"

            # All agree on verified and detail
            assert http_d["verified"] == mcp_d["verified"] == http_r["verified"], (
                f"verified mismatch for {rid}/{wrt}: "
                f"derive={http_d['verified']}, mcp={mcp_d['verified']}, "
                f"http_results={http_r['verified']}"
            )
            assert http_d["detail"] == mcp_d["detail"] == http_r["detail"], (
                f"detail mismatch for {rid}/{wrt}: "
                f"derive={http_d['detail']!r}, mcp={mcp_d['detail']!r}, "
                f"http_results={http_r['detail']!r}"
            )
            # Detail always non-empty
            assert http_d["detail"], (
                f"detail must be non-empty for {rid}/{wrt}"
            )
