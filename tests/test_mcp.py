"""MCP adapter: same session surface, no-guessing contract as tool results.

Skips cleanly when the [mcp] extra is not installed. Tool logic is tested
through NoetherTools (plain methods); the FastMCP wrapper is checked for the
expected tool registry.
"""

import asyncio

import pytest

pytest.importorskip("mcp")

from noether.kernels.cadabra import CadabraAdapter, templates  # noqa: E402
from noether.llm import StubLLMAdapter  # noqa: E402
from noether.mcp import NoetherTools, create_mcp_server  # noqa: E402
from noether.orchestrator.store import SessionStore  # noqa: E402

requires_cadabra = pytest.mark.skipif(
    not CadabraAdapter().available(), reason="cadabra2 not installed"
)


@pytest.fixture()
def tools(tmp_path):
    return NoetherTools(SessionStore(tmp_path / "sessions"))


class TestTools:
    def test_kernels(self, tools):
        kernels = tools.kernels()
        assert kernels["sympy"]["available"] is True

    def test_ingest_returns_questions(self, tools):
        body = tools.ingest("R")
        assert body["well_posed"] is False
        assert body["questions"]
        assert tools.session(body["session_id"])["session_id"] == body["session_id"]
        assert body["session_id"] in tools.sessions()["sessions"]

    def test_parse_error_is_data(self, tools):
        assert "error" in tools.ingest(r"R_{\mu")

    def test_unknown_session_is_data(self, tools):
        assert "error" in tools.session("s-doesnotexist")
        assert "error" in tools.plan("s-doesnotexist")
        assert "error" in tools.resolve("s-doesnotexist", {"amb-x": "y"})

    def test_plan_blocked_until_resolved(self, tools):
        body = tools.ingest("R")
        blocked = tools.plan(body["session_id"])
        assert blocked["blocked"] is True and blocked["questions"]

    def test_off_menu_resolution_rejected(self, tools):
        body = tools.ingest("R")
        question = body["questions"][0]
        result = tools.resolve(body["session_id"], {question["id"]: "not-an-option"})
        assert "error" in result
        # rejection must not have mutated the session
        assert tools.session(body["session_id"])["well_posed"] is False

    def test_confirmed_resolutions_unblock_plan(self, tools):
        body = tools.ingest("R")
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        resolved = tools.resolve(body["session_id"], resolutions)
        assert resolved["well_posed"] is True
        plan = tools.plan(body["session_id"])
        assert plan["blocked"] is False
        assert plan["task_type"] == "vary"

    def test_store_shared_with_other_frontends(self, tmp_path):
        store = SessionStore(tmp_path / "sessions")
        body = NoetherTools(store).ingest("R")
        again = NoetherTools(store).session(body["session_id"])
        assert again["questions"] == body["questions"]


SCALAR_TENSOR = r"F(\phi) R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)"
MAXWELL = r"-\tfrac14 F_{\mu\nu} F^{\mu\nu}"


class TestDefinitionTools:
    def test_propose_and_adopt(self, tools):
        body = tools.ingest(SCALAR_TENSOR)
        sid = body["session_id"]
        proposals = tools.propose_definitions(sid)
        assert proposals["confirmed"] is False
        assert "F_phi" in {p["symbol"] for p in proposals["proposals"]}
        adopted = tools.adopt_definitions(sid, ["def-F-phi"])
        assert "F_phi" in {o["name"] for o in adopted["objects"]}

    def test_unknown_definition_is_data(self, tools):
        body = tools.ingest(SCALAR_TENSOR)
        assert "error" in tools.adopt_definitions(body["session_id"], ["def-nope"])

    def test_empty_accept_is_data(self, tools):
        body = tools.ingest(SCALAR_TENSOR)
        assert "error" in tools.adopt_definitions(body["session_id"], [])


def _well_posed_scalar_tensor(tools) -> str:
    body = tools.ingest(SCALAR_TENSOR)
    resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
    resolved = tools.resolve(body["session_id"], resolutions)
    assert resolved["well_posed"] is True
    return body["session_id"]


@requires_cadabra
class TestDeriveTools:
    def _tools(self, tmp_path):
        return NoetherTools(
            SessionStore(tmp_path / "sessions"),
            llm=StubLLMAdapter(reply=templates.get("eval3_scalar_tensor_metric")),
            results_root=tmp_path / "results",
        )

    def test_derive_returns_verified_eom(self, tmp_path):
        tools = self._tools(tmp_path)
        sid = _well_posed_scalar_tensor(tools)
        result = tools.derive(sid)
        derivations = result["derivations"]
        assert [d["wrt"] for d in derivations] == ["g"]
        assert derivations[0]["verified"] is True
        assert derivations[0]["result_tex"]

    def test_derive_blocked_until_resolved(self, tmp_path):
        tools = self._tools(tmp_path)
        body = tools.ingest(SCALAR_TENSOR)
        blocked = tools.derive(body["session_id"])
        assert blocked["blocked"] is True and blocked["questions"]

    def test_derive_undeclared_field_is_data(self, tmp_path):
        tools = self._tools(tmp_path)
        sid = _well_posed_scalar_tensor(tools)
        assert "error" in tools.derive(sid, ["not_a_field"])

    def test_unknown_session_is_data(self, tmp_path):
        tools = self._tools(tmp_path)
        assert "error" in tools.derive("s-doesnotexist")

    def test_unknown_kind_is_data(self, tmp_path):
        tools = self._tools(tmp_path)
        sid = _well_posed_scalar_tensor(tools)
        assert "error" in tools.derive(sid, kind="bogus")

    def test_perturbation_returns_verified_quadratic_action(self, tmp_path):
        tools = NoetherTools(
            SessionStore(tmp_path / "sessions"),
            llm=StubLLMAdapter(reply=templates.get("pert_scalar_quadratic")),
            results_root=tmp_path / "results",
        )
        sid = _well_posed_scalar_tensor(tools)
        result = tools.derive(sid, ["phi"], kind="perturbation")
        derivations = result["derivations"]
        assert [d["wrt"] for d in derivations] == ["phi"]
        assert derivations[0]["kind"] == "perturbation"
        assert derivations[0]["verified"] is True

    def test_perturbation_metric_returns_verified_quadratic_action(self, tmp_path):
        tools = NoetherTools(
            SessionStore(tmp_path / "sessions"),
            llm=StubLLMAdapter(reply=templates.get("pert_metric_quadratic")),
            results_root=tmp_path / "results",
        )
        sid = _well_posed_scalar_tensor(tools)
        result = tools.derive(sid, ["g"], kind="perturbation")
        g = result["derivations"][0]
        assert g["wrt"] == "g"
        assert g["kind"] == "perturbation"
        assert g["verified"] is True

    def test_perturbation_refuses_unsupported_field(self, tmp_path):
        tools = NoetherTools(
            SessionStore(tmp_path / "sessions"),
            llm=StubLLMAdapter(reply=templates.get("pert_scalar_quadratic")),
            results_root=tmp_path / "results",
        )
        body = tools.ingest(MAXWELL)
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        assert tools.resolve(body["session_id"], resolutions)["well_posed"] is True
        assert "error" in tools.derive(body["session_id"], ["F"], kind="perturbation")


class TestAdmTools:
    """ADM is verified by the SymPy component kernel, so the MCP tool needs
    neither cadabra nor an LLM backend."""

    def _tools(self, tmp_path):
        return NoetherTools(SessionStore(tmp_path / "sessions"), results_root=tmp_path / "results")

    def _well_posed(self, tools, lagrangian) -> str:
        body = tools.ingest(lagrangian)
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        assert tools.resolve(body["session_id"], resolutions)["well_posed"] is True
        return body["session_id"]

    def test_adm_returns_verified_decomposition(self, tmp_path):
        tools = self._tools(tmp_path)
        sid = self._well_posed(tools, "R")
        result = tools.derive(sid, kind="adm")
        derivations = result["derivations"]
        assert len(derivations) == 3
        assert all(d["kind"] == "adm" and d["verified"] for d in derivations)
        assert derivations[0]["checks"]["lagrangian-split"] == "True"

    def test_unknown_kind_is_data(self, tmp_path):
        tools = self._tools(tmp_path)
        sid = self._well_posed(tools, "R")
        assert "error" in tools.derive(sid, kind="bogus")

    def test_results_record_reload_dedupe_and_stale(self, tmp_path):
        tools = self._tools(tmp_path)
        body = tools.ingest("R")
        sid = body["session_id"]
        resolutions = {q["id"]: q["options"][0] for q in body["questions"]}
        assert tools.resolve(sid, resolutions)["well_posed"] is True
        assert tools.results(sid) == {"session_id": sid, "results": [], "stale_result_ids": []}
        tools.derive(sid, kind="adm")
        tools.derive(sid, kind="adm")  # repeat must not duplicate history
        recorded = tools.results(sid)
        assert len(recorded["results"]) == 3
        assert recorded["stale_result_ids"] == []
        # a resolution after results exist marks them stale
        first = body["questions"][0]
        tools.resolve(sid, {first["id"]: first["options"][0]})
        stale = tools.results(sid)
        assert stale["stale_result_ids"] == [stale["results"][0]["result_id"]]

    def test_unknown_session_results_is_data(self, tmp_path):
        tools = self._tools(tmp_path)
        assert "error" in tools.results("s-doesnotexist")


class TestServerWiring:
    def test_expected_tools_registered(self, tmp_path):
        server = create_mcp_server(SessionStore(tmp_path / "sessions"))
        names = {t.name for t in asyncio.run(server.list_tools())}
        assert names == {
            "noether_kernels",
            "noether_ingest",
            "noether_sessions",
            "noether_session",
            "noether_resolve",
            "noether_propose_definitions",
            "noether_adopt_definitions",
            "noether_plan",
            "noether_derive",
            "noether_results",
        }


# ---------------------------------------------------------------------------
# VAL-EOM-008: MCP noether_derive reaches the connection variation and
# refuses when ambiguous
# ---------------------------------------------------------------------------

PALATINI_LAGRANGIAN = r"g^{\mu\nu} R_{\mu\nu}(\Gamma)"


def _resolve_all_geometry(body, tools, *, connection="independent", torsion="torsion-allowed",
                          nonmetricity="nonmetricity-allowed",
                          metric_compat="not-metric-compatible",
                          conventions="noether-default-v1",
                          vary_wrt="g and Gamma",
                          ricci_contraction="first-third",
                          field_strength_definition="exterior-derivative"):
    """Resolve all ambiguities for a Palatini-style session, returning the
    updated session payload.

    After resolving the connection to 'independent', a new
    amb-ricci-contraction ambiguity is opened, so this helper does a
    two-pass resolution. If a vector potential is present, the
    amb-field-strength-definition ambiguity is also opened."""
    sid = body["session_id"]
    resolutions = {}
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
            resolutions[q["id"]] = conventions
        elif q["id"] == "amb-vary-wrt":
            # Use the compound option when available, else the first option
            if vary_wrt in q["options"]:
                resolutions[q["id"]] = vary_wrt
            else:
                resolutions[q["id"]] = q["options"][0]
    result = tools.resolve(sid, resolutions)

    # Second pass: resolve any ambiguities opened by the first pass
    # (e.g. amb-ricci-contraction when connection=independent,
    #  amb-field-strength-definition when a vector potential is present).
    remaining = {}
    for q in result.get("questions", []):
        if q.get("resolution") is None:
            if q["id"] == "amb-ricci-contraction":
                remaining[q["id"]] = ricci_contraction
            elif q["id"] == "amb-field-strength-definition":
                remaining[q["id"]] = field_strength_definition
            else:
                remaining[q["id"]] = q["options"][0]
    if remaining:
        result = tools.resolve(sid, remaining)
    return result


class TestPalatiniMcpReachability:
    """VAL-EOM-008: MCP noether_derive reaches the connection EOM on a
    resolved Palatini session and returns a refusal (blocked dict) when
    the connection ambiguity is still open."""

    def test_ingest_palatini_adds_gamma_object(self, tools):
        """Ingesting a Palatini action must add Gamma as a declared
        connection object so it can appear in with_respect_to."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        object_names = {o["name"] for o in body["objects"]}
        assert "Gamma" in object_names, (
            f"Gamma must be a declared object after ingesting a Palatini "
            f"action; got {object_names}"
        )
        gamma_obj = next(o for o in body["objects"] if o["name"] == "Gamma")
        assert gamma_obj["kind"] == "connection"

    def test_ingest_palatini_raises_connection_question(self, tools):
        """Ingesting a Palatini action must raise the connection-type
        ambiguity with 'independent' as an option."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        conn_q = next(
            (q for q in body["questions"] if q["id"] == "amb-connection"), None
        )
        assert conn_q is not None, "amb-connection question must be raised"
        assert "independent" in conn_q["options"]

    def test_ingest_palatini_vary_wrt_includes_compound_option(self, tools):
        """When both g and Gamma are present, amb-vary-wrt must offer a
        compound 'g and Gamma' option."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        vary_q = next(
            (q for q in body["questions"] if q["id"] == "amb-vary-wrt"), None
        )
        assert vary_q is not None
        assert "g and Gamma" in vary_q["options"], (
            f"'g and Gamma' must be an option; got {vary_q['options']}"
        )

    def test_plan_blocked_while_connection_ambiguity_open(self, tools):
        """VAL-EOM-018: while the connection question is unresolved,
        noether_plan must return blocked=true with the questions, not a
        guess."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        plan = tools.plan(body["session_id"])
        assert plan["blocked"] is True
        assert plan["questions"]
        question_ids = {q for q in plan["questions"]}
        assert "amb-connection" in question_ids or any(
            "connection" in q for q in plan["questions"]
        )

    def test_derive_blocked_while_connection_ambiguity_open(self, tools):
        """VAL-EOM-008: on a session with the connection ambiguity open,
        noether_derive must return a refusal (blocked dict), not a guess."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        result = tools.derive(body["session_id"], ["g", "Gamma"])
        assert result.get("blocked") is True, (
            f"derive must return blocked=true while ambiguities are open; "
            f"got {result}"
        )
        assert result.get("questions"), "blocked derive must include questions"
        # No derivations produced
        assert "derivations" not in result or not result.get("derivations")

    def test_resolve_independent_enables_independent_connection_step(self, tools):
        """VAL-EOM-018: resolving the connection to 'independent' enables
        the INDEPENDENT_CONNECTION plan step."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        resolved = _resolve_all_geometry(body, tools)
        assert resolved["well_posed"] is True
        plan = tools.plan(body["session_id"])
        assert plan["blocked"] is False
        capabilities = [s["capability"] for s in plan["steps"]]
        assert "independent-connection" in capabilities, (
            f"plan must include independent-connection step; "
            f"got capabilities {capabilities}"
        )

    def test_resolve_levi_civita_no_independent_connection_step(self, tools):
        """Resolving to levi-civita must not include the
        INDEPENDENT_CONNECTION step."""
        body = tools.ingest(PALATINI_LAGRANGIAN)
        resolved = _resolve_all_geometry(body, tools, connection="levi-civita")
        assert resolved["well_posed"] is True
        plan = tools.plan(body["session_id"])
        capabilities = [s["capability"] for s in plan["steps"]]
        assert "independent-connection" not in capabilities

    @requires_cadabra
    def test_derive_resolved_palatini_returns_both_eoms(self, tmp_path):
        """VAL-EOM-008: on a resolved Palatini session, noether_derive with
        with_respect_to=['g','Gamma'] returns derivations for both g and
        Gamma."""
        tools = NoetherTools(
            SessionStore(tmp_path / "sessions"),
            llm=StubLLMAdapter(reply=templates.get("eval2_palatini_metric")),
            results_root=tmp_path / "results",
        )
        body = tools.ingest(PALATINI_LAGRANGIAN)
        _resolve_all_geometry(body, tools)
        sid = body["session_id"]
        result = tools.derive(sid, ["g", "Gamma"])
        assert "derivations" in result, f"expected derivations; got {result}"
        wrt_set = {d["wrt"] for d in result["derivations"]}
        assert "g" in wrt_set, f"must include wrt='g'; got {wrt_set}"
        assert "Gamma" in wrt_set, f"must include wrt='Gamma'; got {wrt_set}"

    @requires_cadabra
    def test_derive_gamma_uses_independent_connection_capability(self, tmp_path):
        """The Gamma derivation must carry the INDEPENDENT_CONNECTION
        capability, not the generic VARY."""
        tools = NoetherTools(
            SessionStore(tmp_path / "sessions"),
            llm=StubLLMAdapter(reply=templates.get("eval2_palatini_connection")),
            results_root=tmp_path / "results",
        )
        body = tools.ingest(PALATINI_LAGRANGIAN)
        _resolve_all_geometry(body, tools)
        sid = body["session_id"]
        result = tools.derive(sid, ["Gamma"])
        gamma = next(d for d in result["derivations"] if d["wrt"] == "Gamma")
        assert gamma["capability"] == "independent-connection"

    @requires_cadabra
    def test_derive_unverified_eom_has_nonempty_detail(self, tmp_path):
        """VAL-EOM-017: an unverified EOM derivation must have
        verified==False and a non-empty detail naming the blocker."""
        tools = NoetherTools(
            SessionStore(tmp_path / "sessions"),
            llm=StubLLMAdapter(reply=(
                'print("NOETHER_RESULT: x");\n'
                'print("NOETHER_CHECK: residue_zero=False");\n'
            )),
            results_root=tmp_path / "results",
        )
        body = tools.ingest(PALATINI_LAGRANGIAN)
        _resolve_all_geometry(body, tools)
        sid = body["session_id"]
        result = tools.derive(sid, ["g"])
        d = result["derivations"][0]
        assert d["verified"] is False, f"expected verified=False; got {d['verified']}"
        assert d["detail"], "unverified derivation must have non-empty detail"
        assert "unverified" in d["detail"].lower() or "nonzero" in d["detail"].lower(), (
            f"detail must name the blocker; got: {d['detail']}"
        )

    @requires_cadabra
    def test_derive_script_failure_has_nonempty_detail(self, tmp_path):
        """VAL-EOM-017: a derivation whose script never reaches the residue
        check must have verified==False with a detail about the failure."""
        # A script that produces no NOETHER_CHECK at all
        tools = NoetherTools(
            SessionStore(tmp_path / "sessions"),
            llm=StubLLMAdapter(reply="ex := 1;\n"),
            results_root=tmp_path / "results",
        )
        body = tools.ingest(PALATINI_LAGRANGIAN)
        _resolve_all_geometry(body, tools)
        sid = body["session_id"]
        result = tools.derive(sid, ["g"])
        d = result["derivations"][0]
        assert d["verified"] is False
        assert d["detail"], "gated derivation must have non-empty detail"
        assert "no residue check" in d["detail"] or "did not run" in d["detail"], (
            f"detail must name the blocker; got: {d['detail']}"
        )
