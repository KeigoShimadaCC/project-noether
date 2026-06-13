"""MCP adapter: same session surface, no-guessing contract as tool results.

Skips cleanly when the [mcp] extra is not installed. Tool logic is tested
through NoetherTools (plain methods); the FastMCP wrapper is checked for the
expected tool registry.
"""

import asyncio

import pytest

pytest.importorskip("mcp")

from noether.mcp import NoetherTools, create_mcp_server  # noqa: E402
from noether.orchestrator.store import SessionStore  # noqa: E402


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
        }
