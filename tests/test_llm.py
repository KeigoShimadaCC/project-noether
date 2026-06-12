"""LLM adapters: stub behaviour, JSON extraction, CLI detection/transport.

No test here touches the network or a real model: the CLI adapter is driven
with an injected `which` and an injected subprocess runner, and the in-process
StubLLMAdapter returns canned text.
"""

import subprocess

import pytest

from noether.llm import (
    KNOWN_BACKENDS,
    CliBackend,
    CliLLMAdapter,
    LLMError,
    StubLLMAdapter,
    stub_reply,
)
from noether.orchestrator.elicit import parse_llm_json


class TestStub:
    def test_complete_returns_reply(self):
        adapter = StubLLMAdapter(reply='{"a": 1}')
        assert adapter.available() and adapter.complete("sys", "prompt") == '{"a": 1}'

    def test_stub_reply_is_parseable(self):
        raw = stub_reply({"amb-x": "opt1"}, rationale="because")
        parsed = parse_llm_json(raw)
        assert parsed["amb-x"] == {"choice": "opt1", "rationale": "because"}


class TestJsonExtraction:
    def test_extracts_from_fenced_noise(self):
        text = 'Sure!\n```json\n{"k": {"choice": "a"}}\n```\nhope that helps'
        assert parse_llm_json(text) == {"k": {"choice": "a"}}

    def test_braces_inside_strings_are_ignored(self):
        text = '{"rationale": "use {braces} carefully", "choice": "a"}'
        assert parse_llm_json(text)["choice"] == "a"

    def test_no_json_raises(self):
        with pytest.raises(LLMError):
            parse_llm_json("no json here")


def _fake_which(present: str):
    return lambda name: f"/usr/bin/{name}" if name == present else None


class TestCliDetection:
    def test_detects_first_available_in_order(self):
        # codex absent, claude present -> claude-code backend chosen.
        adapter = CliLLMAdapter(which=_fake_which("claude"))
        assert adapter.available()
        assert adapter.name == "claude-code"

    def test_none_available(self):
        adapter = CliLLMAdapter(which=lambda _name: None)
        assert not adapter.available()
        assert adapter.name == "none"

    def test_complete_without_backend_raises(self):
        adapter = CliLLMAdapter(which=lambda _name: None)
        with pytest.raises(LLMError):
            adapter.complete("sys", "prompt")


class TestCliTransport:
    def _runner(self, captured, returncode=0, stdout="reply", stderr=""):
        def run(argv, stdin, timeout):
            captured.append(list(argv))
            return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)

        return run

    def test_builds_argv_and_returns_stdout(self):
        captured: list[list[str]] = []
        backend = CliBackend("codex", "codex", ("--version",), ("exec",))
        adapter = CliLLMAdapter(
            backend, which=_fake_which("codex"), runner=self._runner(captured, stdout="hi")
        )
        out = adapter.complete("SYS", "PROMPT")
        assert out == "hi"
        argv = captured[0]
        assert argv[0] == "/usr/bin/codex"
        assert argv[1] == "exec"
        assert "SYS" in argv[-1] and "PROMPT" in argv[-1]

    def test_nonzero_exit_raises(self):
        captured: list[list[str]] = []
        backend = KNOWN_BACKENDS[0]
        adapter = CliLLMAdapter(
            backend,
            which=_fake_which(backend.executable),
            runner=self._runner(captured, returncode=3, stderr="boom"),
        )
        with pytest.raises(LLMError, match="exited 3"):
            adapter.complete("", "p")
