"""LLM adapter interface and an in-process stub for tests.

An adapter's only job is `complete(system, prompt) -> text`. It carries no
authority over physics: it cannot inject a computed expression into a result
(only kernels can) and it cannot resolve an ambiguity (only a confirmed human
answer can, via noether.orchestrator.elicit.apply_resolutions).
"""

from __future__ import annotations

import json
from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLMAdapter(Protocol):
    name: str

    def available(self) -> bool: ...

    def version(self) -> str: ...

    def complete(self, system: str, prompt: str) -> str: ...


class StubLLMAdapter:
    """Deterministic in-process adapter: returns a fixed reply, ignoring the
    prompt. Used to test the elicitation plumbing without a real model."""

    name = "stub"

    def __init__(self, reply: str = "{}") -> None:
        self._reply = reply

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return "stub-1"

    def complete(self, system: str, prompt: str) -> str:
        return self._reply


def stub_reply(answers: dict[str, str], rationale: str = "stub rationale") -> str:
    """Build the JSON an obedient model would return for the given choices."""
    return json.dumps(
        {amb_id: {"choice": choice, "rationale": rationale} for amb_id, choice in answers.items()}
    )
