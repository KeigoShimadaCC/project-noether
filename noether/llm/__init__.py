"""LLM adapters: the orchestrator's swappable model backend.

The model orchestrates, narrates, and PROPOSES; it never computes verified
mathematics (kernels do that) and never resolves a physics ambiguity on its
own (the human does, AGENTS.md rule 4). Adapters expose a single capability:
turn a prompt into text. Everything physics-bearing is gated downstream.
"""

from noether.llm.base import LLMAdapter, LLMError, StubLLMAdapter, stub_reply
from noether.llm.cli import KNOWN_BACKENDS, CliBackend, CliLLMAdapter

__all__ = [
    "KNOWN_BACKENDS",
    "CliBackend",
    "CliLLMAdapter",
    "LLMAdapter",
    "LLMError",
    "StubLLMAdapter",
    "stub_reply",
]
