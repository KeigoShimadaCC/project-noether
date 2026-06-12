"""Ambient-auth LLM via a local agent CLI subprocess (no API key).

Auto-detects an installed agent CLI and runs it in one-shot headless mode.
Credentials live in that CLI's own login session, never in Noether or its
environment. This mirrors the cadabra subprocess transport (kernels/cadabra):
the model is just another sandboxed tool behind a clean adapter.

The exact headless flags differ across CLIs and versions; the table below is
best-effort and easy to adjust. The detection order is the table order.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from noether.llm.base import LLMError

Runner = Callable[[Sequence[str], str | None, int], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class CliBackend:
    name: str
    executable: str
    version_args: tuple[str, ...]
    prompt_flags: tuple[str, ...]  # placed before the prompt argument


# One-shot, non-interactive invocation per CLI. Prompt is passed as the final
# positional argument after these flags.
KNOWN_BACKENDS: tuple[CliBackend, ...] = (
    CliBackend("codex", "codex", ("--version",), ("exec",)),
    CliBackend("claude-code", "claude", ("--version",), ("-p",)),
    CliBackend("gemini", "gemini", ("--version",), ("-p",)),
    CliBackend("droid", "droid", ("--version",), ("exec",)),
)


def _subprocess_runner(
    argv: Sequence[str], stdin: str | None, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), input=stdin, capture_output=True, text=True, timeout=timeout)


class CliLLMAdapter:
    """Drive an already-authenticated agent CLI as a subprocess."""

    def __init__(
        self,
        backend: CliBackend | None = None,
        *,
        which: Callable[[str], str | None] = shutil.which,
        runner: Runner | None = None,
        timeout: int = 120,
    ) -> None:
        self._which = which
        self._runner = runner or _subprocess_runner
        self.timeout = timeout
        if backend is not None:
            self.backend: CliBackend | None = backend
            self.executable = which(backend.executable)
        else:
            self.backend, self.executable = self._detect()

    def _detect(self) -> tuple[CliBackend | None, str | None]:
        for backend in KNOWN_BACKENDS:
            exe = self._which(backend.executable)
            if exe:
                return backend, exe
        return None, None

    @property
    def name(self) -> str:
        return self.backend.name if self.backend else "none"

    def available(self) -> bool:
        return self.backend is not None and self.executable is not None

    def version(self) -> str:
        if not self.available():
            return "unavailable"
        assert self.backend is not None and self.executable is not None
        try:
            cp = self._runner([self.executable, *self.backend.version_args], None, 30)
            lines = (cp.stdout or cp.stderr or "").strip().splitlines()
            return lines[0] if lines else "unknown"
        except Exception:
            return "unknown"

    def complete(self, system: str, prompt: str) -> str:
        if not self.available():
            looked = ", ".join(b.executable for b in KNOWN_BACKENDS)
            raise LLMError(f"no agent CLI detected (looked for: {looked})")
        assert self.backend is not None and self.executable is not None
        full = f"{system}\n\n{prompt}" if system else prompt
        argv = [self.executable, *self.backend.prompt_flags, full]
        try:
            cp = self._runner(argv, None, self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"{self.name} timed out after {self.timeout}s") from exc
        if cp.returncode != 0:
            detail = (cp.stderr or cp.stdout or "").strip()[:200]
            raise LLMError(f"{self.name} exited {cp.returncode}: {detail}")
        return cp.stdout or ""
