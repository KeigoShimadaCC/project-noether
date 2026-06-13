"""Cadabra2 adapter.

Executes audited script templates in a subprocess (`cadabra2` CLI) with a
timeout, captures everything verbatim for provenance, and parses results from
sentinel-marked output lines only.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from noether.kernels.base import (
    Capability,
    ComputedResult,
    KernelRawOutput,
    KernelScript,
    KernelTask,
    KernelUnavailable,
)
from noether.kernels.cadabra import templates

RESULT_SENTINEL = "NOETHER_RESULT:"
CHECK_SENTINEL = "NOETHER_CHECK:"


def _find_executable() -> str | None:
    env = os.environ.get("NOETHER_CADABRA")
    if env and Path(env).exists():
        return env
    return shutil.which("cadabra2")


class CadabraAdapter:
    name = "cadabra"

    def __init__(self, executable: str | None = None, timeout: int = 300):
        self.executable = executable or _find_executable()
        self.timeout = timeout

    def available(self) -> bool:
        return self.executable is not None

    def version(self) -> str:
        if not self.available():
            return "unavailable"
        try:
            out = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            first = (out.stdout or out.stderr).strip().splitlines()
            return first[0] if first else "unknown"
        except Exception:
            return "unknown"

    def capabilities(self) -> set[Capability]:
        return {
            Capability.VARY,
            Capability.IBP,
            Capability.CANONICALIZE,
            Capability.SUBSTITUTE,
        }

    def run(self, task: KernelTask, npr: Any = None) -> ComputedResult:
        if not self.available():
            raise KernelUnavailable(
                "cadabra2 executable not found; install via "
                "`brew tap kpeeters/repo && brew install cadabra2` "
                "or set NOETHER_CADABRA"
            )
        template_name = task.payload.get("template")
        inline = task.payload.get("script")
        if template_name:
            # Frozen, golden-tested script: the trusted offline core (evals).
            source = templates.get(template_name)
            origin = f"template: {template_name}"
            value_extra: dict[str, Any] = {"template": template_name}
        elif inline:
            # Parameterized script (e.g. LLM-generated for an arbitrary action).
            # It carries no authority: the result is trusted only after the
            # verification ladder confirms it (AGENTS.md rules 1, 3).
            source = inline
            origin = "generated (parameterized; unverified until the ladder confirms it)"
            value_extra = {"generated": True}
        else:
            raise ValueError(
                "cadabra tasks must name an audited template or provide an inline 'script'"
            )

        script = KernelScript(kernel_name=self.name, language="cadabra", source=source)
        raw = self._execute(source)
        result_tex, checks = _parse_sentinels(raw.stdout)
        return ComputedResult(
            kernel_name=self.name,
            kernel_version=self.version(),
            script=script,
            raw=raw,
            expression_tex=result_tex,
            value={"checks": checks, **value_extra},
            notes=[origin],
        )

    def _execute(self, source: str) -> KernelRawOutput:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "script.cdb"
            path.write_text(source)
            start = time.monotonic()
            proc = subprocess.run(
                [self.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tmp,
            )
            duration = time.monotonic() - start
        return KernelRawOutput(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            duration_s=round(duration, 3),
        )


def _parse_sentinels(stdout: str) -> tuple[str | None, dict[str, str]]:
    """Only sentinel-marked lines count as results; everything else is noise."""
    result_tex: str | None = None
    checks: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(RESULT_SENTINEL):
            result_tex = line[len(RESULT_SENTINEL) :].strip()
        elif line.startswith(CHECK_SENTINEL):
            body = line[len(CHECK_SENTINEL) :].strip()
            key, _, val = body.partition("=")
            checks[key.strip()] = val.strip()
    return result_tex, checks
