"""Kernel adapter contract.

A result that did not come through `ComputedResult` carries no provenance and
must never reach the user (AGENTS.md rule 3).
"""

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Capability(StrEnum):
    VARY = "vary"
    IBP = "integrate-by-parts"
    CANONICALIZE = "canonicalize"
    SUBSTITUTE = "substitute"
    PERTURB = "perturb"
    ADM = "adm"
    COMPONENT_EVAL = "component-eval"
    INDEPENDENT_CONNECTION = "independent-connection"


class KernelUnavailable(RuntimeError):
    pass


class KernelTask(BaseModel):
    """A single capability-tagged unit of kernel work."""

    capability: Capability
    description: str
    payload: dict[str, Any] = Field(default_factory=dict)


class KernelScript(BaseModel):
    kernel_name: str
    language: str  # "cadabra", "python-sympy", "wolfram"
    source: str


class KernelRawOutput(BaseModel):
    stdout: str
    stderr: str = ""
    returncode: int = 0
    duration_s: float = 0.0


class ComputedResult(BaseModel):
    """A kernel-computed expression plus its receipt."""

    kernel_name: str
    kernel_version: str
    script: KernelScript
    raw: KernelRawOutput
    expression_tex: str | None = None
    value: Any = None  # structured payload (e.g. check verdict details)
    notes: list[str] = Field(default_factory=list)


@runtime_checkable
class KernelAdapter(Protocol):
    name: str

    def available(self) -> bool: ...

    def version(self) -> str: ...

    def capabilities(self) -> set[Capability]: ...

    def run(self, task: KernelTask, npr: Any) -> ComputedResult:
        """Compile, execute sandboxed, parse output. Raises KernelUnavailable
        if the backing engine is not installed."""
        ...
