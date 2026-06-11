"""Kernel adapters. The only code allowed to speak a CAS dialect.

One interface, N implementations (docs/02_TECH_SPEC.md section 5). The planner
selects kernels by capability, never by name.
"""

from noether.kernels.base import (
    Capability,
    ComputedResult,
    KernelAdapter,
    KernelRawOutput,
    KernelScript,
    KernelTask,
    KernelUnavailable,
)

__all__ = [
    "Capability",
    "ComputedResult",
    "KernelAdapter",
    "KernelRawOutput",
    "KernelScript",
    "KernelTask",
    "KernelUnavailable",
]
