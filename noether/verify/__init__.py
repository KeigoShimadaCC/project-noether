"""Verification ladder V0..V4 (docs/03_METHODOLOGY.md section 4).

Every result climbs as far as its class allows. A result whose checks did not
all pass is never presented as verified.
"""

from noether.verify.checks import (
    CheckResult,
    DivergenceFreeCheck,
    EqualOnBackgroundCheck,
    SymmetricCheck,
    WellFormedCheck,
)
from noether.verify.ladder import run_ladder

__all__ = [
    "CheckResult",
    "DivergenceFreeCheck",
    "EqualOnBackgroundCheck",
    "SymmetricCheck",
    "WellFormedCheck",
    "run_ladder",
]
