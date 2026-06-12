"""Single source of truth for pinned kernel versions (AGENTS.md section 6).

The golden adapter tests, the audited cadabra templates, and the sympy
component checks were all validated against the versions pinned here. Bumping
either pin is a deliberate act: re-run the full eval suite and the cadabra
golden tests, confirm every check is still green, and update this file in the
same commit. Nothing else in the tree should hard-code a kernel version.
"""

from __future__ import annotations

import sympy as sp

# major.minor of the sympy used for component verification (V0-V3).
SYMPY_PINNED = "1.14"

# cadabra2 CLI version the audited templates in cadabra/templates.py target.
CADABRA_PINNED = "2.5.15"


def sympy_version() -> str:
    return sp.__version__


def sympy_matches_pin() -> bool:
    """True when the installed sympy is the pinned major.minor series."""
    return ".".join(sp.__version__.split(".")[:2]) == SYMPY_PINNED
