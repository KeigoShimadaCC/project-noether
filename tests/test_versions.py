"""Kernel version pinning: catch silent drift that could change canon forms.

Correctness over speed (AGENTS.md rule 7) includes reproducibility: a sympy
minor-version bump can change canonicalisation, so the component checks are
only meaningful against the pinned series. This test fails loudly when the
installed kernel drifts from noether.kernels.versions, forcing a deliberate
re-audit rather than a silent change.
"""

from noether.kernels.versions import SYMPY_PINNED, sympy_matches_pin, sympy_version


def test_sympy_matches_pin():
    assert sympy_matches_pin(), (
        f"installed sympy {sympy_version()} is not the pinned {SYMPY_PINNED}.x series; "
        "re-run the eval suite and the cadabra golden tests, then update "
        "noether/kernels/versions.py in the same commit"
    )
