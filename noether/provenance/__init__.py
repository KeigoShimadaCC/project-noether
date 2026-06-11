"""Provenance bundles (docs/02_TECH_SPEC.md section 7).

Every result is a directory with its assumptions, scripts, raw kernel output,
check verdicts, and narrative. There is no API for a bare expression.
"""

from noether.provenance.bundle import ResultBundle, write_bundle

__all__ = ["ResultBundle", "write_bundle"]
