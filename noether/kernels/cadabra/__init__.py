"""Cadabra2 kernel adapter: the Horizon 1 symbolic deriver.

Variation, integration by parts, canonicalization (docs/01_RESEARCH.md
section 1.2). Scripts are generated from audited templates; raw user LaTeX
never reaches the kernel.
"""

from noether.kernels.cadabra.adapter import CadabraAdapter

__all__ = ["CadabraAdapter"]
