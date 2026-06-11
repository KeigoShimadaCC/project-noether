"""SymPy utility kernel.

Component-level differential geometry on explicit metrics. Role
(docs/01_RESEARCH.md section 1.3): scalar algebra, component evaluation on
explicit backgrounds, independent spot checks that falsify wrong tensor
claims. It is a falsifier, not the symbolic deriver.
"""

from noether.kernels.sympy_kernel.adapter import SympyKernelAdapter
from noether.kernels.sympy_kernel.geometry import (
    ComponentGeometry,
    components,
    random_diagonal_metric,
)

__all__ = ["ComponentGeometry", "SympyKernelAdapter", "components", "random_diagonal_metric"]
