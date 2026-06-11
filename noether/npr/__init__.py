"""NPR: the Noether Problem Representation.

The backend-agnostic contract between "what the physicist meant" and "what any
kernel executes" (docs/02_TECH_SPEC.md section 4). Nothing outside
noether.kernels may speak anything but NPR.
"""

from noether.npr.ast import Deriv, Expr, Func, Index, Num, Pow, Prod, Sum, Sym, Tensor
from noether.npr.conventions import NOETHER_DEFAULT_V1, Conventions
from noether.npr.schema import (
    NPR,
    Action,
    Ambiguity,
    ConnectionSpec,
    Geometry,
    ObjectDecl,
    TargetForm,
    Task,
)

__all__ = [
    "NPR",
    "NOETHER_DEFAULT_V1",
    "Action",
    "Ambiguity",
    "ConnectionSpec",
    "Conventions",
    "Deriv",
    "Expr",
    "Func",
    "Geometry",
    "Index",
    "Num",
    "ObjectDecl",
    "Pow",
    "Prod",
    "Sum",
    "Sym",
    "TargetForm",
    "Task",
    "Tensor",
]
