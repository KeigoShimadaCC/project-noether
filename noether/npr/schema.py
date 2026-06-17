"""Top-level NPR schema (docs/02_TECH_SPEC.md section 4).

The ambiguity ledger is the load-bearing part: a non-empty list of unresolved
ambiguities structurally blocks planning. Guessing is impossible, not just
discouraged.
"""

from typing import Literal

from pydantic import BaseModel, Field

from noether.npr.ast import Expr
from noether.npr.conventions import Conventions

Role = Literal["dynamical", "background", "coupling", "constant", "shorthand"]
SymmetryKind = Literal["none", "symmetric", "antisymmetric"]


class ObjectDecl(BaseModel):
    name: str
    kind: Literal["metric", "tensor-field", "scalar-field", "connection", "function", "shorthand"]
    role: Role
    symmetry: SymmetryKind = "none"
    rank: int = 0
    args: list[str] = []  # for kind="function": names of scalar arguments
    definition_tex: str | None = None  # for shorthands, e.g. X
    # for kind="tensor-field": the gauge group, when the field is a gauge
    # potential. None or "U(1)" is abelian (Maxwell); a non-abelian group such
    # as "SU(2)"/"SU(N)" selects the Yang-Mills sector. Never inferred silently.
    gauge_group: str | None = None


class ConnectionSpec(BaseModel):
    type: Literal["levi-civita", "independent"]
    torsion: bool = False
    nonmetricity: bool = False
    metric_compatible: bool = True
    # Whether the curvature of the connection is constrained to vanish.
    # Distinguishes teleparallel (T!=0, Q=0, R=0) from Riemann-Cartan
    # (T!=0, Q=0, R!=0), and symmetric-teleparallel (T=0, Q!=0, R=0)
    # from general non-metric (T=0, Q!=0, R!=0).
    curvature_free: bool = False
    family: Literal[
        "riemannian",
        "metric-affine",
        "riemann-cartan",
        "teleparallel",
        "symmetric-teleparallel",
    ] = "riemannian"


class Geometry(BaseModel):
    metric_name: str = "g"
    connection_name: str = "Gamma"
    connection: ConnectionSpec = Field(
        default_factory=lambda: ConnectionSpec(type="levi-civita")
    )


class Action(BaseModel):
    measure_tex: str
    lagrangian: Expr
    lagrangian_tex: str | None = None  # cached rendering, never the source of truth


class TargetForm(BaseModel):
    basis: str = "curvature-canonical"
    collect_by: str = "tensor-structure"


class Task(BaseModel):
    type: Literal["vary", "reduce", "adm", "perturb", "identity-check"]
    with_respect_to: list[str] = []
    target_form: TargetForm = TargetForm()


class Ambiguity(BaseModel):
    id: str
    question: str
    kind: Literal["inferable", "conventional", "undecidable"]
    options: list[str] = []
    resolution: str | None = None

    @property
    def resolved(self) -> bool:
        return self.resolution is not None


class NPR(BaseModel):
    npr_version: Literal["0.1"] = "0.1"
    conventions: Conventions
    geometry: Geometry
    objects: list[ObjectDecl] = []
    action: Action
    task: Task
    ambiguities: list[Ambiguity] = Field(default_factory=list)

    def unresolved_ambiguities(self) -> list[Ambiguity]:
        return [a for a in self.ambiguities if not a.resolved]

    def is_well_posed(self) -> bool:
        return not self.unresolved_ambiguities()

    def object_named(self, name: str) -> ObjectDecl:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"no object named {name!r} in NPR")
