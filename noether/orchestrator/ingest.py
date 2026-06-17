"""INGEST: raw LaTeX action -> NPR draft + ambiguity ledger.

This is the semantic half of the INGEST beat (docs/02_TECH_SPEC.md section
3.2). It sits on top of the purely syntactic parser (noether.npr.parse) and
does two things, both deliberately conservative:

  1. Object discovery. It walks the parsed Lagrangian and classifies each
     distinct symbol syntactically (metric, curvature shorthand, function,
     tensor field, scalar field). It always includes a metric, because the
     action measure carries sqrt(-g).

  2. Ambiguity ledger. It NEVER assigns physics meaning. Field roles, the
     fields to vary, the curvature/connection interpretation of composite
     symbols, the coupling-vs-constant status of functions, the spacetime
     dimension, and the conventions are all emitted as open questions. The
     resulting NPR therefore has a non-empty ledger, so build_plan() raises
     AmbiguityBlocked: a freshly ingested action is structurally un-plannable
     until a human resolves it (AGENTS.md rule 4, the no-guessing contract).

Provisional ObjectDecl roles are placeholders only; they carry no authority
while the ledger is open and are meant to be confirmed or overwritten during
elicitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noether.npr.ast import Deriv, Expr, Func, Num, Pow, Prod, Sum, Sym, Tensor
from noether.npr.conventions import NOETHER_DEFAULT_V1, Conventions
from noether.npr.parse import GEOMETRIC_NAMES, parse_action, tokenize
from noether.npr.schema import (
    NPR,
    Action,
    Ambiguity,
    ConnectionSpec,
    Geometry,
    ObjectDecl,
    Task,
)

_GREEK_SCALARS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
}

_GEOMETRIC_SYMMETRIES = {
    "T": "antisymmetric",
    "Q": "symmetric",
}
_COMPOSITE_GEOMETRIC_NAMES = {"G", "C", "W"}
_CURVATURE_NAMES = {"R", "G", "C", "W"}
_EXPLICIT_CONNECTION_NAMES = {"Gamma"}


@dataclass
class _SymbolInfo:
    name: str
    max_rank: int = 0
    seen_scalar: bool = False
    seen_indexed: bool = False
    seen_func: bool = False
    differentiated: bool = False
    func_args: list[str] = field(default_factory=list)


@dataclass
class _GeometryCues:
    has_curvature: bool = False
    has_connection: bool = False
    has_torsion: bool = False
    has_nonmetricity: bool = False


def _collect(expr: Expr, into: dict[str, _SymbolInfo], *, under_deriv: bool = False) -> None:
    match expr:
        case Num():
            return
        case Sym(name=name):
            info = into.setdefault(name, _SymbolInfo(name))
            info.seen_scalar = True
            info.differentiated |= under_deriv
        case Func(name=name, args=args):
            info = into.setdefault(name, _SymbolInfo(name))
            info.seen_func = True
            for a in args:
                # record scalar argument names so derivative shorthands can be
                # proposed later; only simple scalar symbols qualify as args
                if isinstance(a, Sym) and a.name not in info.func_args:
                    info.func_args.append(a.name)
                _collect(a, into)
        case Tensor(name=name, indices=indices):
            info = into.setdefault(name, _SymbolInfo(name))
            info.max_rank = max(info.max_rank, len(indices))
            if indices:
                info.seen_indexed = True
            else:
                info.seen_scalar = True
            info.differentiated |= under_deriv
        case Deriv(expr=inner):
            _collect(inner, into, under_deriv=True)
        case Pow(base=base):
            _collect(base, into)
        case Prod(factors=factors):
            for f in factors:
                _collect(f, into)
        case Sum(terms=terms):
            for t in terms:
                _collect(t, into)
        case _:
            raise TypeError(f"unhandled expr node {expr!r}")


def _has_explicit_connection(expr: Expr) -> bool:
    match expr:
        case Num() | Sym():
            return False
        case Func(args=args):
            return any(_has_explicit_connection(arg) for arg in args)
        case Tensor(connection=connection):
            return connection is not None
        case Deriv(expr=inner, connection=connection):
            return connection not in (None, "metric") or _has_explicit_connection(inner)
        case Pow(base=base):
            return _has_explicit_connection(base)
        case Prod(factors=factors):
            return any(_has_explicit_connection(factor) for factor in factors)
        case Sum(terms=terms):
            return any(_has_explicit_connection(term) for term in terms)
        case _:
            raise TypeError(f"unhandled expr node {expr!r}")


def _geometry_cues(expr: Expr) -> _GeometryCues:
    cues = _GeometryCues()

    def walk(node: Expr) -> None:
        match node:
            case Num() | Sym():
                return
            case Func(args=args):
                for arg in args:
                    walk(arg)
            case Tensor(name=name, connection=connection):
                cues.has_curvature |= name in _CURVATURE_NAMES
                cues.has_connection |= name in _EXPLICIT_CONNECTION_NAMES or connection is not None
                cues.has_torsion |= name == "T"
                cues.has_nonmetricity |= name == "Q"
            case Deriv(expr=inner, connection=connection):
                cues.has_connection |= connection not in (None, "metric")
                walk(inner)
            case Pow(base=base):
                walk(base)
            case Prod(factors=factors):
                for factor in factors:
                    walk(factor)
            case Sum(terms=terms):
                for term in terms:
                    walk(term)
            case _:
                raise TypeError(f"unhandled expr node {node!r}")

    walk(expr)
    return cues


def _needs_geometry_questionnaire(cues: _GeometryCues) -> bool:
    return any(
        (
            cues.has_curvature,
            cues.has_connection,
            cues.has_torsion,
            cues.has_nonmetricity,
        )
    )


def _append_geometry_ambiguities(ambiguities: list[Ambiguity], cues: _GeometryCues) -> None:
    independent_first = cues.has_connection or cues.has_torsion or cues.has_nonmetricity
    connection_options = (
        ["independent", "levi-civita"] if independent_first else ["levi-civita", "independent"]
    )
    if cues.has_connection:
        connection_question = (
            "The action carries an explicit connection: should it be treated as "
            "an independent connection or as the metric Levi-Civita one?"
        )
    elif cues.has_torsion:
        connection_question = (
            "The action uses explicit torsion T, which points to connection-dependent "
            "geometry: is the connection independent or Levi-Civita?"
        )
    elif cues.has_nonmetricity:
        connection_question = (
            "The action uses explicit non-metricity Q, which points to "
            "connection-dependent geometry: is the connection independent or "
            "Levi-Civita?"
        )
    else:
        connection_question = (
            "The action carries curvature: is the geometry Levi-Civita, or does it "
            "use an independent connection?"
        )
    ambiguities.append(
        Ambiguity(
            id="amb-connection",
            question=connection_question,
            kind="inferable",
            options=connection_options,
        )
    )

    torsion_options = (
        ["torsion-present", "torsion-free"]
        if cues.has_torsion
        else ["torsion-free", "torsion-allowed"]
    )
    torsion_question = (
        "The action uses explicit torsion T. Should torsion be treated as present, "
        "or should the connection be torsion-free?"
        if cues.has_torsion
        else "Should the connection be torsion-free, or should torsion be allowed?"
    )
    ambiguities.append(
        Ambiguity(
            id="amb-torsion",
            question=torsion_question,
            kind="inferable",
            options=torsion_options,
        )
    )

    nonmetricity_options = (
        ["nonmetricity-present", "nonmetricity-free"]
        if cues.has_nonmetricity
        else ["nonmetricity-free", "nonmetricity-allowed"]
    )
    nonmetricity_question = (
        "The action uses explicit non-metricity Q. Should non-metricity be treated "
        "as present, or should the connection stay metric-compatible there?"
        if cues.has_nonmetricity
        else "Should the connection be metric-compatible, or should non-metricity be allowed?"
    )
    ambiguities.append(
        Ambiguity(
            id="amb-nonmetricity",
            question=nonmetricity_question,
            kind="inferable",
            options=nonmetricity_options,
        )
    )

    metric_compatibility_options = (
        ["not-metric-compatible", "metric-compatible"]
        if cues.has_nonmetricity
        else ["metric-compatible", "not-metric-compatible"]
    )
    ambiguities.append(
        Ambiguity(
            id="amb-metric-compatibility",
            question=(
                "Should the connection be metric-compatible, or should the metric "
                "have a nonzero covariant derivative?"
            ),
            kind="inferable",
            options=metric_compatibility_options,
        )
    )


def _classify(info: _SymbolInfo) -> ObjectDecl:
    """Syntactic classification only. Roles are provisional placeholders."""
    name = info.name
    if name == "g":
        return ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2)
    if info.seen_func:
        return ObjectDecl(
            name=name, kind="function", role="coupling", rank=0, args=list(info.func_args)
        )
    if name in GEOMETRIC_NAMES:
        symmetry = _GEOMETRIC_SYMMETRIES.get(
            name, "symmetric" if info.max_rank == 2 else "none"
        )
        return ObjectDecl(
            name=name, kind="shorthand", role="shorthand", symmetry=symmetry, rank=info.max_rank
        )
    if info.seen_indexed:
        return ObjectDecl(name=name, kind="tensor-field", role="dynamical", rank=info.max_rank)
    return ObjectDecl(name=name, kind="scalar-field", role="dynamical", rank=0)


def _scalar_tex(name: str) -> str:
    return f"\\{name}" if name in _GREEK_SCALARS else name


def _recognize_kinetic_scalar(
    objects: list[ObjectDecl], symbols: dict[str, _SymbolInfo]
) -> str | None:
    """Reclassify a bare `X` as the convention-named kinetic shorthand.

    `noether-default-v1` names the canonical kinetic scalar X = -1/2
    nabla_mu phi nabla^mu phi (AGENTS.md section 5). When the action carries
    exactly one dynamical scalar field and `X` only ever appears as a plain
    scalar (never indexed, differentiated, or called), treat it as that
    shorthand of the scalar rather than as an independent dynamical field, so
    it is not silently offered as something to vary. The reading is still put
    to the human as `amb-kinetic-X`; this only sets the conservative default.
    """
    x = next((o for o in objects if o.name == "X"), None)
    if x is None or x.kind != "scalar-field":
        return None
    info = symbols.get("X")
    if info is None or info.seen_indexed or info.differentiated or info.seen_func:
        return None
    scalars = [o for o in objects if o.kind == "scalar-field" and o.name != "X"]
    if len(scalars) != 1:
        return None
    phi = scalars[0].name
    tex = _scalar_tex(phi)
    x.kind = "shorthand"
    x.role = "shorthand"
    x.rank = 0
    x.definition_tex = rf"-\tfrac12 \nabla_\mu {tex} \nabla^\mu {tex}"
    return phi


def _measure_dimension(measure_tex: str) -> int | str | None:
    """Pull the dimension out of a measure like 'd^4x' or 'd^Dx'."""
    toks = tokenize(measure_tex)
    for i in range(len(toks) - 1):
        if toks[i].kind == "name" and toks[i].value == "d" and toks[i + 1].value == "^":
            target = toks[i + 2] if i + 2 < len(toks) else None
            if target is None:
                return None
            if target.kind == "num":
                return int(target.value)
            return target.value
    return None


@dataclass
class IngestResult:
    npr: NPR
    lagrangian: Expr
    object_names: list[str]
    questions: list[str] = field(default_factory=list)


def ingest_action(
    measure_tex: str,
    lagrangian_tex: str,
    *,
    conventions: Conventions = NOETHER_DEFAULT_V1,
) -> IngestResult:
    """Parse an action and produce a draft NPR with an open ambiguity ledger.

    The returned NPR is intentionally not well-posed: build_plan() on it will
    raise AmbiguityBlocked until the questions are resolved by elicitation.
    """
    parsed = parse_action(lagrangian_tex)
    lagrangian = parsed.expr
    geometry_cues = _geometry_cues(lagrangian)

    symbols: dict[str, _SymbolInfo] = {}
    _collect(lagrangian, symbols)

    objects: list[ObjectDecl] = [
        ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2)
    ]
    for name in sorted(symbols):
        if name == "g":
            continue
        objects.append(_classify(symbols[name]))

    kinetic_scalar = _recognize_kinetic_scalar(objects, symbols)

    vary_candidates = sorted(
        obj.name for obj in objects if obj.kind in ("metric", "scalar-field", "tensor-field")
    )

    ambiguities: list[Ambiguity] = [
        Ambiguity(
            id="amb-conventions",
            question=(
                "Which conventions: noether-default-v1 (dimension, signature, "
                "curvature signs, symmetrization weight) or a custom block?"
            ),
            kind="conventional",
            options=["noether-default-v1", "custom"],
        ),
        Ambiguity(
            id="amb-vary-wrt",
            question="Vary the action with respect to which field(s)? Candidates: "
            + ", ".join(vary_candidates),
            kind="undecidable",
            options=vary_candidates,
        ),
    ]

    for obj in objects:
        if obj.kind == "function":
            ambiguities.append(
                Ambiguity(
                    id=f"amb-coupling-{obj.name}",
                    question=(
                        f"Is {obj.name} an arbitrary function of its argument(s), "
                        "or a fixed constant?"
                    ),
                    kind="undecidable",
                    options=["arbitrary-function", "constant"],
                )
            )
        if obj.kind == "shorthand" and obj.name in _COMPOSITE_GEOMETRIC_NAMES:
            ambiguities.append(
                Ambiguity(
                    id=f"amb-composite-{obj.name}",
                    question=(
                        f"Is {obj.name} the standard curvature combination built "
                        "from the metric, or an independent field?"
                    ),
                    kind="undecidable",
                    options=[f"{obj.name}-of-metric", "independent-field"],
                )
            )

    if kinetic_scalar is not None:
        tex = _scalar_tex(kinetic_scalar)
        ambiguities.append(
            Ambiguity(
                id="amb-kinetic-X",
                question=(
                    rf"Is X the canonical kinetic scalar -\tfrac12 \nabla_\mu {tex} "
                    rf"\nabla^\mu {tex} of {kinetic_scalar} (noether-default-v1), "
                    "or an independent field?"
                ),
                kind="conventional",
                options=["kinetic-scalar", "independent-field"],
            )
        )

    connection = ConnectionSpec(type="levi-civita")
    if _needs_geometry_questionnaire(geometry_cues):
        _append_geometry_ambiguities(ambiguities, geometry_cues)

    dimension = _measure_dimension(measure_tex)
    if dimension is not None and dimension != 4:
        ambiguities.append(
            Ambiguity(
                id="amb-dimension",
                question=(
                    f"The measure is d^{dimension}x: keep the dimension symbolic "
                    "(general D) or fix a specific integer?"
                ),
                kind="undecidable",
                options=["symbolic", "fixed-4"],
            )
        )

    npr = NPR(
        conventions=conventions,
        geometry=Geometry(connection=connection),
        objects=objects,
        action=Action(
            measure_tex=measure_tex,
            lagrangian=lagrangian,
            lagrangian_tex=lagrangian_tex,
        ),
        task=Task(type="vary", with_respect_to=vary_candidates),
        ambiguities=ambiguities,
    )

    return IngestResult(
        npr=npr,
        lagrangian=lagrangian,
        object_names=[obj.name for obj in objects],
        questions=[a.question for a in ambiguities],
    )
