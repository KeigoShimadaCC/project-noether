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


@dataclass
class _SymbolInfo:
    name: str
    max_rank: int = 0
    seen_scalar: bool = False
    seen_indexed: bool = False
    seen_func: bool = False
    differentiated: bool = False


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


def _classify(info: _SymbolInfo) -> ObjectDecl:
    """Syntactic classification only. Roles are provisional placeholders."""
    name = info.name
    if name == "g":
        return ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2)
    if info.seen_func:
        return ObjectDecl(name=name, kind="function", role="coupling", rank=0)
    if name in GEOMETRIC_NAMES:
        symmetry = "symmetric" if info.max_rank == 2 else "none"
        return ObjectDecl(
            name=name, kind="shorthand", role="shorthand", symmetry=symmetry, rank=info.max_rank
        )
    if info.seen_indexed:
        return ObjectDecl(name=name, kind="tensor-field", role="dynamical", rank=info.max_rank)
    return ObjectDecl(name=name, kind="scalar-field", role="dynamical", rank=0)


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

    symbols: dict[str, _SymbolInfo] = {}
    _collect(lagrangian, symbols)

    objects: list[ObjectDecl] = [
        ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2)
    ]
    for name in sorted(symbols):
        if name == "g":
            continue
        objects.append(_classify(symbols[name]))

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
        if obj.kind == "shorthand" and obj.name != "R":
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

    connection = ConnectionSpec(type="levi-civita")
    if parsed.connection_annotated:
        ambiguities.append(
            Ambiguity(
                id="amb-connection",
                question=(
                    "The curvature is annotated with an explicit connection: is it "
                    "Levi-Civita (metric-compatible) or an independent connection?"
                ),
                kind="undecidable",
                options=["levi-civita", "independent"],
            )
        )

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
