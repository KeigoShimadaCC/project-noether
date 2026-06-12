"""Evaluate NPR expressions to components on an explicit metric.

Vocabulary (resolved by name + index count, under noether-default-v1):
  g  rank 2   metric (any variance; raised/lowered as requested)
  R  rank 4   Riemann, base variance ^rho_{sigma mu nu}
  R  rank 2   Ricci
  R  rank 0   Ricci scalar
  G  rank 2   Einstein tensor

This evaluator exists to falsify claimed tensor equations on explicit
backgrounds. It is deliberately small; it is not a CAS.
"""

import sympy as sp

from noether.kernels.sympy_kernel.geometry import ComponentGeometry, _all_indices
from noether.npr.ast import Deriv, Expr, Func, Index, Num, Pow, Prod, Sum, Sym, Tensor

Array = sp.ImmutableDenseNDimArray


class EvaluationError(ValueError):
    pass


_BASE_VARIANCES = {
    ("R", 4): ["up", "down", "down", "down"],
    ("R", 2): ["down", "down"],
    ("G", 2): ["down", "down"],
    ("g", 2): ["down", "down"],
    ("H", 2): ["down", "down"],
}


def _base_components(name: str, rank: int, geom: ComponentGeometry):
    if name == "g" and rank == 2:
        return Array(geom.g)
    if name == "R" and rank == 4:
        return geom.riemann
    if name == "R" and rank == 2:
        return geom.ricci
    if name == "R" and rank == 0:
        return geom.ricci_scalar
    if name == "G" and rank == 2:
        return geom.einstein
    if name == "H" and rank == 2:
        return geom.gauss_bonnet
    if name == "GB" and rank == 0:
        return geom.gauss_bonnet_scalar
    raise EvaluationError(f"no component rule for tensor {name!r} of rank {rank}")


def _adjust_variances(arr, base: list[str], requested: list[str], geom: ComponentGeometry):
    for axis, (b, r) in enumerate(zip(base, requested, strict=True)):
        if b == r:
            continue
        arr = geom.raise_first_index(arr, axis) if r == "up" else geom.lower_index(arr, axis)
    return arr


def _contract_duplicates(arr, indices: list[Index], geom: ComponentGeometry):
    """Trace over repeated index names (validator guarantees up/down pairing)."""
    while True:
        names = [ix.name for ix in indices]
        dup = next((nm for nm in names if names.count(nm) == 2), None)
        if dup is None:
            return arr, indices
        i = names.index(dup)
        j = names.index(dup, i + 1)
        arr = sp.tensorcontraction(arr, (i, j))
        indices = [ix for k, ix in enumerate(indices) if k not in (i, j)]


def _sort_axes(arr, indices: list[Index]):
    if len(indices) < 2:
        return arr, indices
    order = sorted(range(len(indices)), key=lambda k: indices[k].name)
    if order == list(range(len(indices))):
        return arr, indices
    arr = sp.permutedims(arr, order)
    return arr, [indices[k] for k in order]


def evaluate(
    expr: Expr,
    geom: ComponentGeometry,
    scalars: dict[str, sp.Expr] | None = None,
    fields: dict[str, tuple[Array | sp.Expr, list[str]]] | None = None,
):
    """Return (value, free_indices). Rank 0 values are plain sympy exprs;
    higher ranks are Arrays with axes sorted by index name.

    `fields` binds extra named tensors: name -> (components, base_variances),
    e.g. {"F": (antisym_array, ["down", "down"]), "phi": (expr, [])}.
    """
    scalars = dict(scalars or {})
    fields = dict(fields or {})
    for name, (val, variances) in fields.items():
        if not variances:
            scalars.setdefault(name, val)

    def ev(e: Expr):
        match e:
            case Num(p=p, q=q):
                return sp.Rational(p, q), []
            case Sym(name=name):
                if name not in scalars:
                    raise EvaluationError(f"no value bound for scalar {name!r}")
                return scalars[name], []
            case Func():
                raise EvaluationError("function nodes are not supported by the component evaluator")
            case Pow(base=base, exp=exp):
                val, free = ev(base)
                if free:
                    raise EvaluationError("power of an indexed expression")
                return val**exp, []
            case Tensor(name=name, indices=indices):
                rank = len(indices)
                if name in fields and len(fields[name][1]) == rank:
                    arr, base = fields[name]
                else:
                    arr = _base_components(name, rank, geom)
                    base = _BASE_VARIANCES.get((name, rank), [])
                if rank == 0:
                    return arr, []
                arr = _adjust_variances(arr, base, [ix.variance for ix in indices], geom)
                arr, remaining = _contract_duplicates(arr, list(indices), geom)
                if not remaining:
                    return arr, []
                return _sort_axes(arr, remaining)
            case Deriv(op=op, index=index, expr=inner):
                if op != "covariant":
                    raise EvaluationError("component evaluator handles covariant derivatives only")
                val, free = ev(inner)
                arr = geom.covariant_derivative(val, [ix.variance for ix in free])
                if index.variance == "up":
                    arr = geom.raise_first_index(arr, 0)
                indices = [index, *free]
                arr, remaining = _contract_duplicates(arr, indices, geom)
                if not remaining:
                    return arr, []
                return _sort_axes(arr, remaining)
            case Prod(factors=factors):
                acc_val: sp.Expr | None = None
                acc_free: list[Index] = []
                scalar_part = sp.Integer(1)
                for f in factors:
                    val, free = ev(f)
                    if not free:
                        scalar_part *= val
                        continue
                    if acc_val is None:
                        acc_val, acc_free = val, list(free)
                    else:
                        acc_val = sp.tensorproduct(acc_val, val)
                        acc_free = acc_free + list(free)
                if acc_val is None:
                    return scalar_part, []
                if scalar_part != 1:
                    acc_val = _scale(acc_val, scalar_part)
                acc_val, remaining = _contract_duplicates(acc_val, acc_free, geom)
                if not remaining:
                    return acc_val, []
                return _sort_axes(acc_val, remaining)
            case Sum(terms=terms):
                if not terms:
                    return sp.Integer(0), []
                vals = [ev(t) for t in terms]
                _, free0 = vals[0]
                total = None
                for val, free in vals:
                    if [ix.name for ix in free] != [ix.name for ix in free0]:
                        raise EvaluationError("sum terms disagree on free indices")
                    total = val if total is None else _add(total, val)
                return total, free0
        raise EvaluationError(f"unknown node {e!r}")

    return ev(expr)


def _scale(arr, scalar):
    return scalar * arr  # NDimArray supports scalar multiplication


def _add(a, b):
    return a + b  # NDimArray supports elementwise addition


def _is_zero(component) -> tuple[bool, sp.Expr]:
    """Exact zero test: cancel() is canonical for rational expressions, so it
    decides most cases fast; full simplify() only runs on what survives."""
    c = sp.cancel(sp.together(component))
    if c == 0:
        return True, c
    s = sp.simplify(c)
    return (s == 0), s


def all_zero(value) -> tuple[bool, str]:
    shape = getattr(value, "shape", ())
    if not shape:
        ok, s = _is_zero(value)
        return ok, f"scalar residue: {s}"
    worst = None
    n_nonzero = 0
    for idx in _all_indices(shape[0], len(shape)):
        ok, s = _is_zero(value[idx])
        if not ok:
            n_nonzero += 1
            worst = (idx, s)
    if n_nonzero == 0:
        return True, "all components vanish"
    return False, f"{n_nonzero} nonzero components, e.g. {worst[0]}: {worst[1]}"
