"""Structural validation of NPR expressions (verification rung V0, in part).

Checks index balance: within a product, an index name may appear at most twice
and only as an up/down pair (abstract-index contraction); across the terms of
a sum, free indices must agree exactly.
"""

from collections import Counter

from noether.npr.ast import Deriv, Expr, Func, Index, Num, Pow, Prod, Sum, Sym, Tensor


class ValidationError(ValueError):
    pass


def _contract(counts: Counter) -> Counter:
    """Cancel up/down pairs; reject same-variance repeats or >2 occurrences."""
    result: Counter = Counter()
    names = {name for (name, _v) in counts}
    for name in names:
        n_up = counts[(name, "up")]
        n_down = counts[(name, "down")]
        if n_up > 1 or n_down > 1:
            raise ValidationError(
                f"index '{name}' repeated with the same variance "
                f"(up x{n_up}, down x{n_down}); not a valid contraction"
            )
        if n_up == 1 and n_down == 1:
            continue  # contracted pair
        if n_up:
            result[(name, "up")] = 1
        if n_down:
            result[(name, "down")] = 1
    return result


def free_indices(expr: Expr, *, metric_compatible: bool | None = None) -> Counter:
    """Multiset of free (name, variance) pairs; raises on malformed index use.

    This is intentionally a structural check. Unless ``metric_compatible`` is
    explicitly set, validation never applies any rule that would commute the
    metric through a covariant derivative or otherwise treat index
    raising/lowering across ``nabla`` as free.
    """
    match expr:
        case Num() | Sym():
            return Counter()
        case Func(args=args):
            for a in args:
                fa = free_indices(a, metric_compatible=metric_compatible)
                if fa:
                    raise ValidationError(
                        f"function argument carries free indices {sorted(fa)}; "
                        "function arguments must be scalars"
                    )
            return Counter()
        case Tensor(indices=indices):
            counts: Counter = Counter()
            for ix in indices:
                counts[(ix.name, ix.variance)] += 1
            return _contract(counts)
        case Deriv(index=index, expr=inner):
            counts = Counter()
            for (name, v), n in free_indices(inner, metric_compatible=metric_compatible).items():
                counts[(name, v)] += n
            counts[(index.name, index.variance)] += 1
            return _contract(counts)
        case Pow(base=base, exp=_):
            fb = free_indices(base, metric_compatible=metric_compatible)
            if fb:
                raise ValidationError(
                    f"power of an expression with free indices {sorted(fb)} is ambiguous; "
                    "contract explicitly instead"
                )
            return Counter()
        case Prod(factors=factors):
            counts = Counter()
            for f in factors:
                for key, n in free_indices(f, metric_compatible=metric_compatible).items():
                    counts[key] += n
            return _contract(counts)
        case Sum(terms=terms):
            if not terms:
                return Counter()
            first = free_indices(terms[0], metric_compatible=metric_compatible)
            for t in terms[1:]:
                ft = free_indices(t, metric_compatible=metric_compatible)
                if ft != first:
                    raise ValidationError(
                        f"sum terms have mismatched free indices: {sorted(first)} vs {sorted(ft)}"
                    )
            return first
    raise ValidationError(f"unknown node: {expr!r}")


def validate_expression(
    expr: Expr,
    expected_free: list[Index] | None = None,
    *,
    metric_compatible: bool | None = None,
) -> None:
    """Raise ValidationError if the expression is structurally ill-formed."""
    free = free_indices(expr, metric_compatible=metric_compatible)
    if expected_free is not None:
        expected = Counter((ix.name, ix.variance) for ix in expected_free)
        if free != expected:
            raise ValidationError(
                f"free indices {sorted(free)} do not match expected {sorted(expected)}"
            )
