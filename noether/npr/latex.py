"""Deterministic LaTeX rendering of NPR expressions.

Determinism law (docs/03_METHODOLOGY.md section 3): same NPR in,
byte-identical LaTeX out. Rendering must never reorder physics content
non-deterministically; sums render in stored order (the good-form pipeline,
not the renderer, owns canonical term ordering).
"""

from noether.npr.ast import Deriv, Expr, Func, Index, Num, Pow, Prod, Sum, Sym, Tensor

_GREEK = {
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
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Phi",
    "Psi",
    "Omega",
}

_NAME_OVERRIDES = {
    "Box": r"\Box",
    "sqrt(-g)": r"\sqrt{-g}",
    "nabla": r"\nabla",
}


def _name_tex(name: str) -> str:
    if name in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[name]
    if name in _GREEK:
        return "\\" + name
    return name


def _index_tex(ix: Index) -> str:
    return _name_tex(ix.name)


def _tensor_tex(t: Tensor) -> str:
    """Render index groups in declared order: R^{rho}{}_{sigma mu nu}."""
    if not t.indices:
        return _name_tex(t.name)
    out = _name_tex(t.name)
    i = 0
    first_group = True
    while i < len(t.indices):
        variance = t.indices[i].variance
        group = []
        while i < len(t.indices) and t.indices[i].variance == variance:
            group.append(_index_tex(t.indices[i]))
            i += 1
        sym = "^" if variance == "up" else "_"
        joiner = "" if not first_group else ""
        out += f"{joiner}{'' if first_group else '{}'}{sym}{{{' '.join(group)}}}"
        first_group = False
    return out


def _needs_parens_in_product(e: Expr) -> bool:
    return isinstance(e, Sum)


def render(expr: Expr) -> str:
    match expr:
        case Num(p=p, q=q):
            if q == 1:
                return str(p)
            sign = "-" if (p < 0) != (q < 0) else ""
            return f"{sign}\\tfrac{{{abs(p)}}}{{{abs(q)}}}"
        case Sym(name=name):
            return _name_tex(name)
        case Func(name=name, args=args):
            inner = ", ".join(render(a) for a in args)
            return f"{_name_tex(name)}({inner})"
        case Tensor():
            return _tensor_tex(expr)
        case Deriv(op=op, index=index, expr=inner):
            op_tex = r"\nabla" if op == "covariant" else r"\partial"
            sym = "^" if index.variance == "up" else "_"
            body = render(inner)
            if isinstance(inner, Sum | Prod):
                body = f"\\left( {body} \\right)"
            return f"{op_tex}{sym}{{{_index_tex(index)}}} {body}"
        case Pow(base=base, exp=exp):
            body = render(base)
            if not isinstance(base, Sym | Num) or (isinstance(base, Num) and base.q != 1):
                body = f"\\left( {body} \\right)"
            return f"{body}^{{{exp}}}"
        case Prod(factors=factors):
            parts = []
            for f in factors:
                body = render(f)
                if _needs_parens_in_product(f):
                    body = f"\\left( {body} \\right)"
                parts.append(body)
            return " \\, ".join(parts)
        case Sum(terms=terms):
            if not terms:
                return "0"
            out = render(terms[0])
            for t in terms[1:]:
                body = render(t)
                if body.startswith("-"):
                    out += f" {body}"
                else:
                    out += f" + {body}"
            return out
    raise ValueError(f"unknown node: {expr!r}")
