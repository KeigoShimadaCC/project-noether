"""Compositional scalar-EOM derivation by Lagrangian building blocks.

This is the non-tailored counterpart to a per-theory template. An additive
scalar Lagrangian is decomposed into recognized building blocks (canonical
kinetic, potential, cubic-Galileon coupling, k-essence), each carrying a free
coupling function of its own. Variation is linear, so the equation of motion
of the sum is the sum of each block's contribution.

Crucially, the trust does not rest on summing pre-blessed formulas. The
decomposition assembles ONE Cadabra script that carries the user's actual
action (the exact couplings and coefficients the parser found) and an
independently stated candidate equation built from the same blocks. The
kernel's residue check then verifies the real assembled action, exactly as
the frozen templates do. A block whose candidate were wrong would leave a
nonzero residue. So adding a block extends coverage without weakening the
no-asserted-result rule (AGENTS.md rule 1): the kernel still decides.

What composes today is the scalar sector under noether-default-v1:

  - kinetic   c (nabla phi)^2                 -> -2c box phi
  - potential c U(phi)                        ->  c U_phi
  - cubic     c G(phi) box phi (Horndeski G3) ->  c (2 G_phi box phi
                                                     + G_phiphi (nabla phi)^2)
  - kessence  c K(phi, X)      (Horndeski G2) ->  c (K_phi + K_X box phi
                                                     + K_Xphi (nabla phi)^2
                                                     - K_XX nabla^a phi nabla^b phi
                                                            nabla_a nabla_b phi)

where X = -1/2 (nabla phi)^2 is expanded to its primitive in-kernel and only
collapsed back to the shorthand for display (the operational-definition path).
A term matching no block leaves the decomposition incomplete and the caller
refuses rather than guessing (rule 4). Curvature-coupled blocks (G4, G5,
nonminimal F(phi) R) are not registered yet; those actions fall back to the
model-written script path or are refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from noether.npr.ast import Deriv, Expr, Func, Num, Prod, Sum, Sym

KINETIC = "kinetic"
POTENTIAL = "potential"
CUBIC = "cubic"
KESSENCE = "kessence"


@dataclass(frozen=True)
class BlockMatch:
    """One Lagrangian term recognized as a building block."""

    block: str
    coeff: Fraction
    coupling: str | None = None  # None for the canonical kinetic block


@dataclass
class Decomposition:
    """The result of decomposing a scalar Lagrangian for one field."""

    field: str
    matches: list[BlockMatch] = field(default_factory=list)
    unmatched: list[Expr] = field(default_factory=list)

    @property
    def full(self) -> bool:
        return not self.unmatched and bool(self.matches)


# -- term inspection ---------------------------------------------------------


def _terms(lag: Expr) -> list[Expr]:
    return list(lag.terms) if isinstance(lag, Sum) else [lag]


def _split_coeff(term: Expr) -> tuple[Fraction, list[Expr]]:
    """Pull the rational numeric coefficient out of a term, returning it with
    the remaining (non-numeric) factors."""
    if isinstance(term, Num):
        return Fraction(term.p, term.q), []
    if isinstance(term, Prod):
        coeff = Fraction(1, 1)
        rest: list[Expr] = []
        for f in term.factors:
            if isinstance(f, Num):
                coeff *= Fraction(f.p, f.q)
            else:
                rest.append(f)
        return coeff, rest
    return Fraction(1, 1), [term]


def _is_first_cov(e: Expr, fieldname: str) -> bool:
    return (
        isinstance(e, Deriv)
        and e.op == "covariant"
        and isinstance(e.expr, Sym)
        and e.expr.name == fieldname
    )


def _is_box(e: Expr, fieldname: str) -> bool:
    return (
        isinstance(e, Deriv)
        and e.op == "covariant"
        and isinstance(e.expr, Deriv)
        and e.expr.op == "covariant"
        and isinstance(e.expr.expr, Sym)
        and e.expr.expr.name == fieldname
    )


def _is_grad_sq(factors: list[Expr], fieldname: str) -> bool:
    """Two first covariant derivatives of the field with a contracted index,
    i.e. nabla_a phi nabla^a phi."""
    if len(factors) != 2 or not all(_is_first_cov(f, fieldname) for f in factors):
        return False
    i0, i1 = factors[0].index, factors[1].index  # type: ignore[attr-defined]
    return i0.name == i1.name and i0.variance != i1.variance


def _scalar_func(factors: list[Expr], fieldname: str) -> Func | None:
    """The single Func factor if `factors` is exactly one function."""
    if len(factors) == 1 and isinstance(factors[0], Func):
        return factors[0]
    return None


def _func_args(fn: Func) -> set[str]:
    return {a.name for a in fn.args if isinstance(a, Sym)}


# -- matching ----------------------------------------------------------------


def _match_term(term: Expr, fieldname: str) -> BlockMatch | None:
    coeff, rest = _split_coeff(term)

    if _is_grad_sq(rest, fieldname):
        return BlockMatch(KINETIC, coeff, None)

    # k-essence: a single function of (phi, X). Requires both, so it is the
    # general Horndeski G2; a pure K(X) or K(phi) is handled elsewhere.
    fn = _scalar_func(rest, fieldname)
    if fn is not None:
        args = _func_args(fn)
        if fieldname in args and "X" in args:
            return BlockMatch(KESSENCE, coeff, fn.name)
        if args == {fieldname}:
            return BlockMatch(POTENTIAL, coeff, fn.name)

    # cubic Galileon: G(phi) box phi.
    if len(rest) == 2:
        funcs = [f for f in rest if isinstance(f, Func)]
        boxes = [f for f in rest if _is_box(f, fieldname)]
        if len(funcs) == 1 and len(boxes) == 1 and _func_args(funcs[0]) == {fieldname}:
            return BlockMatch(CUBIC, coeff, funcs[0].name)

    return None


def decompose_scalar(lag: Expr, fieldname: str) -> Decomposition:
    """Decompose a scalar Lagrangian into building blocks for `fieldname`."""
    dec = Decomposition(field=fieldname)
    for term in _terms(lag):
        match = _match_term(term, fieldname)
        if match is None:
            dec.unmatched.append(term)
        else:
            dec.matches.append(match)
    return dec


# -- kernel-symbol naming ----------------------------------------------------


def _base(coupling: str) -> str:
    """A safe Cadabra identifier base for a coupling name (alnum only)."""
    return "".join(ch for ch in coupling if ch.isalnum()) or "F"


def coupling_symbols(match: BlockMatch) -> dict[str, str]:
    """Kernel symbol names for a block's coupling derivatives."""
    if match.coupling is None:
        return {}
    b = _base(match.coupling)
    if match.block == POTENTIAL:
        return {"d1": f"{b}p"}
    if match.block == CUBIC:
        return {"d1": f"{b}p", "d2": f"{b}pp"}
    if match.block == KESSENCE:
        return {
            "phi": f"{b}phi",
            "X": f"{b}X",
            "Xphi": f"{b}Xphi",
            "XX": f"{b}XX",
        }
    return {}


# -- script assembly ---------------------------------------------------------

_DECL_HEAD = r"""{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}")."""


def _render_cadabra(monomials: list[tuple[Fraction, str]]) -> str:
    """Render signed rational monomials as a Cadabra sum string."""
    out = ""
    for coeff, body in monomials:
        if coeff == 0:
            continue
        sign = "-" if coeff < 0 else "+"
        mag = abs(coeff)
        if mag == 1:
            magstr = ""
        elif mag.denominator == 1:
            magstr = f"{mag.numerator} "
        else:
            magstr = f"{mag.numerator}/{mag.denominator} "
        piece = magstr + body
        out = (("- " if sign == "-" else "") + piece) if out == "" else f"{out} {sign} {piece}"
    return out or "0"


def _integrand_monomial(match: BlockMatch) -> tuple[Fraction, str]:
    c = match.coeff
    if match.block == KINETIC:
        return (c, r"sg g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi}")
    name = match.coupling
    if match.block == POTENTIAL:
        return (c, f"sg {name}")
    if match.block == CUBIC:
        return (
            c,
            f"sg {name} g^{{\\alpha\\beta}} \\nabla_{{\\alpha}}{{\\nabla_{{\\beta}}{{\\phi}}}}",
        )
    if match.block == KESSENCE:
        return (c, f"sg {name}")
    raise ValueError(match.block)


def _target_monomials(match: BlockMatch) -> list[tuple[Fraction, str]]:
    c = match.coeff
    box = r"sg g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{\phi}} dphi"
    if match.block == KINETIC:
        return [(-2 * c, box)]
    sym = coupling_symbols(match)
    if match.block == POTENTIAL:
        return [(c, f"sg {sym['d1']} dphi")]
    if match.block == CUBIC:
        return [
            (
                2 * c,
                f"sg {sym['d1']} g^{{\\alpha\\beta}} \\nabla_{{\\alpha}}{{\\nabla_{{\\beta}}{{\\phi}}}} dphi",
            ),
            (
                c,
                f"sg {sym['d2']} g^{{\\alpha\\beta}} \\nabla_{{\\alpha}}{{\\phi}} \\nabla_{{\\beta}}{{\\phi}} dphi",
            ),
        ]
    if match.block == KESSENCE:
        return [
            (c, f"sg {sym['phi']} dphi"),
            (
                c,
                f"sg {sym['X']} g^{{\\alpha\\beta}} \\nabla_{{\\alpha}}{{\\nabla_{{\\beta}}{{\\phi}}}} dphi",
            ),
            (
                c,
                f"sg {sym['Xphi']} g^{{\\alpha\\beta}} \\nabla_{{\\alpha}}{{\\phi}} \\nabla_{{\\beta}}{{\\phi}} dphi",
            ),
            (
                -c,
                f"sg {sym['XX']} g^{{\\alpha\\beta}} g^{{\\gamma\\sigma}} "
                r"\nabla_{\alpha}{\nabla_{\gamma}{\phi}} \nabla_{\beta}{\phi} \nabla_{\sigma}{\phi} dphi",
            ),
        ]
    raise ValueError(match.block)


def _declarations(matches: list[BlockMatch]) -> str:
    depends = {"\\phi", "dphi"}
    has_kessence = False
    for m in matches:
        for s in coupling_symbols(m).values():
            depends.add(s)
        if m.coupling is not None:
            depends.add(_base(m.coupling) if m.block != POTENTIAL else m.coupling)
            depends.add(m.coupling)
        if m.block == KESSENCE:
            has_kessence = True
    if has_kessence:
        depends.update({"X", "dX"})
    ordered = ", ".join(sorted(depends, key=lambda s: (s.startswith("\\"), s)))
    return f"{_DECL_HEAD}\n{{{ordered}}}::Depends(\\nabla{{#}})."


def _vary_rules(matches: list[BlockMatch]) -> str:
    rules = [r"\phi -> dphi"]
    for m in matches:
        sym = coupling_symbols(m)
        if m.block == POTENTIAL:
            rules.append(f"{m.coupling} -> {sym['d1']} dphi")
        elif m.block == CUBIC:
            rules.append(f"{m.coupling} -> {sym['d1']} dphi")
        elif m.block == KESSENCE:
            rules.append(f"{m.coupling} -> {sym['phi']} dphi + {sym['X']} dX")
    return ", ".join(rules)


_METRIC_SUBS = (
    r"substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);"
    "\n"
    r"substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);"
)


def assemble_scalar_eom_script(matches: list[BlockMatch], fieldname: str = "phi") -> str:
    """Assemble a single Cadabra script that derives and residue-checks the
    scalar EOM of the matched blocks. The script carries the real action and
    an independent candidate; the kernel's residue_zero is the verdict."""
    decl = _declarations(matches)
    integrand = _render_cadabra([_integrand_monomial(m) for m in matches])
    vary_rules = _vary_rules(matches)
    has_kessence = any(m.block == KESSENCE for m in matches)

    lines: list[str] = [
        decl,
        "",
        f"ex := \\int{{ {integrand} }}{{x}};",
        f"vary(ex, ${vary_rules}$);",
    ]
    if has_kessence:
        lines.append(
            r"substitute(ex, $dX -> - g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{dphi}$);"
        )
    lines += ["distribute(ex);", "product_rule(ex);", _METRIC_SUBS, "canonicalise(ex);", ""]

    # Pass 1: strip the outer derivative off any box(dphi) (cubic) and the
    # single derivative off the kinetic / k-essence pieces.
    lines += [
        r"integrate_by_parts(ex, $\nabla_{\beta}{dphi}$);",
        "product_rule(ex);",
        "distribute(ex);",
        _METRIC_SUBS,
        r"substitute(ex, $\nabla_{\mu}{sg} -> 0$);",
    ]
    for m in matches:
        if m.block == CUBIC:
            sym = coupling_symbols(m)
            lines.append(
                f"substitute(ex, $\\nabla_{{\\mu}}{{{m.coupling}}} -> {sym['d1']} \\nabla_{{\\mu}}{{\\phi}}$);"
            )
    lines += ["canonicalise(ex);", ""]

    # Pass 2: strip the remaining derivative off dphi and expand couplings.
    lines += [
        r"integrate_by_parts(ex, $dphi$);",
        "product_rule(ex);",
        "distribute(ex);",
        _METRIC_SUBS,
        r"substitute(ex, $\nabla_{\mu}{sg} -> 0$);",
    ]
    for m in matches:
        sym = coupling_symbols(m)
        if m.block == CUBIC:
            lines.append(
                f"substitute(ex, $\\nabla_{{\\mu}}{{{sym['d1']}}} -> {sym['d2']} \\nabla_{{\\mu}}{{\\phi}}$);"
            )
            lines.append(
                f"substitute(ex, $\\nabla_{{\\mu}}{{{m.coupling}}} -> {sym['d1']} \\nabla_{{\\mu}}{{\\phi}}$);"
            )
        elif m.block == KESSENCE:
            lines.append(
                f"substitute(ex, $\\nabla_{{\\mu}}{{{sym['X']}}} -> "
                f"{sym['Xphi']} \\nabla_{{\\mu}}{{\\phi}} + {sym['XX']} \\nabla_{{\\mu}}{{X}}$);"
            )
            lines.append(
                r"substitute(ex, $\nabla_{\mu}{X} -> "
                r"- g^{\alpha\beta} \nabla_{\mu}{\nabla_{\alpha}{\phi}} \nabla_{\beta}{\phi}$);"
            )
    lines += [
        r"substitute(ex, $\int{A??}{x} -> A??$);",
        "eliminate_kronecker(ex);",
        "sort_product(ex);",
        "canonicalise(ex);",
        "rename_dummies(ex);",
        'print("NOETHER_RESULT: " + str(ex))',
        "",
    ]

    target = _render_cadabra([mono for m in matches for mono in _target_monomials(m)])
    lines += [
        f"target := {target};",
        "distribute(target);",
        "eliminate_kronecker(target);",
        "sort_product(target);",
        "canonicalise(target);",
        "rename_dummies(target);",
        "",
        "residue := @(ex) - @(target);",
        "distribute(residue);",
        "eliminate_kronecker(residue);",
        "sort_product(residue);",
        "canonicalise(residue);",
        "rename_dummies(residue);",
        "meld(residue);",
        'print("NOETHER_CHECK: residue=" + str(residue))',
        'print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))',
    ]
    return "\n".join(lines) + "\n"


# -- display (operational-definition collapse) -------------------------------


def _render_tex(monomials: list[tuple[Fraction, str]]) -> str:
    out = ""
    for coeff, body in monomials:
        if coeff == 0:
            continue
        sign = "-" if coeff < 0 else "+"
        mag = abs(coeff)
        if mag == 1:
            magstr = ""
        elif mag.denominator == 1:
            magstr = f"{mag.numerator} "
        else:
            magstr = f"\\tfrac{{{mag.numerator}}}{{{mag.denominator}}} "
        piece = magstr + body
        out = (("-" + piece) if sign == "-" else piece) if out == "" else f"{out} {sign} {piece}"
    return out or "0"


def _display_monomials(match: BlockMatch) -> list[tuple[Fraction, str]]:
    c = match.coeff
    if match.block == KINETIC:
        return [(-2 * c, r"\Box\phi")]
    name = match.coupling
    if match.block == POTENTIAL:
        return [(c, f"{name}_{{\\phi}}")]
    if match.block == CUBIC:
        return [
            (2 * c, f"{name}_{{\\phi}}\\Box\\phi"),
            (c, f"{name}_{{\\phi\\phi}}(\\nabla\\phi)^2"),
        ]
    if match.block == KESSENCE:
        return [
            (c, f"{name}_{{\\phi}}"),
            (c, f"{name}_{{X}}\\Box\\phi"),
            (c, f"{name}_{{X\\phi}}(\\nabla\\phi)^2"),
            (
                -c,
                f"{name}_{{XX}}\\,\\nabla^{{\\mu}}\\phi\\,\\nabla^{{\\nu}}\\phi\\,\\nabla_{{\\mu}}\\nabla_{{\\nu}}\\phi",
            ),
        ]
    raise ValueError(match.block)


def compose_display_tex(matches: list[BlockMatch], fieldname: str = "phi") -> str:
    """The composed EOM in clean shorthand (X and box collapsed), for display.
    The kernel verified the primitive expansion; this is the human-facing form."""
    monomials = [mono for m in matches for mono in _display_monomials(m)]
    return f"{_render_tex(monomials)} = 0"


def block_summary(matches: list[BlockMatch]) -> list[str]:
    """Human-readable provenance: which block matched each coupling."""
    labels = {
        KINETIC: "canonical kinetic",
        POTENTIAL: "potential",
        CUBIC: "cubic Galileon (Horndeski G3)",
        KESSENCE: "k-essence (Horndeski G2)",
    }
    out = []
    for m in matches:
        name = f" in {m.coupling}" if m.coupling else ""
        out.append(f"{labels[m.block]}{name}")
    return out
