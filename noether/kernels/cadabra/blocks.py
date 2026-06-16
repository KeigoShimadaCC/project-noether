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
A nonminimal F(phi) R term adds F_phi R to the scalar EOM.

The metric equation of motion (delta S / delta g = 0) composes the same way,
through the second half of this module: an additive Lagrangian decomposes into
Einstein-Hilbert (R), nonminimal F(phi) R, kinetic, and potential blocks, and
the eval-3 metric-variation machinery (vary into dGamma, two IBP passes, lower
h to one explicit-g convention) is assembled once and residue-checked. So the
full nonminimal scalar-tensor theory yields both equations of motion with no
per-theory template.

A term matching no block leaves the decomposition incomplete and the caller
refuses rather than guessing (rule 4). The higher Horndeski densities (an
X-dependent G4(phi, X) R, and G5) are held out until they verify, so they match
no block and fall back to the model-written script path or are refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from noether.npr.ast import Deriv, Expr, Func, Num, Prod, Sum, Sym, Tensor

KINETIC = "kinetic"
POTENTIAL = "potential"
CUBIC = "cubic"
KESSENCE = "kessence"
NONMINIMAL = "nonminimal"  # F(phi) R
EINSTEIN_HILBERT = "einstein_hilbert"  # bare R


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


def _is_curvature_scalar(e: Expr) -> bool:
    """The bare Ricci scalar R (a tensor with no free indices)."""
    return isinstance(e, Tensor) and e.name == "R" and not e.indices


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

    # bare Ricci scalar: an Einstein-Hilbert term, inert under phi-variation.
    if len(rest) == 1 and _is_curvature_scalar(rest[0]):
        return BlockMatch(EINSTEIN_HILBERT, coeff, None)

    # k-essence: a single function of (phi, X). Requires both, so it is the
    # general Horndeski G2; a pure K(X) or K(phi) is handled elsewhere.
    fn = _scalar_func(rest, fieldname)
    if fn is not None:
        args = _func_args(fn)
        if fieldname in args and "X" in args:
            return BlockMatch(KESSENCE, coeff, fn.name)
        if args == {fieldname}:
            return BlockMatch(POTENTIAL, coeff, fn.name)

    if len(rest) == 2:
        funcs = [f for f in rest if isinstance(f, Func)]
        # nonminimal coupling F(phi) R: a curvature scalar times a function of
        # phi alone. A function of (phi, X) here would be Horndeski G4, which is
        # held out until it verifies, so it stays unmatched and falls back.
        curvs = [f for f in rest if _is_curvature_scalar(f)]
        if len(funcs) == 1 and len(curvs) == 1 and _func_args(funcs[0]) == {fieldname}:
            return BlockMatch(NONMINIMAL, coeff, funcs[0].name)
        # cubic Galileon: G(phi) box phi.
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
    if match.block in (POTENTIAL, NONMINIMAL):
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


def _integrand_monomial(match: BlockMatch) -> tuple[Fraction, str] | None:
    c = match.coeff
    if match.block == KINETIC:
        return (c, r"sg g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi}")
    if match.block == EINSTEIN_HILBERT:
        # inert under phi-variation; carried for completeness, contributes nothing.
        return (c, "sg R")
    name = match.coupling
    if match.block == POTENTIAL:
        return (c, f"sg {name}")
    if match.block == NONMINIMAL:
        return (c, f"sg {name} R")
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
    if match.block == EINSTEIN_HILBERT:
        return []
    sym = coupling_symbols(match)
    if match.block == POTENTIAL:
        return [(c, f"sg {sym['d1']} dphi")]
    if match.block == NONMINIMAL:
        return [(c, f"sg {sym['d1']} R dphi")]
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
    has_curvature = False
    for m in matches:
        for s in coupling_symbols(m).values():
            depends.add(s)
        if m.coupling is not None:
            depends.add(_base(m.coupling) if m.block != POTENTIAL else m.coupling)
            depends.add(m.coupling)
        if m.block == KESSENCE:
            has_kessence = True
        if m.block in (NONMINIMAL, EINSTEIN_HILBERT):
            has_curvature = True
    if has_kessence:
        depends.update({"X", "dX"})
    if has_curvature:
        depends.add("R")
    ordered = ", ".join(sorted(depends, key=lambda s: (s.startswith("\\"), s)))
    return f"{_DECL_HEAD}\n{{{ordered}}}::Depends(\\nabla{{#}})."


def _vary_rules(matches: list[BlockMatch]) -> str:
    rules = [r"\phi -> dphi"]
    for m in matches:
        sym = coupling_symbols(m)
        if m.block in (POTENTIAL, NONMINIMAL):
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
    integrand = _render_cadabra(
        [mono for m in matches if (mono := _integrand_monomial(m)) is not None]
    )
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
    if match.block == EINSTEIN_HILBERT:
        return []
    name = match.coupling
    if match.block == POTENTIAL:
        return [(c, f"{name}_{{\\phi}}")]
    if match.block == NONMINIMAL:
        return [(c, f"{name}_{{\\phi}} R")]
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
        NONMINIMAL: "nonminimal coupling F(phi) R",
        EINSTEIN_HILBERT: "Einstein-Hilbert",
    }
    out = []
    for m in matches:
        name = f" in {m.coupling}" if m.coupling else ""
        out.append(f"{labels[m.block]}{name}")
    return out


# ===========================================================================
# Metric sector: the gravitational equation of motion delta S / delta g = 0.
#
# The same additive Lagrangian is decomposed for the metric variation. The
# derivation machinery (vary g^{ab} -> -h, the Ricci-tensor variation into
# dGamma, two integration-by-parts passes, lowering h to one explicit-g
# convention) is generic; only the integrand and the candidate EOM tensor
# change per block. The blocks that compose are Einstein-Hilbert (R),
# nonminimal F(phi) R, canonical kinetic, and potential, in any additive
# combination. A G4(phi, X) R term (a function that also depends on X) matches
# no block and is held out, so the action falls back to the model path.
#
# Block contributions to the EOM tensor E_{mu nu} = 0 (noether-default-v1):
#   Einstein-Hilbert c R : c (R_{mu nu} - 1/2 g_{mu nu} R)
#   nonminimal c F R     : c (F R_{mu nu} - 1/2 g_{mu nu} F R
#                              + g_{mu nu} box F - nabla_mu nabla_nu F)
#   kinetic c (nabla phi)^2 : c (nabla_mu phi nabla_nu phi
#                                  - 1/2 g_{mu nu} (nabla phi)^2)
#   potential c V        : -c/2 g_{mu nu} V
# ===========================================================================


def _match_metric_term(term: Expr, fieldname: str) -> BlockMatch | None:
    coeff, rest = _split_coeff(term)

    if len(rest) == 1 and _is_curvature_scalar(rest[0]):
        return BlockMatch(EINSTEIN_HILBERT, coeff, None)

    if _is_grad_sq(rest, fieldname):
        return BlockMatch(KINETIC, coeff, None)

    fn = _scalar_func(rest, fieldname)
    if fn is not None and _func_args(fn) == {fieldname}:
        return BlockMatch(POTENTIAL, coeff, fn.name)

    if len(rest) == 2:
        funcs = [f for f in rest if isinstance(f, Func)]
        curvs = [f for f in rest if _is_curvature_scalar(f)]
        # nonminimal F(phi) R; a function of (phi, X) here is Horndeski G4, held
        # out until it verifies, so it stays unmatched and the caller falls back.
        if len(funcs) == 1 and len(curvs) == 1 and _func_args(funcs[0]) == {fieldname}:
            return BlockMatch(NONMINIMAL, coeff, funcs[0].name)

    return None


def decompose_metric(lag: Expr, fieldname: str = "phi") -> Decomposition:
    """Decompose a Lagrangian into metric-sector building blocks. `fieldname`
    is the dynamical scalar (rendered as phi in the assembled script)."""
    dec = Decomposition(field="g")
    for term in _terms(lag):
        match = _match_metric_term(term, fieldname)
        if match is None:
            dec.unmatched.append(term)
        else:
            dec.matches.append(match)
    return dec


def _metric_integrand_monomial(match: BlockMatch) -> tuple[Fraction, str]:
    c = match.coeff
    if match.block == EINSTEIN_HILBERT:
        return (c, r"sg g^{\alpha\beta} R_{\alpha\beta}")
    if match.block == NONMINIMAL:
        return (c, f"sg {match.coupling} g^{{\\alpha\\beta}} R_{{\\alpha\\beta}}")
    if match.block == KINETIC:
        return (c, r"sg g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi}")
    if match.block == POTENTIAL:
        return (c, f"sg {match.coupling}")
    raise ValueError(match.block)


def _metric_target_monomials(match: BlockMatch) -> list[tuple[Fraction, str]]:
    c = match.coeff
    if match.block == EINSTEIN_HILBERT:
        return [
            (-c, r"sg R_{\mu\nu} h^{\mu\nu}"),
            (c / 2, r"sg g^{\mu\nu} h_{\mu\nu} g^{\alpha\beta} R_{\alpha\beta}"),
        ]
    if match.block == NONMINIMAL:
        n = match.coupling
        ddF = f"\\nabla_{{\\alpha}}{{\\nabla_{{\\beta}}{{{n}}}}}"
        return [
            (-c, f"sg {n} R_{{\\mu\\nu}} h^{{\\mu\\nu}}"),
            (
                c / 2,
                f"sg {n} g^{{\\mu\\nu}} h_{{\\mu\\nu}} g^{{\\alpha\\beta}} R_{{\\alpha\\beta}}",
            ),
            (-c, f"sg g^{{\\mu\\nu}} h_{{\\mu\\nu}} g^{{\\alpha\\beta}} {ddF}"),
            (c, f"sg h_{{\\mu\\nu}} g^{{\\mu\\alpha}} g^{{\\nu\\beta}} {ddF}"),
        ]
    if match.block == KINETIC:
        return [
            (
                -c,
                r"sg h_{\mu\nu} g^{\mu\alpha} g^{\nu\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi}",
            ),
            (
                c / 2,
                r"sg g^{\mu\nu} h_{\mu\nu} g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi}",
            ),
        ]
    if match.block == POTENTIAL:
        return [(c / 2, f"sg g^{{\\mu\\nu}} h_{{\\mu\\nu}} {match.coupling}")]
    raise ValueError(match.block)


def _metric_declarations(matches: list[BlockMatch]) -> str:
    depends = {"h{#}", "R_{\\mu\\nu}", "dGamma^{\\lambda}_{\\mu\\nu}"}
    for m in matches:
        if m.block in (NONMINIMAL, KINETIC):
            depends.add("\\phi")
        if m.coupling is not None:
            depends.add(m.coupling)
    ordered = ", ".join(sorted(depends, key=lambda s: (s.startswith("\\"), s)))
    head = (
        f"{_DECL_HEAD}\n"
        "h_{\\mu\\nu}::Symmetric.\n"
        "h^{\\mu\\nu}::Symmetric.\n"
        "R_{\\mu\\nu}::Symmetric.\n"
        f"{{{ordered}}}::Depends(\\nabla{{#}})."
    )
    return head


_METRIC_VARY = (
    r"vary(ex, $g^{\alpha\beta} -> -h^{\alpha\beta}, "
    r"sg -> 1/2 sg g^{\mu\nu} h_{\mu\nu}, "
    r"R_{\alpha\beta} -> \nabla_{\lambda}{dGamma^{\lambda}_{\beta\alpha}} "
    r"- \nabla_{\beta}{dGamma^{\lambda}_{\lambda\alpha}}$);"
)
_METRIC_DGAMMA = (
    r"substitute(ex, $dGamma^{\lambda}_{\nu\sigma} -> 1/2 g^{\lambda\rho} "
    r"( \nabla_{\nu}{h_{\rho\sigma}} + \nabla_{\sigma}{h_{\rho\nu}} "
    r"- \nabla_{\rho}{h_{\nu\sigma}} )$);"
)


def assemble_metric_eom_script(matches: list[BlockMatch]) -> str:
    """Assemble one Cadabra script for the metric EOM of the matched blocks.
    The script varies the real action and residue-checks it against a candidate
    built from the same blocks; the kernel's residue_zero is the verdict."""
    decl = _metric_declarations(matches)
    integrand = _render_cadabra([_metric_integrand_monomial(m) for m in matches])
    target = _render_cadabra([mono for m in matches for mono in _metric_target_monomials(m)])

    lines: list[str] = [
        decl,
        "",
        f"ex := \\int{{ {integrand} }}{{x}};",
        _METRIC_VARY,
        _METRIC_DGAMMA,
        "distribute(ex);",
        "product_rule(ex);",
        _METRIC_SUBS,
        "canonicalise(ex);",
        r"integrate_by_parts(ex, $\nabla_{\nu}{h_{\rho\sigma}}$);",
        "product_rule(ex);",
        "distribute(ex);",
        _METRIC_SUBS,
        r"substitute(ex, $\nabla_{\mu}{sg} -> 0$);",
        r"integrate_by_parts(ex, $h_{\rho\sigma}$);",
        "product_rule(ex);",
        "distribute(ex);",
        _METRIC_SUBS,
        r"substitute(ex, $\nabla_{\mu}{sg} -> 0$);",
        r"substitute(ex, $\int{A??}{x} -> A??$);",
        r"substitute(ex, $h^{\alpha\beta} -> g^{\alpha\gamma} g^{\beta\chi} h_{\gamma\chi}$);",
        "distribute(ex);",
        "eliminate_kronecker(ex);",
        "sort_product(ex);",
        "canonicalise(ex);",
        "rename_dummies(ex);",
        'print("NOETHER_RESULT: " + str(ex))',
        "",
        f"target := {target};",
        "distribute(target);",
        r"substitute(target, $h^{\alpha\beta} -> g^{\alpha\gamma} g^{\beta\chi} h_{\gamma\chi}$);",
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


def _metric_display_monomials(match: BlockMatch) -> list[tuple[Fraction, str]]:
    c = match.coeff
    if match.block == EINSTEIN_HILBERT:
        return [(c, r"R_{\mu\nu}"), (-c / 2, r"g_{\mu\nu} R")]
    if match.block == NONMINIMAL:
        n = match.coupling
        return [
            (c, f"{n} R_{{\\mu\\nu}}"),
            (-c / 2, f"g_{{\\mu\\nu}} {n} R"),
            (c, f"g_{{\\mu\\nu}}\\Box {n}"),
            (-c, f"\\nabla_{{\\mu}}\\nabla_{{\\nu}} {n}"),
        ]
    if match.block == KINETIC:
        return [
            (c, r"\nabla_{\mu}\phi\,\nabla_{\nu}\phi"),
            (-c / 2, r"g_{\mu\nu}(\nabla\phi)^2"),
        ]
    if match.block == POTENTIAL:
        return [(-c / 2, f"g_{{\\mu\\nu}} {match.coupling}")]
    raise ValueError(match.block)


def compose_metric_display_tex(matches: list[BlockMatch]) -> str:
    """The composed metric EOM tensor E_{mu nu} = 0, for display."""
    monomials = [mono for m in matches for mono in _metric_display_monomials(m)]
    return f"{_render_tex(monomials)} = 0"
