"""Eval 3s (docs/04_EVALS.md, eval 3 stretch task): spectrum around Minkowski.

Theory (eval 3): S = \\int d^4x \\sqrt{-g} [ F(phi) R - 1/2 (d phi)^2 - V(phi) ],
perturbed around g = eta, phi = phi_0 constant with V(phi_0) = V'(phi_0) = 0,
to quadratic order in the action (linear order in the equations of motion).

Kernel-verified results (sympy, noether/kernels/sympy_kernel/linearized.py;
every coefficient pinned by computation, alternatives rejected):

  linear EOMs   (anchored to the cadabra-verified FULL eval-3 equations by
                 recomputing their eps-derivative on concrete fields):
     metric:  F0 G1[h] + (eta box - dd)(F1 chi) = 0
     scalar:  box chi + F1 R1[h] - V2 chi = 0
  shift         h_{mu nu} = hbar_{mu nu} - (F1/F0) chi eta_{mu nu}
                removes the kinetic mixing exactly (+F1/F0 does not):
                metric EOM becomes F0 G1[hbar] = 0.
  graviton      G1[hbar] = 0 is linearized vacuum GR: a TT plane wave solves
                it iff the momentum is null (massless); its trace forces
                R1[hbar] = 0 on shell.
  scalar        K_chi box chi - V2 chi = 0 with
                K_chi = (F0 + 3 F1^2) / F0
                (using R1[chi eta] = -3 box chi, kernel-pinned), so the
                scalar mass is m^2 = V''(phi_0) / K_chi.
  no ghost      F0 > 0 (graviton sector) and F0 + 3 F1^2 > 0 (scalar sector).

F0 = F(phi_0), F1 = F'(phi_0), V2 = V''(phi_0). The diagonalization is
performed at the level of the equations of motion and needs no gauge choice;
the TT gauge enters only the plane-wave counting statement.

The pytest gate lives in evals/test_eval3s.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Expr, Sym, add, down, num, prod, tensor

MU, NU = down("mu"), down("nu")


def diagonalizing_shift() -> Expr:
    """hbar_{mu nu} (the graviton variable after absorbing the mixing)."""
    return tensor(r"\bar{h}", MU, NU)


def graviton_eom() -> Expr:
    """G^{(1)}_{mu nu}[hbar]: linearized vacuum GR for the shifted variable."""
    return tensor(r"G^{(1)}", MU, NU)


def scalar_eom() -> Expr:
    """K_chi box chi - V''(phi_0) chi."""
    return add(
        prod(Sym(name=r"K_\chi"), Sym(name=r"\Box"), Sym(name=r"\chi")),
        prod(num(-1), Sym(name=r"V''(\phi_0)"), Sym(name=r"\chi")),
    )


AMBIGUITIES = [
    Ambiguity(
        id="amb-background",
        question="Perturb around which background?",
        kind="undecidable",
        options=["minkowski-phi0-constant", "custom-background"],
    ),
    Ambiguity(
        id="amb-stationarity",
        question=(
            "Minkowski with constant phi_0 solves the background equations "
            "only if V(phi_0) = V'(phi_0) = 0. Impose that on the potential?"
        ),
        kind="inferable",
        options=["impose-stationarity", "general-potential"],
    ),
    Ambiguity(
        id="amb-order",
        question="Expand to which order?",
        kind="inferable",
        options=["quadratic-action-linear-eom", "higher-order"],
    ),
    Ambiguity(
        id="amb-gauge",
        question=(
            "Gauge handling: diagonalize at the level of the equations of "
            "motion (no gauge needed) with TT only for mode counting, or fix "
            "a gauge in the action first?"
        ),
        kind="conventional",
        options=["eom-level-no-gauge", "tt-gauge-in-action"],
    ),
    Ambiguity(
        id="amb-conventions",
        question="Mostly-plus signature, noether-default-v1 curvature signs?",
        kind="conventional",
        options=["noether-default-v1", "custom"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-background": "minkowski-phi0-constant",
    "amb-stationarity": "impose-stationarity",
    "amb-order": "quadratic-action-linear-eom",
    "amb-gauge": "eom-level-no-gauge",
    "amb-conventions": "noether-default-v1",
}

_LAGRANGIAN_TEX = r"F(\phi) R - \tfrac{1}{2}\nabla_\mu\phi\nabla^\mu\phi - V(\phi)"


def build_npr(resolved: bool = True) -> NPR:
    # The action IS eval 3's action (same AST), only the task differs.
    from evals.eval3_scalar_tensor import build_npr as build_eval3_npr

    lagrangian = build_eval3_npr().action.lagrangian
    ambiguities = [a.model_copy(deep=True) for a in AMBIGUITIES]
    if resolved:
        for amb in ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(),
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(name="phi", kind="scalar-field", role="dynamical", rank=0),
            ObjectDecl(name="F", kind="function", role="shorthand", args=["phi"]),
            ObjectDecl(name="V", kind="function", role="shorthand", args=["phi"]),
            ObjectDecl(
                name=r"\bar{h}",
                kind="shorthand",
                role="shorthand",
                symmetry="symmetric",
                rank=2,
                definition_tex=(
                    r"\bar{h}_{\mu\nu} = h_{\mu\nu} "
                    r"+ \frac{F'(\phi_0)}{F(\phi_0)}\,\chi\,\eta_{\mu\nu}"
                ),
            ),
            ObjectDecl(
                name=r"\chi",
                kind="shorthand",
                role="shorthand",
                rank=0,
                definition_tex=r"\chi = \phi - \phi_0",
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian,
            lagrangian_tex=_LAGRANGIAN_TEX,
        ),
        task=Task(type="perturb", with_respect_to=["g", "phi"]),
        ambiguities=ambiguities,
    )
