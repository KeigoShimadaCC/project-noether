"""Eval 5 (docs/04_EVALS.md): Gauss-Bonnet, dimension-dependent identities.

Input action: S = \\int d^Dx \\sqrt{-g} ( R^2 - 4 R_{mu nu}R^{mu nu}
                                          + R_{mu nu rho sigma}R^{mu nu rho sigma} )

Expected EOM in general D (the Lanczos tensor):

  H_{mu nu} = 2( R R_{mu nu} - 2 R_{mu a}R^a_nu - 2 R^{ab} R_{mu a nu b}
                 + R_mu^{abc} R_{nu abc} ) - 1/2 g_{mu nu} GB = 0

with GB the Gauss-Bonnet scalar. In D = 4, H_{mu nu} == 0 identically (the
Gauss-Bonnet term is topological): Noether must report "identically zero",
not a dynamical 0 = 0.

Verification scope (honest accounting, per AGENTS.md rule 1):
  - cadabra verifies the Lovelock p=2 ALGEBRA in symbolic D: the generalized
    Kronecker delta contraction equals the GB scalar, and the Lovelock
    field-equation contraction (Lovelock 1971, J. Math. Phys. 12 498; the
    citable standard form) equals the Lanczos expression above.
  - sympy verifies the Lovelock PROPERTIES by component evaluation: H is
    symmetric and divergence-free in D=5; H == 0 identically on curved D=4
    backgrounds (including one with GB scalar nonzero, so the cancellation is
    real); H != 0 on a curved D=5 background (falsifier).
  - The variational derivation delta S -> H with Bianchi-identity reduction
    to second-order good form is the Horizon 2 identity-reduction capability
    and is NOT yet kernel-derived here.

The pytest entry points live in evals/test_eval5.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Expr, Pow, add, down, num, prod, tensor, up

MU, NU = down("mu"), down("nu")


def gb_scalar_expr() -> Expr:
    """R^2 - 4 R_{ab}R^{ab} + R_{abcd}R^{abcd} (explicit contractions)."""
    return add(
        Pow(base=tensor("R"), exp=2),
        prod(
            num(-4),
            tensor("R", down("alpha"), down("beta")),
            tensor("R", up("alpha"), up("beta")),
        ),
        prod(
            tensor("R", down("alpha"), down("beta"), down("gamma"), down("delta")),
            tensor("R", up("alpha"), up("beta"), up("gamma"), up("delta")),
        ),
    )


def target_eom() -> Expr:
    """The Lanczos tensor H_{mu nu} in explicit literature form."""
    return add(
        prod(num(2), tensor("R"), tensor("R", MU, NU)),
        prod(num(-4), tensor("R", MU, down("alpha")), tensor("R", up("alpha"), NU)),
        prod(
            num(-4),
            tensor("R", up("alpha"), up("beta")),
            tensor("R", MU, down("alpha"), NU, down("beta")),
        ),
        prod(
            num(2),
            tensor("R", MU, up("alpha"), up("beta"), up("gamma")),
            tensor("R", NU, down("alpha"), down("beta"), down("gamma")),
        ),
        prod(num(-1, 2), tensor("g", MU, NU), gb_scalar_expr()),
    )


def lanczos_shorthand() -> Expr:
    """H_{mu nu} as the component-evaluator shorthand (geometry-computed)."""
    return tensor("H", MU, NU)


AMBIGUITIES = [
    Ambiguity(
        id="amb-dimension",
        question=(
            "The measure is d^Dx: keep the dimension symbolic (general D), "
            "or fix D = 4 (where the Gauss-Bonnet term is topological)?"
        ),
        kind="undecidable",
        options=["symbolic-D", "fixed-4"],
    ),
    Ambiguity(
        id="amb-gb-recognition",
        question=(
            "The Lagrangian is the Gauss-Bonnet combination (Lovelock p=2): "
            "treat it as such, or as a generic quadratic-curvature theory?"
        ),
        kind="inferable",
        options=["gauss-bonnet-combination", "generic-quadratic"],
    ),
    Ambiguity(
        id="amb-conventions",
        question="Mostly-plus signature, noether-default-v1 curvature signs?",
        kind="conventional",
        options=["noether-default-v1", "custom"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-dimension": "symbolic-D",
    "amb-gb-recognition": "gauss-bonnet-combination",
    "amb-conventions": "noether-default-v1",
}


def build_npr(resolved: bool = True) -> NPR:
    ambiguities = [a.model_copy(deep=True) for a in AMBIGUITIES]
    if resolved:
        for amb in ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(),
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(
                name="GB",
                kind="shorthand",
                role="shorthand",
                rank=0,
                definition_tex=(
                    r"R^2 - 4 R_{\mu\nu}R^{\mu\nu} "
                    r"+ R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}"
                ),
            ),
        ],
        action=Action(
            measure_tex=r"d^Dx \sqrt{-g}",
            lagrangian=gb_scalar_expr(),
            lagrangian_tex=(
                r"R^2 - 4 R_{\mu\nu}R^{\mu\nu} + R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}"
            ),
        ),
        task=Task(type="vary", with_respect_to=["g"]),
        ambiguities=ambiguities,
    )
