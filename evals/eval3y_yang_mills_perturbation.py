"""Eval 3y: quadratic-action expansion of a Yang-Mills field (non-abelian gauge).

Input action (noether-default-v1, fixed background metric):
  S = -1/4 \\int d^4x \\sqrt{-g} F^a_{mu nu} F^{a mu nu},
  F^a_{mu nu} = nabla_mu A^a_nu - nabla_nu A^a_mu + g f^{abc} A^b_mu A^c_nu.

Split the potential into background plus fluctuation, A -> Abar + v. To second
order in v the fluctuation action is

  S2 = -1/4 sqrt(-g) ( f1^a_{mu nu} f1^{a mu nu}
                       + 2 g f^{abc} Fbar^a_{mu nu} v^b_mu v^c_nu ),

with the background-covariant linearized strength
  f1^a_{mu nu} = Dbar_mu v^a_nu - Dbar_nu v^a_mu,
  Dbar_mu v^a_nu = nabla_mu v^a_nu + g f^{abc} Abar^b_mu v^c_nu.

Its equation of motion is the non-abelian generalization of the photon
operator, now with the gluon self-coupling to the background:

  Dbar_mu f1^{a mu nu} + g f^{abc} v^b_mu Fbar^{c mu nu} = 0.

It is verified two independent ways inside the kernel:
  - delta S2 / delta v matches the documented linearized YM operator
    (residue_zero);
  - linearizing the full nonlinear YM equation D_mu F^{a mu nu} = 0 reproduces
    it (linearized_eom_match).

The Cadabra scaffold is the frozen template `pert_yang_mills_quadratic`; the
pytest gate lives in evals/test_eval3y.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Expr, down, num, prod, tensor, up

CONVENTIONS = "noether-default-v1"

# Documented quadratic action and linearized EOM (for the human audit; the
# kernel derives and checks these, the strings are not used as input).
QUADRATIC_ACTION_TEX = (
    r"-\tfrac14 \int d^4x\,\sqrt{-g}\left("
    r"f^a_{\mu\nu} f^{a\,\mu\nu} + 2 g f^{abc} \bar F^a_{\mu\nu} v^b{}_\mu v^c{}_\nu\right)"
)
LINEARIZED_EOM_TEX = r"\bar D_\mu f^{a\,\mu\nu} + g f^{abc} v^b{}_\mu \bar F^{c\,\mu\nu} = 0"
TEMPLATE = "pert_yang_mills_quadratic"

GAUGE_GROUP = "SU(N)"


def lagrangian_expr() -> Expr:
    return prod(
        num(-1, 4),
        tensor("F", down("a"), down("mu"), down("nu")),
        tensor("F", up("a"), up("mu"), up("nu")),
    )


AMBIGUITIES = [
    Ambiguity(
        id="amb-metric-role",
        question="Is the metric dynamical (vary it too) or a fixed background?",
        kind="undecidable",
        options=["background", "dynamical"],
    ),
    Ambiguity(
        id="amb-gauge-group",
        question="Is the gauge group abelian U(1) (Maxwell) or non-abelian (Yang-Mills)?",
        kind="undecidable",
        options=["U(1)", "SU(N)"],
    ),
    Ambiguity(
        id="amb-conventions",
        question="Dimension 4, mostly-plus signature, noether-default-v1 conventions?",
        kind="conventional",
        options=["noether-default-v1", "custom"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-metric-role": "background",
    "amb-gauge-group": "SU(N)",
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
            ObjectDecl(name="g", kind="metric", role="background", symmetry="symmetric", rank=2),
            ObjectDecl(
                name="A",
                kind="tensor-field",
                role="dynamical",
                rank=1,
                gauge_group=GAUGE_GROUP,
            ),
            ObjectDecl(
                name="F",
                kind="shorthand",
                role="shorthand",
                symmetry="antisymmetric",
                rank=2,
                definition_tex=(
                    r"\nabla_\mu A^a_\nu - \nabla_\nu A^a_\mu + g f^{abc} A^b_\mu A^c_\nu"
                ),
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian_expr(),
            lagrangian_tex=r"-\tfrac14 F^a_{\mu\nu} F^{a\,\mu\nu}",
        ),
        task=Task(type="perturb", with_respect_to=["A"]),
        ambiguities=ambiguities,
    )
