"""Eval 4 (docs/04_EVALS.md): Maxwell on a fixed curved background.

Input action: S = -1/4 \\int d^4x \\sqrt{-g} F_{mu nu} F^{mu nu}, F = dA.
The metric is a BACKGROUND: only A_mu is dynamical. Expected EOM:

  nabla_mu F^{mu nu} = 0

plus the Noether identity nabla_nu nabla_mu F^{mu nu} == 0 (holds for any
antisymmetric F, off shell), which is what gauge invariance forces.

The pytest entry points live in evals/test_eval4.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Expr, cov, down, num, prod, tensor, up

NU = up("nu")


def target_eom() -> Expr:
    """nabla_mu F^{mu nu}."""
    return cov(down("mu"), tensor("F", up("mu"), NU))


def noether_identity_expr() -> Expr:
    """nabla_nu nabla_mu F^{mu nu}: identically zero for antisymmetric F."""
    return cov(down("nu"), cov(down("mu"), tensor("F", up("mu"), up("nu"))))


def lagrangian_expr() -> Expr:
    return prod(
        num(-1, 4),
        tensor("F", down("mu"), down("nu")),
        tensor("F", up("mu"), up("nu")),
    )


AMBIGUITIES = [
    Ambiguity(
        id="amb-metric-role",
        question="Is the metric dynamical (vary it too) or a fixed background?",
        kind="undecidable",
        options=["background", "dynamical"],
    ),
    Ambiguity(
        id="amb-f-composite",
        question="Is F_{mu nu} the field strength dA of a gauge potential A_mu?",
        kind="inferable",
        options=["field-strength-of-A", "independent-field"],
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
    "amb-f-composite": "field-strength-of-A",
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
            ObjectDecl(name="A", kind="tensor-field", role="dynamical", rank=1),
            ObjectDecl(
                name="F",
                kind="shorthand",
                role="shorthand",
                symmetry="antisymmetric",
                rank=2,
                definition_tex=r"\nabla_\mu A_\nu - \nabla_\nu A_\mu",
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian_expr(),
            lagrangian_tex=r"-\tfrac14 F_{\mu\nu} F^{\mu\nu}",
        ),
        task=Task(type="vary", with_respect_to=["A"]),
        ambiguities=ambiguities,
    )
