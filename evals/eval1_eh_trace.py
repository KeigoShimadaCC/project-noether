"""Eval 1 (docs/04_EVALS.md): Einstein-Hilbert in trace form.

Input action: S = \\int d^4x \\sqrt{-g} g^{mu nu} G_{mu nu}
Expected EOM:  G_{mu nu} = 0

This module builds the eval's NPR (with the elicitation the doc specifies),
the target result expression, and the verification checks. The pytest entry
points live in evals/test_eval1.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Expr, add, down, num, prod, tensor, up

# --- the target result: G_{mu nu} = 0, i.e. the LHS expression G_{mu nu} ----

MU, NU = down("mu"), down("nu")


def target_eom() -> Expr:
    """G_{mu nu}, the canonical good-form LHS of the vacuum equation."""
    return tensor("G", MU, NU)


def target_eom_expanded() -> Expr:
    """R_{mu nu} - 1/2 g_{mu nu} g^{ab} R_{ab}: must equal target on any background."""
    return add(
        tensor("R", MU, NU),
        prod(
            num(-1, 2),
            tensor("g", MU, NU),
            tensor("R", down("alpha"), down("beta")),
            tensor("g", up("alpha"), up("beta")),
        ),
    )


# --- the eval's NPR ----------------------------------------------------------

AMBIGUITIES = [
    Ambiguity(
        id="amb-g-composite",
        question=(
            "Is G_{mu nu} the Einstein tensor built from g (Levi-Civita), "
            "or an independent tensor field?"
        ),
        kind="undecidable",
        options=["einstein-tensor-of-g", "independent-field"],
    ),
    Ambiguity(
        id="amb-conventions",
        question="Dimension 4, mostly-plus signature, noether-default-v1 curvature signs?",
        kind="conventional",
        options=["noether-default-v1", "custom"],
    ),
    Ambiguity(
        id="amb-vary-wrt",
        question="Vary with respect to which field(s)?",
        kind="undecidable",
        options=["g"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-g-composite": "einstein-tensor-of-g",
    "amb-conventions": "noether-default-v1",
    "amb-vary-wrt": "g",
}


def build_npr(resolved: bool = True) -> NPR:
    """The eval-1 problem. With resolved=False, the ambiguity ledger is open
    and planning must be blocked (the structural no-guessing guarantee)."""
    lagrangian = prod(
        tensor("g", up("mu"), up("nu")),
        tensor("G", down("mu"), down("nu")),
    )
    ambiguities = [a.model_copy(deep=True) for a in AMBIGUITIES]
    if resolved:
        for amb in ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(),  # metric g, Levi-Civita connection
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(
                name="G",
                kind="shorthand",
                role="shorthand",
                symmetry="symmetric",
                rank=2,
                definition_tex=r"R_{\mu\nu} - \tfrac12 g_{\mu\nu} R",
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian,
            lagrangian_tex=r"g^{\mu\nu} G_{\mu\nu}",
        ),
        task=Task(type="vary", with_respect_to=["g"]),
        ambiguities=ambiguities,
    )
