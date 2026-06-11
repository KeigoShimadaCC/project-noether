"""Eval 2 (docs/04_EVALS.md): Palatini / first-order gravity.

Input action: S = \\int d^4x \\sqrt{-g} g^{mu nu} R_{mu nu}(Gamma)
with an INDEPENDENT connection Gamma (torsion allowed). Two equations:

  metric:     R_{(mu nu)}(Gamma) - 1/2 g_{mu nu} g^{ab} R_{ab}(Gamma) = 0
  connection: solved by Gamma^lam_{mu nu} = C^lam_{mu nu}(g) + delta^lam_nu A_mu
              (Levi-Civita plus an arbitrary projective mode), under which the
              metric equation reduces to G_{mu nu}(g) = 0.

The pytest entry points live in evals/test_eval2.py.
"""

from noether.npr import (
    NOETHER_DEFAULT_V1,
    NPR,
    Action,
    Ambiguity,
    ConnectionSpec,
    Geometry,
    ObjectDecl,
    Task,
)
from noether.npr.ast import Expr, add, down, num, prod, tensor, up

MU, NU = down("mu"), down("nu")


def target_metric_eom() -> Expr:
    """R_{(mu nu)} - 1/2 g_{mu nu} g^{ab} R_{ab}, Ricci of the independent
    connection (symmetrized explicitly because torsion breaks the symmetry)."""
    return add(
        prod(num(1, 2), tensor("R", MU, NU)),
        prod(num(1, 2), tensor("R", NU, MU)),
        prod(
            num(-1, 2),
            tensor("g", MU, NU),
            tensor("g", up("alpha"), up("beta")),
            tensor("R", down("alpha"), down("beta")),
        ),
    )


CONNECTION_SOLUTION_TEX = (
    r"\Gamma^{\lambda}_{\mu\nu} = \{^{\lambda}_{\mu\nu}\}_g + \delta^{\lambda}_{\nu} A_{\mu}"
)

AMBIGUITIES = [
    Ambiguity(
        id="amb-connection-independence",
        question=(
            "Is the connection in R_{mu nu} the Levi-Civita connection of g, "
            "or an independent field to vary separately (Palatini)?"
        ),
        kind="undecidable",
        options=["levi-civita", "independent"],
    ),
    Ambiguity(
        id="amb-torsion",
        question="Does the independent connection carry torsion?",
        kind="undecidable",
        options=["torsion-allowed", "torsion-free"],
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
        options=["g and Gamma", "g only"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-connection-independence": "independent",
    "amb-torsion": "torsion-allowed",
    "amb-conventions": "noether-default-v1",
    "amb-vary-wrt": "g and Gamma",
}


def build_npr(resolved: bool = True) -> NPR:
    lagrangian = prod(
        tensor("g", up("mu"), up("nu")),
        tensor("R", down("mu"), down("nu")),
    )
    ambiguities = [a.model_copy(deep=True) for a in AMBIGUITIES]
    if resolved:
        for amb in ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(
            connection=ConnectionSpec(type="independent", torsion=True, nonmetricity=True)
        ),
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(name="Gamma", kind="connection", role="dynamical", rank=3),
            ObjectDecl(
                name="R",
                kind="shorthand",
                role="shorthand",
                symmetry="none",  # torsionful Ricci is not symmetric
                rank=2,
                definition_tex=r"R_{\mu\nu}(\Gamma)",
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian,
            lagrangian_tex=r"g^{\mu\nu} R_{\mu\nu}(\Gamma)",
        ),
        task=Task(type="vary", with_respect_to=["g", "Gamma"]),
        ambiguities=ambiguities,
    )
