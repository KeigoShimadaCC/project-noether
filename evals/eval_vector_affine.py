"""Vector (Maxwell) EOM on a metric-affine background.

Verifies VAL-EOM-020 and VAL-EOM-021.

Input action: S = -1/4 int d^4x sqrt(-g) F_{mu nu} F^{mu nu}

Two field-strength definitions are considered:

1. F = dA (exterior derivative): the EOM is nabla^{LC}_mu F^{mu nu} = 0,
   which when expressed with the full-connection nabla^{aff} carries T/Q
   correction terms.  The hypermomentum is zero (no Gamma dependence).

2. F = nabla A (covariant curl): the EOM naturally involves the full-
   connection divergence with a torsion term,
   (1/sqrt(-g)) partial_mu(sqrt(-g) F^{mu nu}) + (1/2) T^nu_{mu rho}
   F^{mu rho} = 0.  The hypermomentum is nonzero:
   Delta^lambda_{mu nu} = -2 A_lambda F^{mu nu} (antisymmetric in mu, nu).

The two derivations differ exactly in the connection-equation source,
each verified or gated.
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
from noether.npr.ast import Expr, add, cov, down, num, prod, tensor, up

MU, NU = down("mu"), up("nu")


def target_eom_dA() -> Expr:
    """nabla_mu F^{mu nu} = 0 (LC divergence, equivalent to the
    full-connection divergence plus T/Q correction terms)."""
    return cov(down("mu"), tensor("F", up("mu"), NU))


def target_eom_covcurl() -> Expr:
    """Covariant-curl EOM: (1/sqrt(-g)) partial_mu(sqrt(-g) F^{mu nu})
    + (1/2) T^nu_{mu rho} F^{mu rho} = 0."""
    return add(
        cov(down("mu"), tensor("F", up("mu"), NU)),
        prod(
            num(1, 2),
            tensor("T", NU, down("mu"), down("rho")),
            tensor("F", up("mu"), up("rho")),
        ),
    )


def hypermomentum_covcurl() -> Expr:
    """Delta^lambda_{mu nu} = -2 A_lambda F^{mu nu} (antisymmetric in
    mu, nu).  This is the spin-type hypermomentum from the covariant-curl
    choice."""
    return prod(num(-2), tensor("A", down("lambda")), tensor("F", up("mu"), up("nu")))


AMBIGUITIES = [
    Ambiguity(
        id="amb-connection-independence",
        question=(
            "Is the connection the Levi-Civita connection of g, "
            "or an independent field to vary separately (metric-affine)?"
        ),
        kind="undecidable",
        options=["levi-civita", "independent"],
    ),
    Ambiguity(
        id="amb-field-strength-definition",
        question=(
            "Is the field strength F_{mu nu} defined as the exterior "
            "derivative dA or as the covariant curl nabla A "
            "(which differs by a torsion term)?"
        ),
        kind="undecidable",
        options=["exterior-derivative", "covariant-curl"],
    ),
    Ambiguity(
        id="amb-conventions",
        question="Dimension 4, mostly-plus signature, noether-default-v1 conventions?",
        kind="conventional",
        options=["noether-default-v1", "custom"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-connection-independence": "independent",
    "amb-field-strength-definition": "exterior-derivative",
    "amb-conventions": "noether-default-v1",
}

ELICITATION_ANSWERS_COVCURL = {
    "amb-connection-independence": "independent",
    "amb-field-strength-definition": "covariant-curl",
    "amb-conventions": "noether-default-v1",
}


def build_npr(field_strength: str = "exterior-derivative") -> NPR:
    """Build an NPR for the Maxwell action on a metric-affine background.

    field_strength: 'exterior-derivative' for F=dA or 'covariant-curl'
    for F=nabla A.
    """
    answers = (
        ELICITATION_ANSWERS
        if field_strength == "exterior-derivative"
        else ELICITATION_ANSWERS_COVCURL
    )
    ambiguities = [a.model_copy(deep=True) for a in AMBIGUITIES]
    for amb in ambiguities:
        amb.resolution = answers[amb.id]

    f_def_tex = (
        r"2\partial_{[\mu}A_{\nu]}"
        if field_strength == "exterior-derivative"
        else r"2\nabla_{[\mu}A_{\nu]}"
    )

    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(
            connection=ConnectionSpec(type="independent", torsion=True, nonmetricity=True)
        ),
        objects=[
            ObjectDecl(
                name="g",
                kind="metric",
                role="background",
                symmetry="symmetric",
                rank=2,
            ),
            ObjectDecl(
                name="Gamma",
                kind="connection",
                role="dynamical",
                rank=3,
            ),
            ObjectDecl(
                name="A",
                kind="tensor-field",
                role="dynamical",
                rank=1,
            ),
            ObjectDecl(
                name="F",
                kind="shorthand",
                role="shorthand",
                symmetry="antisymmetric",
                rank=2,
                definition_tex=f_def_tex,
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=prod(
                num(-1, 4),
                tensor("F", down("mu"), down("nu")),
                tensor("F", up("mu"), up("nu")),
            ),
            lagrangian_tex=r"-\tfrac14 F_{\mu\nu} F^{\mu\nu}",
        ),
        task=Task(type="vary", with_respect_to=["A", "Gamma"]),
        ambiguities=ambiguities,
    )
