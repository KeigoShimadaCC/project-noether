"""Eval: metric-affine ADM (3+1) decomposition (VAL-ADM-001 through VAL-ADM-005).

Input action: S = \\int d^4x \\sqrt{-g} R(Gamma), decomposed with respect to
the foliation by t = const spacelike slices with an independent affine
connection carrying torsion T and non-metricity Q.

Kernel-verified results (SymPy components, 1+2 nondegenerate background with
general affine connection; every sign and factor was fixed by computation):

  Metric sector (same as GR ADM, verified by adm-gr-1p2):
    split      sqrt(-g) R = N sqrt(h) ( R^{(3)} + K_{ab}K^{ab} - K^2 )
                           - 2 d_mu( sqrt(-g) v^mu )
    K          K_ij = +nabla_i n_j (expansion-positive convention)
    Hamiltonian constraint   R^{(3)} + K^2 - K_{ab}K^{ab}
    momentum constraint      D_a ( K^a_b - h^a_b K )

  Connection sector (metric-affine, verified by adm-affine-1p2):
    decomposition  Gamma = LC(g) + K(T) + L(Q), projected into
                   normal (n) and tangential (h) parts
    torsion         T^i_{jk} (spatial), T^n_{jk} (normal-upper),
                    T^i_{nk} (mixed)
    non-metricity   Q_{ijk} (spatial), Q_{nij} (normal-first),
                    Q_{inj} (mixed)
    constraints     Primary: algebraic connection EOM constrains Gamma
                    without time derivatives; Secondary: gated when
                    Dirac chain cannot be closed (Q != 0)

Convention: noether-default-v1 + metric-affine-v1.
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
from noether.npr.ast import tensor

AMBIGUITIES = [
    Ambiguity(
        id="amb-foliation",
        question=(
            "Decompose with respect to which foliation? The stated request is "
            "t = const spacelike slices with future-pointing unit normal."
        ),
        kind="undecidable",
        options=["t-constant-spacelike", "custom-foliation"],
    ),
    Ambiguity(
        id="amb-k-sign",
        question=(
            "Extrinsic curvature sign convention: K_ij = +nabla_i n_j "
            "(expansion positive) or K_ij = -nabla_i n_j?"
        ),
        kind="conventional",
        options=["K=+nabla-n", "K=-nabla-n"],
    ),
    Ambiguity(
        id="amb-boundary",
        question=(
            "The split produces a total-derivative term: report it explicitly "
            "or discard it (Gibbons-Hawking-York absorbed; constraints and "
            "evolution equations are unaffected either way)?"
        ),
        kind="undecidable",
        options=["keep-boundary-term", "discard-total-derivative"],
    ),
    Ambiguity(
        id="amb-conventions",
        question="Mostly-plus signature, noether-default-v1 curvature signs?",
        kind="conventional",
        options=["noether-default-v1", "custom"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-foliation": "t-constant-spacelike",
    "amb-k-sign": "K=+nabla-n",
    "amb-boundary": "keep-boundary-term",
    "amb-conventions": "noether-default-v1",
}


def build_npr(resolved: bool = True) -> NPR:
    ambiguities = [a.model_copy(deep=True) for a in AMBIGUITIES]
    if resolved:
        for amb in ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(
            connection=ConnectionSpec(
                type="independent",
                torsion=True,
                nonmetricity=True,
                metric_compatible=False,
                family="metric-affine",
            ),
        ),
        objects=[
            ObjectDecl(
                name="g",
                kind="metric",
                role="dynamical",
                symmetry="symmetric",
                rank=2,
            ),
            ObjectDecl(
                name="Gamma",
                kind="connection",
                role="dynamical",
                rank=3,
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=tensor("R"),
            lagrangian_tex="R",
        ),
        task=Task(type="adm", with_respect_to=["g"]),
        ambiguities=ambiguities,
    )
