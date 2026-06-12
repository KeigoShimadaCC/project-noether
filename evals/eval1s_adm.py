"""Eval 1s (docs/04_EVALS.md, eval 1 stretch task): ADM decomposition of GR.

Input action: S = \\int d^4x \\sqrt{-g} R, decomposed with respect to the
foliation by t = const spacelike slices (lapse N, shift N^i, spatial metric
h_ij, future-pointing unit normal n_mu = (-N, 0, 0, 0)).

Kernel-verified results (sympy components, 1+2 nondegenerate background;
every sign below was fixed by computation, see kernels/sympy_kernel/adm.py):

  split      sqrt(-g) R = N sqrt(h) ( R^{(3)} + K_{ab}K^{ab} - K^2 )
                          - 2 d_mu( sqrt(-g) v^mu ),
             v^mu = n^nu nabla_nu n^mu - n^mu nabla_nu n^nu
  K          K_ij = (d_t h_ij - D_i N_j - D_j N_i)/(2N) = +nabla_i n_j
  Hamiltonian constraint   R^{(3)} + K^2 - K_{ab}K^{ab} = 0   (vacuum),
             equal to 2 G_{mu nu} n^mu n^nu and to the lapse Euler-Lagrange
             equation of the bulk (N appears undifferentiated there).
  momentum constraint      D_a ( K^a{}_b - h^a{}_b K ) = 0    (vacuum),
             equal to G_{mu b} n^mu.

The constraints are the normal projections of the Einstein equations: first
order in time derivatives (they contain h and K only), so they constrain
initial data rather than evolve it. The evolution equations are the remaining
spatial-spatial projections. D is the Levi-Civita derivative of h on the
slice; indices a, b are spatial.

The pytest gate lives in evals/test_eval1s.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Deriv, Expr, Pow, add, down, num, prod, tensor, up

B = down("b")


def bulk_density() -> Expr:
    """R^{(3)} + K_{ab}K^{ab} - K^2: the bulk Lagrangian over N sqrt(h)."""
    return add(
        tensor("R^{(3)}"),
        prod(tensor("K", down("a"), down("b")), tensor("K", up("a"), up("b"))),
        prod(num(-1), Pow(base=tensor("K"), exp=2)),
    )


def hamiltonian_constraint() -> Expr:
    """R^{(3)} + K^2 - K_{ab}K^{ab} (vacuum; equals 2 G_{mu nu} n^mu n^nu)."""
    return add(
        tensor("R^{(3)}"),
        Pow(base=tensor("K"), exp=2),
        prod(num(-1), tensor("K", down("a"), down("b")), tensor("K", up("a"), up("b"))),
    )


def momentum_constraint() -> Expr:
    """D_a (K^a_b - h^a_b K) (vacuum; equals G_{mu b} n^mu). D is spatial."""
    return Deriv(
        op="covariant",
        index=down("a"),
        expr=add(
            tensor("K", up("a"), B),
            prod(num(-1), tensor("h", up("a"), B), tensor("K")),
        ),
    )


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
        geometry=Geometry(),
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(
                name="N",
                kind="shorthand",
                role="shorthand",
                rank=0,
                definition_tex=r"\text{lapse of the } t \text{ foliation}",
            ),
            ObjectDecl(
                name="N^i",
                kind="shorthand",
                role="shorthand",
                rank=1,
                definition_tex=r"\text{shift of the } t \text{ foliation}",
            ),
            ObjectDecl(
                name="h",
                kind="shorthand",
                role="shorthand",
                symmetry="symmetric",
                rank=2,
                definition_tex=r"h_{ij} = g_{ij} + n_i n_j \text{ (induced spatial metric)}",
            ),
            ObjectDecl(
                name="K",
                kind="shorthand",
                role="shorthand",
                symmetry="symmetric",
                rank=2,
                definition_tex=(
                    r"K_{ij} = \frac{1}{2N}(\partial_t h_{ij} - D_i N_j - D_j N_i) "
                    r"= \nabla_i n_j"
                ),
            ),
            ObjectDecl(
                name="R^{(3)}",
                kind="shorthand",
                role="shorthand",
                rank=0,
                definition_tex=r"\text{Ricci scalar of } h_{ij} \text{ on the slice}",
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
