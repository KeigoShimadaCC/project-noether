r"""Eval 7 (docs/04_EVALS.md): k-essence and the general scalar Horndeski sector.

Input action: S = \int d^4x \sqrt{-g} ( K(phi, X) - V(phi) + G(phi) box phi )

This is the Horndeski G2 term K(phi, X) (k-essence, the X-dependent coupling
that the fidelity pass left unverified) added to a potential and the G3 term
G(phi) box phi. The eval is scoped to the scalar field equation and is derived
WITHOUT a per-theory template: the Lagrangian is decomposed into building
blocks (k-essence, potential, cubic Galileon), one Cadabra script is assembled
for the actual action, and the kernel's residue check verifies it.

X = -1/2 (nabla phi)^2 is expanded to its primitive inside the kernel and
collapsed back to the shorthand for display, so the result reads in clean
notation while the verification runs on primitives.

Expected scalar EOM (noether-default-v1):
  K_phi + K_X box phi + K_Xphi (nabla phi)^2
      - K_XX nabla^a phi nabla^b phi nabla_a nabla_b phi
      - V_phi + 2 G_phi box phi + G_phiphi (nabla phi)^2 = 0.

The pytest entry points live in evals/test_eval7.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Expr, Sym, add, cov, down, num, prod, up
from noether.npr.parse import parse_lagrangian

PHI = Sym(name="phi")

MEASURE_TEX = r"d^4x \sqrt{-g}"
LAGRANGIAN_TEX = r"K(\phi, X) - V(\phi) + G(\phi)\Box\phi"
X_DEF_TEX = r"-\tfrac12 \nabla_\mu \phi \nabla^\mu \phi"


def _box(e: Expr) -> Expr:
    return cov(up("alpha"), cov(down("alpha"), e))


def _kin_trace() -> Expr:
    """(nabla phi)^2 = nabla_a phi nabla^a phi."""
    return prod(cov(down("alpha"), PHI), cov(up("alpha"), PHI))


def _hess_trace() -> Expr:
    """nabla^a phi nabla^b phi nabla_a nabla_b phi."""
    return prod(
        cov(up("alpha"), PHI),
        cov(up("beta"), PHI),
        cov(down("alpha"), cov(down("beta"), PHI)),
    )


def target_scalar_eom() -> Expr:
    """The composed k-essence + potential + cubic Galileon scalar EOM."""
    return add(
        Sym(name="K_phi"),
        prod(Sym(name="K_X"), _box(PHI)),
        prod(Sym(name="K_Xphi"), _kin_trace()),
        prod(num(-1, 1), Sym(name="K_XX"), _hess_trace()),
        prod(num(-1, 1), Sym(name="V_phi")),
        prod(num(2, 1), Sym(name="G_phi"), _box(PHI)),
        prod(Sym(name="G_phiphi"), _kin_trace()),
    )


AMBIGUITIES = [
    Ambiguity(
        id="amb-coupling-K",
        question="Is K(phi, X) an arbitrary function, or a fixed constant?",
        kind="undecidable",
        options=["arbitrary-function", "constant"],
    ),
    Ambiguity(
        id="amb-coupling-V",
        question="Is V(phi) an arbitrary function, or a fixed constant?",
        kind="undecidable",
        options=["arbitrary-function", "constant"],
    ),
    Ambiguity(
        id="amb-coupling-G",
        question="Is G(phi) an arbitrary function, or a fixed constant?",
        kind="undecidable",
        options=["arbitrary-function", "constant"],
    ),
    Ambiguity(
        id="amb-kinetic-X",
        question=(
            "Is X the canonical kinetic shorthand -1/2 (nabla phi)^2, or an independent field?"
        ),
        kind="conventional",
        options=["kinetic-scalar", "independent-field"],
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
        options=["phi only", "g and phi"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-coupling-K": "arbitrary-function",
    "amb-coupling-V": "arbitrary-function",
    "amb-coupling-G": "arbitrary-function",
    "amb-kinetic-X": "kinetic-scalar",
    "amb-conventions": "noether-default-v1",
    "amb-vary-wrt": "phi only",
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
            ObjectDecl(name="phi", kind="scalar-field", role="dynamical"),
            ObjectDecl(
                name="X", kind="shorthand", role="shorthand", rank=0, definition_tex=X_DEF_TEX
            ),
            ObjectDecl(name="K", kind="function", role="coupling", args=["phi", "X"]),
            ObjectDecl(name="V", kind="function", role="coupling", args=["phi"]),
            ObjectDecl(name="G", kind="function", role="coupling", args=["phi"]),
        ],
        action=Action(
            measure_tex=MEASURE_TEX,
            lagrangian=parse_lagrangian(LAGRANGIAN_TEX),
            lagrangian_tex=LAGRANGIAN_TEX,
        ),
        task=Task(type="vary", with_respect_to=["phi"]),
        ambiguities=ambiguities,
    )
