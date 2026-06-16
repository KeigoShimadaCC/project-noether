"""Eval 6 (docs/04_EVALS.md): cubic Galileon scalar sector.

Input action: S = \\int d^4x \\sqrt{-g} ( -1/2 (nabla phi)^2 - V(phi) + K(phi) box phi )

This is the Horndeski G3 term K(phi) box phi added to a canonical scalar. The
eval is scoped to the scalar field equation, the sector whose new mechanics
(a coupling times box phi, peeled by a two-pass integration by parts plus the
coupling chain rule) the audited template `eom_cubic_galileon_scalar` exercises.

Expected scalar EOM (noether-default-v1):
  (1 + 2 K'(phi)) box phi + K''(phi) (nabla phi)^2 - V'(phi) = 0.

The K(phi) box phi term contributes the braiding 2 K' box phi + K'' (nabla phi)^2:
integrating K box(delta phi) by parts twice lands box on K, and box K =
K'' (nabla phi)^2 + K' box phi through the chain rule. The pytest entry points
live in evals/test_eval6.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Expr, Sym, add, cov, down, num, prod, up
from noether.npr.parse import parse_lagrangian

PHI = Sym(name="phi")

LAGRANGIAN_TEX = r"- \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi) + K(\phi)\Box\phi"


def _box(e: Expr) -> Expr:
    return cov(up("alpha"), cov(down("alpha"), e))


def _kin_trace() -> Expr:
    """(nabla phi)^2 = nabla_a phi nabla^a phi."""
    return prod(cov(down("alpha"), PHI), cov(up("alpha"), PHI))


def target_scalar_eom() -> Expr:
    """(1 + 2 K') box phi + K'' (nabla phi)^2 - V'."""
    return add(
        _box(PHI),
        prod(num(2, 1), Sym(name="Kp"), _box(PHI)),
        prod(Sym(name="Kpp"), _kin_trace()),
        prod(num(-1, 1), Sym(name="Vp")),
    )


AMBIGUITIES = [
    Ambiguity(
        id="amb-coupling-K",
        question="Is K(phi) an arbitrary function, or a fixed constant?",
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
        id="amb-conventions",
        question=(
            "Dimension 4, mostly-plus signature, noether-default-v1 curvature signs, "
            "X = -1/2 (nabla phi)^2 kinetic normalization?"
        ),
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
            ObjectDecl(name="K", kind="function", role="coupling", args=["phi"]),
            ObjectDecl(name="V", kind="function", role="coupling", args=["phi"]),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=parse_lagrangian(LAGRANGIAN_TEX),
            lagrangian_tex=LAGRANGIAN_TEX,
        ),
        task=Task(type="vary", with_respect_to=["phi"]),
        ambiguities=ambiguities,
    )
