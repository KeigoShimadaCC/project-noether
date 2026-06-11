"""Eval 3 (docs/04_EVALS.md): scalar-tensor gravity.

Input action: S = \\int d^4x \\sqrt{-g} ( F(phi) R - 1/2 (nabla phi)^2 - V(phi) )

Expected EOMs (noether-default-v1):
  metric: F G_{mu nu} + g_{mu nu} box F - nabla_mu nabla_nu F
          - 1/2 nabla_mu phi nabla_nu phi + 1/4 g_{mu nu} (nabla phi)^2
          + 1/2 g_{mu nu} V = 0
  scalar: box phi + F'(phi) R - V'(phi) = 0

The pytest entry points live in evals/test_eval3.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.ast import Expr, Index, Sym, add, cov, down, num, prod, tensor, up

MU, NU = down("mu"), down("nu")

PHI = Sym(name="phi")


def _dphi(ix: Index) -> Expr:
    return cov(ix, PHI)


def _box(e: Expr) -> Expr:
    return cov(up("alpha"), cov(down("alpha"), e))


def _kin_trace() -> Expr:
    """(nabla phi)^2 = nabla_a phi nabla^a phi."""
    return prod(_dphi(down("alpha")), _dphi(up("alpha")))


def target_metric_eom() -> Expr:
    """F G_{mu nu} + g_{mu nu} box F - nabla_mu nabla_nu F
    - 1/2 d_mu phi d_nu phi + 1/4 g_{mu nu} (d phi)^2 + 1/2 g_{mu nu} V."""
    F, V = Sym(name="F"), Sym(name="V")
    return add(
        prod(F, tensor("G", MU, NU)),
        prod(tensor("g", MU, NU), _box(F)),
        prod(num(-1, 1), cov(MU, cov(NU, F))),
        prod(num(-1, 2), _dphi(MU), _dphi(NU)),
        prod(num(1, 4), tensor("g", MU, NU), _kin_trace()),
        prod(num(1, 2), tensor("g", MU, NU), V),
    )


def target_scalar_eom() -> Expr:
    """box phi + F'(phi) R - V'(phi)."""
    return add(
        _box(PHI),
        prod(Sym(name="Fp"), tensor("R")),
        prod(num(-1, 1), Sym(name="Vp")),
    )


def minimal_limit_metric_eom() -> Expr:
    """The F = 1, V = 0 limit: G_{mu nu} - 1/2 d_mu phi d_nu phi
    + 1/4 g_{mu nu} (d phi)^2. Component-evaluable (no unknown functions)."""
    return add(
        tensor("G", MU, NU),
        prod(num(-1, 2), _dphi(MU), _dphi(NU)),
        prod(num(1, 4), tensor("g", MU, NU), _kin_trace()),
    )


def bianchi_link_lhs() -> Expr:
    """nabla^mu E_{mu nu} for the minimal-limit metric EOM."""
    return cov(up("mu"), minimal_limit_metric_eom())


def bianchi_link_rhs() -> Expr:
    """-1/2 box phi nabla_nu phi: what the generalized Bianchi identity forces
    the divergence to equal (so it vanishes exactly on the scalar shell)."""
    return prod(num(-1, 2), _box(PHI), _dphi(NU))


AMBIGUITIES = [
    Ambiguity(
        id="amb-coupling",
        question="Is F(phi) an arbitrary function, or a fixed constant (minimal coupling)?",
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
        options=["g and phi", "g only", "phi only"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-coupling": "arbitrary-function",
    "amb-conventions": "noether-default-v1",
    "amb-vary-wrt": "g and phi",
}


def build_npr(resolved: bool = True) -> NPR:
    lagrangian = add(
        prod(Sym(name="F"), tensor("R")),
        prod(num(-1, 2), _kin_trace()),
        prod(num(-1, 1), Sym(name="V")),
    )
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
            ObjectDecl(name="F", kind="function", role="coupling", args=["phi"]),
            ObjectDecl(name="V", kind="function", role="coupling", args=["phi"]),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian,
            lagrangian_tex=(r"F(\phi) R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)"),
        ),
        task=Task(type="vary", with_respect_to=["g", "phi"]),
        ambiguities=ambiguities,
    )
