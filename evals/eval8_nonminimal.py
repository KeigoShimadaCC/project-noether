r"""Eval 8 (docs/04_EVALS.md): nonminimal scalar-tensor gravity by composition.

Input action: S = \int d^4x \sqrt{-g} ( F(phi) R - 1/2 (nabla phi)^2 - V(phi) )

This is the same theory as eval 3, but derived WITHOUT a per-theory template.
The Lagrangian decomposes into curvature-coupled building blocks (nonminimal
F(phi) R, canonical kinetic, potential), and the same machinery yields BOTH
equations of motion compositionally:

  - scalar EOM (vary phi):  F_phi R + box phi - V_phi = 0
  - metric EOM (vary g):    F R_{mu nu} - 1/2 g_{mu nu} F R
                              + g_{mu nu} box F - nabla_mu nabla_nu F
                              - 1/2 nabla_mu phi nabla_nu phi
                              + 1/4 g_{mu nu} (nabla phi)^2
                              + 1/2 g_{mu nu} V = 0

The metric block contributions (Einstein-Hilbert, nonminimal, kinetic stress,
potential stress) are assembled into one Cadabra script that varies the real
action and residue-checks it against a candidate built from the same blocks.
The kernel's residue check is the verdict; no result is asserted by the model.

The pytest entry points live in evals/test_eval8.py.
"""

from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Ambiguity, Geometry, ObjectDecl, Task
from noether.npr.parse import parse_lagrangian

MEASURE_TEX = r"d^4x \sqrt{-g}"
LAGRANGIAN_TEX = r"F(\phi) R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)"

SCALAR_EOM_TEX = r"F_{\phi} R + \Box\phi - V_{\phi} = 0"
METRIC_EOM_TEX = (
    r"F R_{\mu\nu} - \tfrac{1}{2} g_{\mu\nu} F R + g_{\mu\nu}\Box F "
    r"- \nabla_{\mu}\nabla_{\nu} F - \tfrac{1}{2} \nabla_{\mu}\phi\,\nabla_{\nu}\phi "
    r"+ \tfrac{1}{4} g_{\mu\nu}(\nabla\phi)^2 + \tfrac{1}{2} g_{\mu\nu} V = 0"
)

AMBIGUITIES = [
    Ambiguity(
        id="amb-coupling-F",
        question="Is F(phi) an arbitrary function, or a fixed constant?",
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
        question="Dimension 4, mostly-plus signature, noether-default-v1 curvature signs?",
        kind="conventional",
        options=["noether-default-v1", "custom"],
    ),
    Ambiguity(
        id="amb-vary-wrt",
        question="Vary with respect to which field(s)?",
        kind="undecidable",
        options=["g and phi", "phi only"],
    ),
]

ELICITATION_ANSWERS = {
    "amb-coupling-F": "arbitrary-function",
    "amb-coupling-V": "arbitrary-function",
    "amb-conventions": "noether-default-v1",
    "amb-vary-wrt": "g and phi",
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
            ObjectDecl(name="F", kind="function", role="coupling", args=["phi"]),
            ObjectDecl(name="V", kind="function", role="coupling", args=["phi"]),
        ],
        action=Action(
            measure_tex=MEASURE_TEX,
            lagrangian=parse_lagrangian(LAGRANGIAN_TEX),
            lagrangian_tex=LAGRANGIAN_TEX,
        ),
        task=Task(type="vary", with_respect_to=["g", "phi"]),
        ambiguities=ambiguities,
    )
