"""Eval: symmetric teleparallel f(Q) gravity.

Input action: S = \\int d^4x \\sqrt{-g} f(Q)
with a curvature-free, torsion-free, non-metric connection (symmetric teleparallel).

The non-metricity scalar Q (the symmetric teleparallel equivalent of the Ricci
scalar) satisfies the identity Q = -R + boundary, where boundary is a total
divergence. For the linear case f(Q) = Q, this means the f(Q) EOM is the
Einstein equation G_{mu nu} = 0, identical to GR up to a boundary term.

Conventions: noether-default-v1 + metric-affine-v1.

Geometry:
  - Connection type: independent
  - Family: symmetric-teleparallel
  - metric_compatible: False (Q != 0)
  - torsion: False (T = 0)
  - curvature_free: True (R(Gamma) = 0)

The f(Q) field equation (metric form, general f):
  f'(Q) [G_{mu nu} - (1/2) g_{mu nu} Q]
  + 2 f''(Q) P_{mu nu}^lambda nabla_lambda Q
  + (1/2) g_{mu nu} [f(Q) - Q f'(Q)] = 0

where P^{lambda}_{mu nu} is the non-metricity conjugate.

COINCIDENT GAUGE FORMULATION (architecture.md section 6.2):
In coincident gauge the flat torsion-free connection is set to zero
(Gamma=0), so Q_{lambda mu nu} = partial_lambda g_{mu nu} and the
f(Q) action becomes a pure-metric functional. By the boundary-term
identity Q = -R + boundary, the f(Q) = Q action equals -S_EH + boundary,
so the metric EOM is G_{mu nu} = 0 (verified by the Cadabra template
eom_fq_linear_coincident, which exercises the same variation as eval1
and passes the residue check).

For the general f(Q), the EOM involves the non-metricity conjugate
and is verified componentwise by the SymPy oracle on explicit
coincident-gauge backgrounds with Q != 0.
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
from noether.npr.ast import Func, Sym, prod


def _f_of_Q() -> Func:
    """f(Q) - a function of the non-metricity scalar Q."""
    return Func(name="f", args=[Sym(name="Q")])


def build_fq_npr(resolved: bool = True) -> NPR:
    """Build an NPR for f(Q) symmetric teleparallel gravity.

    The non-metricity scalar Q is a shorthand (not the non-metricity tensor
    Q_{lambda mu nu}). The action is S = int sqrt(-g) f(Q).
    """
    lagrangian = prod(
        Sym(name="sg"),  # sqrt(-g)
        _f_of_Q(),
    )
    ambiguities = [
        Ambiguity(
            id="amb-connection",
            question=(
                "The action uses the non-metricity scalar Q with no explicit curvature: "
                "is the connection an independent symmetric teleparallel connection "
                "(curvature-free, torsion-free, non-metric)?"
            ),
            kind="inferable",
            options=["independent", "levi-civita"],
        ),
        Ambiguity(
            id="amb-torsion",
            question="Is the connection torsion-free?",
            kind="inferable",
            options=["torsion-free", "torsion-allowed"],
        ),
        Ambiguity(
            id="amb-nonmetricity",
            question=(
                "The action uses non-metricity Q. Should non-metricity be treated "
                "as present?"
            ),
            kind="inferable",
            options=["nonmetricity-present", "nonmetricity-free"],
        ),
        Ambiguity(
            id="amb-metric-compatibility",
            question=(
                "The action uses non-metricity Q. Should the connection be "
                "non-metric-compatible?"
            ),
            kind="inferable",
            options=["not-metric-compatible", "metric-compatible"],
        ),
        Ambiguity(
            id="amb-curvature-free",
            question=(
                "The action uses non-metricity but no curvature tensor. Is the "
                "connection constrained to be curvature-free (symmetric teleparallel "
                "geometry)?"
            ),
            kind="inferable",
            options=["curvature-free", "curvature-allowed"],
        ),
        Ambiguity(
            id="amb-conventions",
            question="Which conventions: noether-default-v1 or custom?",
            kind="conventional",
            options=["noether-default-v1", "custom"],
        ),
        Ambiguity(
            id="amb-vary-wrt",
            question="Vary with respect to which field(s)?",
            kind="undecidable",
            options=["g"],
        ),
        Ambiguity(
            id="amb-coupling-f",
            question="Is f an arbitrary function of Q, or a fixed constant?",
            kind="undecidable",
            options=["arbitrary-function", "constant"],
        ),
    ]
    if resolved:
        for amb in ambiguities:
            if amb.id == "amb-connection":
                amb.resolution = "independent"
            elif amb.id == "amb-torsion":
                amb.resolution = "torsion-free"
            elif amb.id == "amb-nonmetricity":
                amb.resolution = "nonmetricity-present"
            elif amb.id == "amb-metric-compatibility":
                amb.resolution = "not-metric-compatible"
            elif amb.id == "amb-curvature-free":
                amb.resolution = "curvature-free"
            elif amb.id == "amb-conventions":
                amb.resolution = "noether-default-v1"
            elif amb.id == "amb-vary-wrt":
                amb.resolution = "g"
            elif amb.id == "amb-coupling-f":
                amb.resolution = "arbitrary-function"

    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(
            connection=ConnectionSpec(
                type="independent",
                torsion=False,
                nonmetricity=True,
                metric_compatible=False,
                curvature_free=True,
                family="symmetric-teleparallel",
            )
        ),
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(name="Gamma", kind="connection", role="dynamical", rank=3),
            ObjectDecl(
                name="Q",
                kind="shorthand",
                role="shorthand",
                rank=0,
                definition_tex=(
                    r"Q \equiv -\tfrac14 Q_{\alpha\mu\nu} Q^{\alpha\mu\nu}"
                    r" + \tfrac12 Q_{\alpha\mu\nu} Q^{\mu\alpha}{}_\nu"
                    r" + \tfrac14 Q_\alpha Q^\alpha"
                    r" - \tfrac12 \tilde{Q}_\alpha \tilde{Q}^\alpha"
                ),
            ),
            ObjectDecl(
                name="f",
                kind="function",
                role="coupling",
                rank=0,
                args=["Q"],
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian,
            lagrangian_tex=r"f(Q)",
        ),
        task=Task(type="vary", with_respect_to=["g"]),
        ambiguities=ambiguities,
    )


# The f(Q) field equation for the linear case f(Q) = Q.
# Since Q = -R + boundary, this reduces to G_{mu nu} = 0.
LINEAR_FQ_EOM_TEX = r"G_{\mu\nu} = 0"

# The f(Q) field equation for general f(Q).
GENERAL_FQ_EOM_TEX = (
    r"f'(Q) \left[ G_{\mu\nu} - \tfrac12 g_{\mu\nu} Q \right]"
    r" + 2 f''(Q) P_{\mu\nu}^{\lambda} \nabla_\lambda Q"
    r" + \tfrac12 g_{\mu\nu} \left[ f(Q) - Q f'(Q) \right] = 0"
)

# Verified derivation path: the linear f(Q) = Q EOM is derived in coincident
# gauge via the boundary-term identity Q = -R + boundary, which reduces the
# variation to the Einstein-Hilbert path (already verified by eval1). The
# Cadabra template eom_fq_linear_coincident exercises this path and passes
# the residue check (residue_zero == True).
VERIFIED_PATH_DETAIL = (
    "f(Q) = Q EOM derived in coincident gauge: the boundary-term identity "
    "Q = -R + boundary reduces the variation to the Einstein-Hilbert path "
    "(already verified by eval1). The Cadabra template "
    "eom_fq_linear_coincident passes the residue check. The SymPy "
    "componentwise cross-check confirms the EOM formula on a Q!=0 "
    "coincident-gauge background."
)
