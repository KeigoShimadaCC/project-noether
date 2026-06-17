"""Eval: metric teleparallel f(T) gravity.

Input action: S = \\int d^4x \\sqrt{-g} f(T)
with a curvature-free, metric-compatible, torsionful connection (teleparallel).

The torsion scalar T (the teleparallel equivalent of the Ricci scalar) satisfies
the identity T = -R + B, where B = 2 nabla_mu T^mu is a boundary term (total
divergence). For the linear case f(T) = T, this means the f(T) EOM is the
Einstein equation G_{mu nu} = 0, identical to GR up to a boundary term.

Conventions: noether-default-v1 + metric-affine-v1.

Geometry:
  - Connection type: independent
  - Family: teleparallel
  - metric_compatible: True (Q = 0)
  - torsion: True (T != 0)
  - curvature_free: True (R(Gamma) = 0)

The f(T) field equation (metric form, general f):
  f'(T) [G_{mu nu} - (1/2) g_{mu nu} T]
  + S_{mu nu}^rho nabla_rho f'(T)
  + (1/2) g_{mu nu} [f(T) - T f'(T)] = 0

where S^{rho mu nu} = (1/2)(K^{rho mu nu} + g^{rho mu} T^nu - g^{rho nu} T^mu)
is the modified superpotential.

NOTE (blocker): the current Cadabra/SymPy infrastructure does not support the
vierbein/tetrad formulation required for f(T) gravity. The metric variation of
the torsion scalar T involves the metric variation of the Weitzenbock connection
(constrained to be curvature-free), which requires either:
  (a) tetrad (vierbein) formulation, or
  (b) explicit treatment of the curvature-free constraint on the connection.

The existing derive path handles either Levi-Civita (connection dependent on g)
or Palatini (connection independent), but NOT a constrained connection. The
derivation is therefore gated with a blocker detail.

The linear case f(T) = T is equivalent to GR by the boundary-term identity and
can be verified via SymPy cross-check on explicit backgrounds.
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


def _f_of_T() -> Func:
    """f(T) - a function of the torsion scalar T."""
    return Func(name="f", args=[Sym(name="T")])


def build_ft_npr(resolved: bool = True) -> NPR:
    """Build an NPR for f(T) teleparallel gravity.

    The torsion scalar T is a shorthand (not the torsion tensor T^lambda_{mu nu}).
    The action is S = int sqrt(-g) f(T).
    """
    lagrangian = prod(
        Sym(name="sg"),  # sqrt(-g)
        _f_of_T(),
    )
    ambiguities = [
        Ambiguity(
            id="amb-connection",
            question=(
                "The action uses the torsion scalar T with no explicit curvature: "
                "is the connection an independent teleparallel connection "
                "(curvature-free, metric-compatible, torsionful)?"
            ),
            kind="inferable",
            options=["independent", "levi-civita"],
        ),
        Ambiguity(
            id="amb-torsion",
            question="The action uses torsion T. Should torsion be treated as present?",
            kind="inferable",
            options=["torsion-present", "torsion-free"],
        ),
        Ambiguity(
            id="amb-nonmetricity",
            question="Is the connection metric-compatible (no non-metricity)?",
            kind="inferable",
            options=["nonmetricity-free", "nonmetricity-allowed"],
        ),
        Ambiguity(
            id="amb-metric-compatibility",
            question="Should the connection be metric-compatible?",
            kind="inferable",
            options=["metric-compatible", "not-metric-compatible"],
        ),
        Ambiguity(
            id="amb-curvature-free",
            question=(
                "The action uses torsion but no curvature tensor. Is the connection "
                "constrained to be curvature-free (teleparallel geometry)?"
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
            question="Is f an arbitrary function of T, or a fixed constant?",
            kind="undecidable",
            options=["arbitrary-function", "constant"],
        ),
    ]
    if resolved:
        for amb in ambiguities:
            if amb.id == "amb-connection":
                amb.resolution = "independent"
            elif amb.id == "amb-torsion":
                amb.resolution = "torsion-present"
            elif amb.id == "amb-nonmetricity":
                amb.resolution = "nonmetricity-free"
            elif amb.id == "amb-metric-compatibility":
                amb.resolution = "metric-compatible"
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
                torsion=True,
                nonmetricity=False,
                metric_compatible=True,
                curvature_free=True,
                family="teleparallel",
            )
        ),
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(name="Gamma", kind="connection", role="dynamical", rank=3),
            ObjectDecl(
                name="T",
                kind="shorthand",
                role="shorthand",
                rank=0,
                definition_tex=r"T \equiv S^{\rho\mu\nu} T_{\rho\mu\nu}",
            ),
            ObjectDecl(
                name="f",
                kind="function",
                role="coupling",
                rank=0,
                args=["T"],
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=lagrangian,
            lagrangian_tex=r"f(T)",
        ),
        task=Task(type="vary", with_respect_to=["g"]),
        ambiguities=ambiguities,
    )


# The f(T) field equation for the linear case f(T) = T.
# Since T = -R + boundary, this reduces to G_{mu nu} = 0.
LINEAR_FT_EOM_TEX = r"G_{\mu\nu} = 0"

# The f(T) field equation for general f(T).
GENERAL_FT_EOM_TEX = (
    r"f'(T) \left[ G_{\mu\nu} - \tfrac12 g_{\mu\nu} T \right]"
    r" + S_{\mu\nu}^{\rho} \nabla_\rho f'(T)"
    r" + \tfrac12 g_{\mu\nu} \left[ f(T) - T f'(T) \right] = 0"
)

BLOCKER_DETAIL = (
    "f(T) EOM derivation gated: the current derive infrastructure handles "
    "Levi-Civita (connection dependent on g) and Palatini (connection independent "
    "of g) variations, but NOT a constrained connection where the connection "
    "depends on g through the curvature-free constraint R(Gamma) = 0. "
    "The f(T) metric variation requires either (a) vierbein/tetrad formulation "
    "or (b) explicit enforcement of the curvature-free constraint during variation. "
    "Neither is currently supported. The linear case f(T) = T is equivalent to "
    "GR by the boundary-term identity T = -R + 2 nabla_mu T^mu."
)
