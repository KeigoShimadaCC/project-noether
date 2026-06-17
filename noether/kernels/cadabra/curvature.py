r"""Reusable covariant-curvature reduction primitives for Cadabra scripts.

A generic ``\nabla{#}::Derivative`` does three things Cadabra will not do on its
own: it does not commute covariant derivatives (picking up curvature), it does
not fold a metric-contracted Riemann tensor into the Ricci tensor or scalar, and
it does not know the second covariant derivative of a scalar is symmetric. The
higher Horndeski densities ``G4(phi, X) R`` and G5 need all three to reduce
their equations of motion to second order (the no-Ostrogradski cancellation).

This module supplies the missing identities as small Cadabra substitution
snippets, parameterized by the expression variable and field name. Each primitive
is pinned by a residue-checking test in ``tests/test_curvature.py`` so the layer
is audited the same way the frozen templates are, rather than trusted by eye.

Conventions: noether-default-v1.
  - Riemann: R^{rho}_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - ... ;
  - Ricci:   R_{mu nu} = R^{lambda}_{mu lambda nu},  R = g^{mu nu} R_{mu nu};
  - commutator on a covector: [nabla_a, nabla_b] V_c = - R^{d}_{c a b} V_d,
    so [nabla_a, nabla_b] nabla_c phi = - R^{d}_{c a b} nabla_d phi.

The commutator snippet matches the explicit antisymmetric difference
``nabla_a nabla_b nabla_c phi - nabla_b nabla_a nabla_c phi`` and rewrites it to
the Riemann term; callers arrange the third-derivative terms into that difference
(or symmetrize the inner Hessian first) before applying it.
"""

from __future__ import annotations

# Declarations to append after the base index/metric block. The symmetric H
# stand-in carries the (symmetric) scalar Hessian during a reduction.
CURVATURE_DECL = (
    r"R_{\mu\nu\rho\sigma}::RiemannTensor."
    "\n"
    r"R_{\mu\nu}::Symmetric."
    "\n"
    r"H_{\mu\nu}::Symmetric."
)


def fold_ricci(ex: str = "ex") -> str:
    r"""Fold a metric-contracted Riemann tensor into the Ricci tensor:
    ``g^{mu nu} R_{alpha mu beta nu} -> R_{alpha beta}`` (apply after
    ``canonicalise``, which lands the contraction in this shape)."""
    return f"substitute({ex}, $g^{{\\mu\\nu}} R_{{\\alpha\\mu\\beta\\nu}} -> R_{{\\alpha\\beta}}$);"


def fold_scalar(ex: str = "ex") -> str:
    r"""Fold the Ricci trace into the curvature scalar:
    ``g^{alpha beta} R_{alpha beta} -> R``."""
    return f"substitute({ex}, $g^{{\\alpha\\beta}} R_{{\\alpha\\beta}} -> R$);"


def commute_third_derivative(field: str, ex: str = "ex") -> str:
    r"""Reduce the antisymmetrized third derivative of ``field`` to its Riemann
    term, ``[nabla_a, nabla_b] nabla_c field = - R^{d}_{c a b} nabla_d field``,
    written with the Riemann index lowered so it canonicalises cleanly."""
    return (
        f"substitute({ex}, $"
        f"\\nabla_{{\\mu}}{{\\nabla_{{\\nu}}{{\\nabla_{{\\rho}}{{{field}}}}}}} "
        f"- \\nabla_{{\\nu}}{{\\nabla_{{\\mu}}{{\\nabla_{{\\rho}}{{{field}}}}}}} "
        f"-> - g^{{\\delta\\lambda}} R_{{\\lambda\\rho\\mu\\nu}} \\nabla_{{\\delta}}{{{field}}}$);"
    )


def commute_third_derivative_oneway(field: str, ex: str = "ex") -> str:
    r"""Single-term form of the third-derivative commutator,
    ``nabla_a nabla_b nabla_c field -> nabla_b nabla_a nabla_c field
    - R^{d}_{c a b} nabla_d field``.

    Unlike :func:`commute_third_derivative`, this matches one term at a time, so
    it fires even when the derivative carries a coupling or metric factor (as it
    always does in a real equation of motion). It swaps the outer two derivative
    indices; one application moves each third derivative one transposition toward
    a chosen ordering. Apply it in a controlled pass (it is not idempotent: a
    second blind pass would swap back)."""
    return (
        f"substitute({ex}, $"
        f"\\nabla_{{\\mu}}{{\\nabla_{{\\nu}}{{\\nabla_{{\\rho}}{{{field}}}}}}} "
        f"-> \\nabla_{{\\nu}}{{\\nabla_{{\\mu}}{{\\nabla_{{\\rho}}{{{field}}}}}}} "
        f"- g^{{\\delta\\lambda}} R_{{\\lambda\\rho\\mu\\nu}} \\nabla_{{\\delta}}{{{field}}}$);"
    )


def commute_fourth_cross(field: str, ex: str = "ex") -> str:
    r"""Targeted reduction of the cross-contracted fourth derivative that the
    quartic Horndeski counterterm produces:
    ``g^{a c} g^{b d} nabla_a nabla_b nabla_c nabla_d field``
    (metric on the 1st-3rd and 2nd-4th derivatives) into the
    ``box^2`` ordering plus its middle-pair curvature term.

    This is the move that closes the quartic no-Ostrogradski combination
    ``box^2 field - nabla_a nabla_b nabla^b nabla^a field``: it matches only the
    cross contraction (leaving an already-canonical ``box^2`` term untouched), so
    after canonicalise the two fourth-derivative pieces cancel and only the
    curvature coupling ``nabla R . nabla field + R . nabla nabla field`` remains.
    A blind single-term swap instead oscillates, trading the two contractions'
    identities back and forth."""
    return (
        f"substitute({ex}, $"
        f"g^{{\\mu\\rho}} g^{{\\nu\\sigma}} "
        f"\\nabla_{{\\mu}}{{\\nabla_{{\\nu}}{{\\nabla_{{\\rho}}{{\\nabla_{{\\sigma}}{{{field}}}}}}}}} "
        f"-> g^{{\\mu\\rho}} g^{{\\nu\\sigma}} "
        f"\\nabla_{{\\mu}}{{\\nabla_{{\\rho}}{{\\nabla_{{\\nu}}{{\\nabla_{{\\sigma}}{{{field}}}}}}}}} "
        f"- g^{{\\mu\\rho}} g^{{\\nu\\sigma}} "
        f"\\nabla_{{\\mu}}{{ g^{{\\epsilon\\lambda}} R_{{\\lambda\\sigma\\nu\\rho}} "
        f"\\nabla_{{\\epsilon}}{{{field}}} }}$);"
    )


def hessian_to_symmetric(field: str, ex: str = "ex") -> str:
    r"""Route the scalar Hessian ``nabla_mu nabla_nu field`` through the symmetric
    stand-in ``H_{mu nu}``, so its (vanishing) antisymmetric part drops under
    canonicalise. Pair with :func:`hessian_from_symmetric` to restore it."""
    return f"substitute({ex}, $\\nabla_{{\\mu}}{{\\nabla_{{\\nu}}{{{field}}}}} -> H_{{\\mu\\nu}}$);"


def hessian_from_symmetric(field: str, ex: str = "ex") -> str:
    r"""Restore the scalar Hessian from the symmetric stand-in ``H_{mu nu}``."""
    return f"substitute({ex}, $H_{{\\mu\\nu}} -> \\nabla_{{\\mu}}{{\\nabla_{{\\nu}}{{{field}}}}}$);"


def contracted_bianchi(ex: str = "ex") -> str:
    r"""Apply the once-contracted second Bianchi identity
    ``g^{mu nu} nabla_mu R_{nu beta} = 1/2 nabla_beta R``.

    This is a citable standard result (the divergence of the Einstein tensor
    vanishes), not derived here; it enters wherever a derivative of the Ricci
    tensor survives the reduction. Marked as such per AGENTS.md rule 1.

    **Levi-Civita only.** Under torsion or non-metricity this identity is
    modified and must not be reused; see the metric-affine primitives below."""
    return (
        f"substitute({ex}, $g^{{\\mu\\nu}} \\nabla_{{\\mu}}{{R_{{\\nu\\beta}}}} "
        f"-> 1/2 \\nabla_{{\\beta}}{{R}}$);"
    )


# ---------------------------------------------------------------------------
# Metric-affine (independent-connection) primitives.
#
# These operate with an independent affine connection Gamma^lambda_{mu nu}
# declared as a \partial-Depends object (no symmetry in the lower pair;
# torsion T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
# is generally nonzero).  The key difference from the Levi-Civita primitives
# above: R_{mu nu} carries NO symmetry declaration, so canonicalise will
# not symmetrize it.  The existing Levi-Civita primitives are preserved as
# the T = Q = 0 special case; do not fork a parallel code path where the
# general primitive can subsume them.
#
# Conventions: noether-default-v1, extended per architecture.md section 7.
#   R^rho_{sigma mu nu}(Gamma) = d_mu Gamma^rho_{nu sigma}
#                                - d_nu Gamma^rho_{mu sigma}
#                                + Gamma^rho_{mu lam} Gamma^lam_{nu sigma}
#                                - Gamma^rho_{nu lam} Gamma^lam_{mu sigma}
#   R_{sigma nu} = R^lambda_{sigma lambda nu}
# ---------------------------------------------------------------------------

# Declaration block for metric-affine geometry.  Like CURVATURE_DECL but
# WITHOUT R_{mu nu}::Symmetric.  The connection object (default name G)
# must be separately declared as \partial-Depends by the caller (see
# AFFINE_CONNECTION_DEPENDS).  The symmetric H stand-in is kept so that
# Levi-Civita reductions that route through it still work in the T=Q=0 limit.
AFFINE_CURVATURE_DECL = (
    r"R_{\mu\nu\rho\sigma}::RiemannTensor."
    "\n"
    r"H_{\mu\nu}::Symmetric."
)

# Depends block for the independent connection object (default name G).
# Append after the metric/InverseMetric/KroneckerDelta declarations.
AFFINE_CONNECTION_DEPENDS = (
    r"{G^{\lambda}_{\mu\nu}, g_{\mu\nu}, g^{\mu\nu}}::Depends(\partial{#})."
)

# Same, using nabla instead of partial derivatives (for scripts that keep
# nabla as the derivative operator throughout).
AFFINE_CONNECTION_DEPENDS_NABLA = (
    r"{G^{\lambda}_{\mu\nu}, g_{\mu\nu}, g^{\mu\nu}}::Depends(\nabla{#})."
)


def expand_riemann_affine(conn: str = "G", ex: str = "ex") -> str:
    r"""Expand the fully-lowered Riemann tensor in terms of partial
    derivatives of an independent connection:

    ``R_{rho sigma mu nu} -> g_{rho alpha} (d_mu conn^alpha_{nu sigma}
        - d_nu conn^alpha_{mu sigma} + conn^alpha_{mu lam} conn^lam_{nu sigma}
        - conn^alpha_{nu lam} conn^lam_{mu sigma})``

    No ``R_{mu nu}::Symmetric`` declaration is assumed; the Ricci tensor
    is not symmetric in general.  Use with `\partial{#}::PartialDerivative`
    and the `conn` object declared as `\partial`-Depends (following the
    eval2 Palatini pattern).  The connection name defaults to ``G``."""
    return (
        f"substitute({ex}, $"
        f"R_{{\\rho\\sigma\\mu\\nu}} -> "
        f"g_{{\\rho\\alpha}} ( "
        f"\\partial_{{\\mu}}{{{conn}^{{\\alpha}}_{{\\nu\\sigma}}}} "
        f"- \\partial_{{\\nu}}{{{conn}^{{\\alpha}}_{{\\mu\\sigma}}}} "
        f"+ {conn}^{{\\alpha}}_{{\\mu\\lambda}} {conn}^{{\\lambda}}_{{\\nu\\sigma}} "
        f"- {conn}^{{\\alpha}}_{{\\nu\\lambda}} {conn}^{{\\lambda}}_{{\\mu\\sigma}} "
        f")$);"
    )


def expand_ricci_affine(conn: str = "G", ex: str = "ex") -> str:
    r"""Expand the Ricci tensor in terms of partial derivatives of an
    independent connection (no symmetry assumed):

    ``R_{sigma nu} -> d_lambda conn^lambda_{nu sigma}
        - d_nu conn^lambda_{lambda sigma}
        + conn^lambda_{lambda rho} conn^rho_{nu sigma}
        - conn^lambda_{nu rho} conn^rho_{lambda sigma}``

    This is the traced form of :func:`expand_riemann_affine`:
    ``R_{sigma nu} = R^lambda_{sigma lambda nu}``.
    Because the connection is independent, ``R_{sigma nu} != R_{nu sigma}``
    in general; the Ricci tensor recovers symmetry only when torsion
    vanishes (T=0) and the connection is metric-compatible (Q=0)."""
    return (
        f"substitute({ex}, $"
        f"R_{{\\sigma\\nu}} -> "
        f"\\partial_{{\\lambda}}{{{conn}^{{\\lambda}}_{{\\nu\\sigma}}}} "
        f"- \\partial_{{\\nu}}{{{conn}^{{\\lambda}}_{{\\lambda\\sigma}}}} "
        f"+ {conn}^{{\\lambda}}_{{\\lambda\\rho}} {conn}^{{\\rho}}_{{\\nu\\sigma}} "
        f"- {conn}^{{\\lambda}}_{{\\nu\\rho}} {conn}^{{\\rho}}_{{\\lambda\\sigma}} "
        f"$);"
    )


def fold_ricci_affine(ex: str = "ex") -> str:
    r"""Fold a metric-contracted Riemann tensor into the Ricci tensor
    on the metric-affine path (no symmetry assumed):

    ``g^{mu nu} R_{alpha mu beta nu} -> R_{alpha beta}``

    This is the same algebraic substitution as :func:`fold_ricci`, but
    it is used in a context where ``R_{mu nu}`` has no ``::Symmetric``
    declaration, so ``canonicalise`` will not reorder its indices."""
    return f"substitute({ex}, $g^{{\\mu\\nu}} R_{{\\alpha\\mu\\beta\\nu}} -> R_{{\\alpha\\beta}}$);"


# ---------------------------------------------------------------------------
# Torsion primitives (independent-connection path).
#
# Torsion: T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
# Convention: noether-default-v1 (AGENTS.md section 5).  The torsion tensor
# is antisymmetric in the lower pair.
#
# The irreducible decomposition splits T into three parts under the Lorentz
# group:
#   _(1)T^lambda_{mu nu} = (1/3)(delta^lambda_mu T_nu - delta^lambda_nu T_mu)
#                          trace-vector part (4 components in dim 4)
#   _(2)T^lambda_{mu nu} = -(1/6) epsilon^lambda_{mu nu rho} A^rho
#                          axial-vector part (4 components in dim 4)
#   q^lambda_{mu nu}     = T^lambda_{mu nu} - _(1)T - _(2)T
#                          traceless-tensor part (16 components in dim 4)
# where:
#   T_mu = T^lambda_{lambda mu}            (torsion trace vector)
#   A^rho = epsilon^{rho sigma kappa lambda} T_{sigma kappa lambda} / 6
#                                          (torsion axial vector)
#
# The decomposition is distinct from the contortion form K(T) defined by
# the post-Riemannian decomposition Gamma = LC + K(T) + L(Q).  K(T) is
# built from T but with different index structure; the irreducible parts
# above are the Lorentz-irreducible decomposition of T itself.
#
# Convention sign note: the axial vector sign follows from the definition
# A^rho = (1/6) epsilon^{rho sigma kappa lambda} T_{sigma kappa lambda},
# with epsilon^{0123} = +1/sqrt(-g).  The decomposition formula uses
# epsilon^lambda_{mu nu rho} A^rho with the minus sign shown, which
# reproduces the totally antisymmetric part of T_{lambda mu nu}.
# These signs are NOT asserted from memory; they are residue-pinned and
# SymPy-cross-checked.
# ---------------------------------------------------------------------------

# Declaration for the torsion tensor T^lambda_{mu nu} and its trace T_mu.
# The TableauSymmetry on T enforces antisymmetry in the lower pair.
TORSION_DECL = (
    r"T^{\lambda}_{\mu\nu}::TableauSymmetry(shape={1,1}, indices={1,2})."
    "\n"
    r"T_{\mu}::Depends(\partial{#})."
)

# Same but with nabla as the derivative operator.
TORSION_DECL_NABLA = (
    r"T^{\lambda}_{\mu\nu}::TableauSymmetry(shape={1,1}, indices={1,2})."
    "\n"
    r"T_{\mu}::Depends(\nabla{#})."
)

# Epsilon tensor declaration for the axial-vector decomposition.
# Must be appended alongside a metric declaration for the delta= option.
EPSILON_DECL = r"\epsilon_{\mu\nu\rho\sigma}::EpsilonTensor(delta=g_{\mu\nu})."


def define_torsion(conn: str = "G", ex: str = "ex") -> str:
    r"""Define the torsion tensor as the antisymmetric difference of the
    connection:

    ``T^lambda_{mu nu} -> Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}``

    Applies the substitution to the named expression. The connection object
    ``conn`` defaults to ``G``, matching the independent-connection convention
    in :data:`AFFINE_CONNECTION_DEPENDS`."""
    return (
        f"substitute({ex}, $"
        f"T^{{\\lambda}}_{{\\mu\\nu}} -> "
        f"{conn}^{{\\lambda}}_{{\\mu\\nu}} - {conn}^{{\\lambda}}_{{\\nu\\mu}}$);"
    )


def torsion_trace_vector(conn: str = "G", ex: str = "ex") -> str:
    r"""Define the torsion trace vector:

    ``T_mu -> g^{lambda kappa}(Gamma^kappa_{lambda mu} - Gamma^kappa_{mu lambda})``

    This is the contraction T^lambda_{lambda mu} written out explicitly
    in terms of the independent connection."""
    return (
        f"substitute({ex}, $"
        f"T_{{\\mu}} -> "
        f"g^{{\\lambda\\kappa}}"
        f"({conn}^{{\\kappa}}_{{\\lambda\\mu}} - {conn}^{{\\kappa}}_{{\\mu\\lambda}})$);"
    )


def torsion_trace_part(ex: str = "ex") -> str:
    r"""Substitute the trace-vector irreducible part of the torsion:

    ``t1^lambda_{mu nu} -> (1/3)(delta^lambda_mu T_nu - delta^lambda_nu T_mu)``

    The trace vector T_mu must already be defined or substituted in the
    expression.  Uses the KroneckerDelta declaration from the base block."""
    return (
        f"substitute({ex}, $"
        f"t1^{{\\lambda}}_{{\\mu\\nu}} -> "
        f"(1/3)(g^{{\\lambda}}_{{\\mu}} T_{{\\nu}} - g^{{\\lambda}}_{{\\nu}} T_{{\\mu}})$);"
    )


def torsion_axial_part(ex: str = "ex") -> str:
    r"""Substitute the axial-vector irreducible part of the torsion:

    ``t2^lambda_{mu nu} -> -(1/6) epsilon^lambda_{mu nu rho} A^rho``

    The axial vector A^rho and the epsilon tensor must be declared
    (see :data:`EPSILON_DECL`).  A^rho is typically defined as a named
    expression computed from the torsion, not substituted by this function."""
    return (
        f"substitute({ex}, $"
        f"t2^{{\\lambda}}_{{\\mu\\nu}} -> "
        f"-(1/6) \\epsilon^{{\\lambda}}_{{\\mu\\nu\\rho}} A^{{\\rho}}$);"
    )


def torsion_traceless_part(ex: str = "ex") -> str:
    r"""Define the traceless-tensor irreducible part as the remainder:

    ``q^lambda_{mu nu} -> T^lambda_{mu nu} - t1^lambda_{mu nu} - t2^lambda_{mu nu}``

    The trace and axial parts (t1, t2) must already be present in the
    expression for this to produce the correct result."""
    return (
        f"substitute({ex}, $"
        f"q^{{\\lambda}}_{{\\mu\\nu}} -> "
        f"T^{{\\lambda}}_{{\\mu\\nu}} - t1^{{\\lambda}}_{{\\mu\\nu}} - t2^{{\\lambda}}_{{\\mu\\nu}}$);"
    )


def reassemble_torsion(ex: str = "ex") -> str:
    r"""Substitute the reassembled torsion from its three irreducible parts:

    ``T^lambda_{mu nu} -> t1^lambda_{mu nu} + t2^lambda_{mu nu} + q^lambda_{mu nu}``

    This is the reconstruction substitution for the residue check: applying
    it and then subtracting the original T should yield zero."""
    return (
        f"substitute({ex}, $"
        f"T^{{\\lambda}}_{{\\mu\\nu}} -> "
        f"t1^{{\\lambda}}_{{\\mu\\nu}} + t2^{{\\lambda}}_{{\\mu\\nu}} + q^{{\\lambda}}_{{\\mu\\nu}}$);"
    )


# ---------------------------------------------------------------------------
# Torsionful commutator and non-symmetric Hessian (independent-connection).
#
# These generalize the Levi-Civita commutator and symmetric Hessian to a
# connection with torsion.  The key differences from the LC primitives above:
#
# - The commutator gains a torsion term:
#   [nabla_a, nabla_b] nabla_c field = -R^d_{cab} nabla_d field
#                                       - T^d_{ab} nabla_d nabla_c field
#   When T=0 this reduces to the existing commute_third_derivative (the
#   torsion term vanishes and R^d_{cab} is the same Riemann contraction).
#
# - The scalar Hessian is NON-symmetric under torsion:
#   nabla_mu nabla_nu field - nabla_nu nabla_mu field = -T^lambda_{mu nu} nabla_lambda field
#   The LC hessian_to_symmetric (which routes through a symmetric stand-in
#   H_{mu nu}) is INVALID here; it silently drops the antisymmetric part.
#
# Conventions: noether-default-v1, extended per architecture.md section 7.
#   T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
#   R^rho_{sigma mu nu}(Gamma) as defined in AFFINE_CURVATURE_DECL.
#
# The torsion tensor T must be declared (see TORSION_DECL) and the
# connection must be declared as Depends (see AFFINE_CONNECTION_DEPENDS or
# AFFINE_CONNECTION_DEPENDS_NABLA) before applying these primitives.
# ---------------------------------------------------------------------------


def commute_third_derivative_affine(field: str, ex: str = "ex") -> str:
    r"""Reduce the antisymmetrized third derivative of ``field`` under a
    torsionful connection, carrying the torsion term the LC primitive omits:

    ``[nabla_a, nabla_b] nabla_c field = -R^d_{cab} nabla_d field
                                         - T^d_{ab} nabla_d nabla_c field``

    Written with the Riemann index lowered for clean canonicalisation.
    When torsion vanishes (T=0) this reduces to :func:`commute_third_derivative`.

    Requires ``T^{\\lambda}_{\\mu\\nu}`` declared (see :data:`TORSION_DECL`).
    The ``R_{\\mu\\nu\\rho\\sigma}`` RiemannTensor declaration must be present
    (see :data:`AFFINE_CURVATURE_DECL`)."""
    return (
        f"substitute({ex}, $"
        f"\\nabla_{{\\mu}}{{\\nabla_{{\\nu}}{{\\nabla_{{\\rho}}{{{field}}}}}}} "
        f"- \\nabla_{{\\nu}}{{\\nabla_{{\\mu}}{{\\nabla_{{\\rho}}{{{field}}}}}}} "
        f"-> - g^{{\\delta\\lambda}} R_{{\\lambda\\rho\\mu\\nu}} \\nabla_{{\\delta}}{{{field}}}"
        f" - T^{{\\delta}}_{{\\mu\\nu}} \\nabla_{{\\delta}}{{\\nabla_{{\\rho}}{{{field}}}}}$);"
    )


def hessian_antisymmetry_affine(field: str, ex: str = "ex") -> str:
    r"""Route the antisymmetric part of the scalar Hessian through the
    torsion tensor under a torsionful connection:

    ``nabla_mu nabla_nu field - nabla_nu nabla_mu field
        -> -T^lambda_{mu nu} nabla_lambda field``

    This is the general (torsionful) version of the scalar Hessian identity.
    Under Levi-Civita (T=0) the antisymmetric part vanishes and the Hessian
    is symmetric, so :func:`hessian_to_symmetric` is valid.  Under torsion
    the antisymmetric part is NONZERO and ``hessian_to_symmetric`` would
    silently drop it, producing a wrong result (the torsion trap).

    Requires ``T^{\\lambda}_{\\mu\\nu}`` declared (see :data:`TORSION_DECL`)."""
    return (
        f"substitute({ex}, $"
        f"\\nabla_{{\\mu}}{{\\nabla_{{\\nu}}{{{field}}}}} "
        f"- \\nabla_{{\\nu}}{{\\nabla_{{\\mu}}{{{field}}}}} "
        f"-> - T^{{\\lambda}}_{{\\mu\\nu}} \\nabla_{{\\lambda}}{{{field}}}$);"
    )
