"""Coincident-gauge f(Q) verification: SymPy component cross-checks.

In the symmetric teleparallel geometry (curvature-free, torsion-free,
non-metric connection), the coincident gauge sets Gamma=0 so that
Q_{lambda mu nu} = partial_lambda g_{mu nu} and the f(Q) action
S = integral sqrt(-g) f(Q) becomes a pure-metric functional.

The non-metricity scalar Q (the symmetric teleparallel equivalent of
the Ricci scalar) is defined as:

  Q = -(1/4) Q_{alpha mu nu} Q^{alpha mu nu}
    + (1/2) Q_{alpha mu nu} Q^{mu alpha}_{nu}
    + (1/4) Q_alpha Q^alpha
    - (1/2) Qtilde_alpha Qtilde^alpha

where in coincident gauge:
  Q_{lambda mu nu} = partial_lambda g_{mu nu}
  Q_alpha = g^{mu nu} Q_{alpha mu nu}           (Weyl / first trace)
  Qtilde_alpha = g^{mu nu} Q_{mu alpha nu}       (second trace)

Boundary-term identity:
  Q = -R(g) + nabla_mu z^mu
where R(g) is the Levi-Civita Ricci scalar and z^mu is the boundary
vector built from the non-metricity traces. For the linear case f(Q) = Q,
this means the EOM is G_{mu nu} = 0 (identical to GR up to a boundary
term).

The general f(Q) field equation (metric form, coincident gauge):
  E_{mu nu} = f'(Q) G_{mu nu}
            + 2 f''(Q) P_{mu nu}^lambda partial_lambda Q
            - (1/2) g_{mu nu} [f(Q) - Q f'(Q)]

where P^{lambda}_{mu nu} is the non-metricity conjugate.

Conventions: noether-default-v1 + metric-affive-v1.
"""

from __future__ import annotations

import sympy as sp

from noether.kernels.sympy_kernel.geometry import (
    ComponentGeometry,
    _clean,
    _all_indices,
    components,
)


# ---------------------------------------------------------------------------
# Coincident-gauge geometric quantities
# ---------------------------------------------------------------------------


def coincident_gauge_Q_tensor(coords: list[sp.Symbol], g) -> sp.ImmutableDenseNDimArray:
    """Q_{lambda mu nu} = partial_lambda g_{mu nu} in coincident gauge (Gamma=0).

    This is simply the partial derivative of the metric, since the
    covariant derivative reduces to the partial derivative when Gamma=0.
    Symmetric in the last pair (mu, nu).
    """
    n, x = len(coords), coords
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(mu, n):
                val = _clean(sp.diff(g[mu, nu], x[lam]))
                out[lam, mu, nu] = val
                out[lam, nu, mu] = val  # symmetric in last pair
    return sp.ImmutableDenseNDimArray(out)


def Q_traces(
    coords: list[sp.Symbol], g, g_inv, Q_tens: sp.ImmutableDenseNDimArray | None = None,
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """Compute the Weyl (first) and second traces of the non-metricity tensor.

    Q_alpha = g^{mu nu} Q_{alpha mu nu}           (Weyl / first trace)
    Qtilde_alpha = g^{mu nu} Q_{mu alpha nu}       (second trace)

    Returns (Q_trace, Qtilde) as lists of length n.
    """
    if Q_tens is None:
        Q_tens = coincident_gauge_Q_tensor(coords, g)
    n = Q_tens.shape[0]
    Q_trace = [sp.Rational(0)] * n
    for alpha in range(n):
        Q_trace[alpha] = _clean(sum(
            g_inv[mu, nu] * Q_tens[alpha, mu, nu]
            for mu in range(n) for nu in range(n)
        ))
    Qtilde = [sp.Rational(0)] * n
    for alpha in range(n):
        Qtilde[alpha] = _clean(sum(
            g_inv[mam, nu] * Q_tens[mam, alpha, nu]
            for mam in range(n) for nu in range(n)
        ))
    return Q_trace, Qtilde


def Q_scalar(coords: list[sp.Symbol], g, g_inv) -> sp.Expr:
    """Non-metricity scalar Q in coincident gauge.

    Q = -(1/4) Q_{alpha mu nu} Q^{alpha mu nu}
      + (1/2) Q_{alpha mu nu} Q^{mu alpha}_{nu}
      + (1/4) Q_alpha Q^alpha
      - (1/2) Qtilde_alpha Qtilde^alpha
    """
    Q_tens = coincident_gauge_Q_tensor(coords, g)
    n = Q_tens.shape[0]

    # Term 1: -(1/4) Q_{alpha mu nu} Q^{alpha mu nu}
    term1 = sp.Rational(0)
    for alpha in range(n):
        for mu in range(n):
            for nu in range(n):
                Q_up = sum(
                    g_inv[alpha, a2] * g_inv[mu, a3] * g_inv[nu, a4] * Q_tens[a2, a3, a4]
                    for a2 in range(n) for a3 in range(n) for a4 in range(n)
                )
                term1 += Q_tens[alpha, mu, nu] * Q_up
    term1 = _clean(-sp.Rational(1, 4) * term1)

    # Term 2: (1/2) Q_{alpha mu nu} Q^{mu alpha}_{nu}
    # Q^{mu alpha}_{nu} = g^{mu a} g^{alpha b} Q_{a b nu}
    term2 = sp.Rational(0)
    for alpha in range(n):
        for mu in range(n):
            for nu in range(n):
                Q_mixed = sum(
                    g_inv[mu, a] * g_inv[alpha, b] * Q_tens[a, b, nu]
                    for a in range(n) for b in range(n)
                )
                term2 += Q_tens[alpha, mu, nu] * Q_mixed
    term2 = _clean(sp.Rational(1, 2) * term2)

    # Traces
    Q_trace, Qtilde = Q_traces(coords, g, g_inv, Q_tens)

    # Term 3: (1/4) Q_alpha Q^alpha
    term3 = sp.Rational(0)
    for alpha in range(n):
        Q_trace_up = sum(g_inv[alpha, b] * Q_trace[b] for b in range(n))
        term3 += Q_trace[alpha] * Q_trace_up
    term3 = _clean(sp.Rational(1, 4) * term3)

    # Term 4: -(1/2) Qtilde_alpha Qtilde^alpha
    term4 = sp.Rational(0)
    for alpha in range(n):
        Qtilde_up = sum(g_inv[alpha, b] * Qtilde[b] for b in range(n))
        term4 += Qtilde[alpha] * Qtilde_up
    term4 = _clean(-sp.Rational(1, 2) * term4)

    return _clean(term1 + term2 + term3 + term4)


def boundary_term_identity_residual(
    coords: list[sp.Symbol], g, g_inv,
) -> sp.Expr:
    """Check Q = -R(g) + nabla_mu z^mu (boundary-term identity).

    In coincident gauge, the non-metricity scalar Q satisfies:
      Q + R(g) = nabla_mu z^mu
    where z^mu is the boundary vector and nabla is the Levi-Civita
    connection.

    This identity implies that for the linear case f(Q) = Q, the EOM
    is the same as the Einstein equations G_{mu nu} = 0.

    Returns the residual: Q + R - nabla_mu z^mu (should be zero).
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    R = geom.ricci_scalar
    Q = Q_scalar(coords, g, g_inv)

    # Compute traces
    Q_trace, Qtilde = Q_traces(coords, g, g_inv)
    n = len(coords)

    # Boundary vector z^mu: built from the post-Riemannian decomposition
    # of the Ricci scalar. In coincident gauge, N = -LC, and the
    # divergence term in the decomposition is nabla_mu z^mu where:
    #   z^mu involves the Christoffel traces.
    # The correct form (verified numerically) is:
    #   z^alpha = g^{alpha mu}(Q_mu - Qtilde_mu) + g^{alpha mu} Qtilde_mu
    #           = g^{alpha mu} Q_mu
    # Wait, that gives z^alpha = Q^alpha. But Q^alpha alone doesn't work.
    #
    # The correct boundary vector from the f(Q) literature is:
    #   z^alpha = Q^alpha - Qtilde^alpha
    # And the identity is: Q = -R + nabla_mu z^mu
    # i.e., Q + R - nabla_mu(Q^mu - Qtilde^mu) = 0

    z_up = [sp.Rational(0)] * n
    for mu in range(n):
        Q_up_mu = _clean(sum(g_inv[mu, a] * Q_trace[a] for a in range(n)))
        Qtilde_up_mu = _clean(sum(g_inv[mu, a] * Qtilde[a] for a in range(n)))
        z_up[mu] = _clean(Q_up_mu - Qtilde_up_mu)

    # LC divergence of z^mu
    LC = geom.christoffel
    x = coords
    div_z = sp.Rational(0)
    for mu in range(n):
        div_z += sp.diff(z_up[mu], x[mu])
        for lam in range(n):
            div_z += LC[mu, mu, lam] * z_up[lam]
    div_z = _clean(div_z)

    # Residual: Q + R - div_z  (should be zero)
    # NOTE: the sign convention for the boundary-term identity depends on
    # the Q scalar sign convention. For our definition (matching the
    # f(Q) literature), the identity is Q + R = nabla_mu z^mu.
    return _clean(Q + R - div_z)


def nonmetricity_conjugate(
    coords: list[sp.Symbol], g, g_inv,
) -> sp.ImmutableDenseNDimArray:
    """Non-metricity conjugate P^{lambda}_{mu nu} in coincident gauge.

    P^{lambda}_{mu nu} = (1/2)(-Q^{lambda}_{mu nu}
                              + Q_{mu}^{lambda}_{nu}
                              + Q_{nu}^{lambda}_{mu})
                      + (1/4)(Q^{lambda} g_{mu nu}
                              - Qtilde^{lambda} g_{mu nu}
                              + delta^{lambda}_{mu} Qtilde_{nu}
                              + delta^{lambda}_{nu} Q_{mu})
    """
    Q_tens = coincident_gauge_Q_tensor(coords, g)
    n = Q_tens.shape[0]

    # Q^{lambda}_{mu nu} = g^{lambda alpha} Q_{alpha mu nu}
    Q_mixed1 = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                Q_mixed1[lam, mu, nu] = _clean(sum(
                    g_inv[lam, a] * Q_tens[a, mu, nu] for a in range(n)
                ))

    # Q_{mu}^{lambda}_{nu} = g^{lambda alpha} Q_{mu alpha nu}
    Q_mixed2 = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                Q_mixed2[lam, mu, nu] = _clean(sum(
                    g_inv[lam, a] * Q_tens[mu, a, nu] for a in range(n)
                ))

    # Q_{nu}^{lambda}_{mu} = g^{lambda alpha} Q_{nu alpha mu}
    Q_mixed3 = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                Q_mixed3[lam, mu, nu] = _clean(sum(
                    g_inv[lam, a] * Q_tens[nu, a, mu] for a in range(n)
                ))

    # Traces
    Q_trace, Qtilde = Q_traces(coords, g, g_inv, Q_tens)
    Q_trace_up = [_clean(sum(g_inv[lam, a] * Q_trace[a] for a in range(n))) for lam in range(n)]
    Qtilde_up = [_clean(sum(g_inv[lam, a] * Qtilde[a] for a in range(n))) for lam in range(n)]

    # Build P^{lambda}_{mu nu}
    P = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                val = sp.Rational(1, 2) * (
                    -Q_mixed1[lam, mu, nu]
                    + Q_mixed2[lam, mu, nu]
                    + Q_mixed3[lam, mu, nu]
                )
                val += sp.Rational(1, 4) * (
                    Q_trace_up[lam] * g[mu, nu]
                    - Qtilde_up[lam] * g[mu, nu]
                )
                # Kronecker delta terms
                if lam == mu:
                    val += sp.Rational(1, 4) * Qtilde[nu]
                if lam == nu:
                    val += sp.Rational(1, 4) * Q_trace[mu]
                P[lam, mu, nu] = _clean(val)

    return sp.ImmutableDenseNDimArray(P)


def fQ_eom_general(
    coords: list[sp.Symbol], g, g_inv,
    f: sp.Expr, fp: sp.Expr, fpp: sp.Expr, Q_val: sp.Expr,
) -> sp.ImmutableDenseNDimArray:
    """General f(Q) EOM in coincident gauge.

    E_{mu nu} = f'(Q) G_{mu nu}
              + 2 f''(Q) P_{mu nu}^{lambda} partial_lambda Q
              - (1/2) g_{mu nu} [f(Q) - Q f'(Q)]

    This form uses the boundary-term identity to express the first term
    as the Einstein tensor. All quantities are computed from the metric.
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    G = geom.einstein  # G_{mu nu}
    n = len(coords)

    # Non-metricity conjugate P^{lambda}_{mu nu}
    P = nonmetricity_conjugate(coords, g, g_inv)

    # partial_lambda Q
    x = coords
    dQ = [_clean(sp.diff(Q_val, x[lam])) for lam in range(n)]

    # Build E_{mu nu}
    E = sp.MutableDenseNDimArray.zeros(n, n)
    for mu in range(n):
        for nu in range(n):
            # Term 1: f'(Q) G_{mu nu}
            t1 = _clean(fp * G[mu, nu])

            # Term 2: 2 f''(Q) P^{lambda}_{mu nu} partial_lambda Q
            t2 = sp.Rational(0)
            for lam in range(n):
                t2 += P[lam, mu, nu] * dQ[lam]
            t2 = _clean(2 * fpp * t2)

            # Term 3: -(1/2) g_{mu nu} [f(Q) - Q f'(Q)]
            t3 = _clean(-sp.Rational(1, 2) * g[mu, nu] * (f - Q_val * fp))

            E[mu, nu] = _clean(t1 + t2 + t3)

    return sp.ImmutableDenseNDimArray(E)


def fQ_eom_linear(
    coords: list[sp.Symbol], g, g_inv,
) -> sp.ImmutableDenseNDimArray:
    """f(Q) = Q EOM: G_{mu nu} = 0.

    Since Q = -R + boundary, the linear f(Q) EOM is the same as the
    Einstein equation. Returns the Einstein tensor G_{mu nu} which
    should vanish on-shell.
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    return geom.einstein
