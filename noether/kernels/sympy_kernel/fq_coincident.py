"""Coincident-gauge f(Q) verification: SymPy component cross-checks.

In the symmetric teleparallel geometry (curvature-free, torsion-free,
non-metric connection), the coincident gauge sets Gamma=0 so that
Q_{lambda mu nu} = partial_lambda g_{mu nu} and the f(Q) action
S = integral sqrt(-g) f(Q) becomes a pure-metric functional.

The non-metricity scalar Q is defined as the contraction
Q = Q_{alpha beta gamma} P^{alpha beta gamma}
where P is the non-metricity conjugate (superpotential). In coincident
gauge:
  Q_{lambda mu nu} = partial_lambda g_{mu nu}
  Q_alpha = g^{mu nu} Q_{alpha mu nu}           (Weyl / first trace)
  Qtilde_alpha = g^{mu nu} Q_{mu alpha nu}       (second trace)

The disformation tensor (coincident gauge Gamma=0):
  L^lambda_{mu nu} = (1/2)(Q^lambda_{mu nu}
                         - Q_mu^lambda_nu - Q_nu^lambda_mu)

Non-metricity conjugate (superpotential), following
Beltran Jimenez, Heisenberg, Koivisto (2018) eq (2.11):
  P^lambda_{mu nu} = (1/4)[-2 L^lambda_{mu nu}
                        + Q^lambda g_{mu nu} - Qtilde^lambda g_{mu nu}
                        - delta^lambda_{(mu} Q_{nu)}]

Boundary-term identity (eq 2.14 of De, Loo, Saridakis 2023):
  Q - R(g) = nabla_mu(Q^mu - Qtilde^mu)
where R(g) is the Levi-Civita Ricci scalar and nabla is the LC
covariant derivative. For the linear case f(Q) = Q, this means
S_Q = S_GR + boundary, so the EOM is G_{mu nu} = 0 (identical to
GR up to a boundary term).

The general f(Q) field equation (metric form, coincident gauge):
  E_{mu nu} = f'(Q) [G_{mu nu} - (1/2) g_{mu nu} Q]
            + 2 f''(Q) P_{mu nu}^lambda partial_lambda Q
            + (1/2) g_{mu nu} [f(Q) - Q f'(Q)]

Conventions: noether-default-v1 + metric-affine-v1.
"""

from __future__ import annotations

import sympy as sp

from noether.kernels.sympy_kernel.geometry import (
    ComponentGeometry,
    _clean,
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

    Q = Q_{alpha beta gamma} P^{alpha beta gamma}
    where P is the non-metricity conjugate computed by
    nonmetricity_conjugate.
    """
    Q_tens = coincident_gauge_Q_tensor(coords, g)
    n = Q_tens.shape[0]

    # P^{lambda}_{mu nu} from the corrected nonmetricity_conjugate
    P = nonmetricity_conjugate(coords, g, g_inv)

    # Q = Q_{alpha beta gamma} P^{alpha beta gamma}
    # P^{alpha beta gamma} = g^{beta mu} g^{gamma nu} P^alpha_{mu nu}
    Q_val = sp.Rational(0)
    for alpha in range(n):
        for beta in range(n):
            for gamma in range(n):
                P_up = sum(
                    g_inv[beta, mu] * g_inv[gamma, nu] * P[alpha, mu, nu]
                    for mu in range(n) for nu in range(n)
                )
                Q_val += Q_tens[alpha, beta, gamma] * P_up
    return _clean(Q_val)


def boundary_term_identity_residual(
    coords: list[sp.Symbol], g, g_inv,
) -> sp.Expr:
    """Residual of the boundary-term identity Q - R = nabla_mu(Q^mu - Qtilde^mu).

    In coincident gauge the non-metricity scalar Q satisfies (De, Loo,
    Saridakis 2023, eq 2.14):
      Q - R(g) = nabla_mu(Q^mu - Qtilde^mu)
    where R(g) is the Levi-Civita Ricci scalar and nabla is the LC
    covariant derivative.

    This identity implies that for the linear case f(Q) = Q, the EOM
    is the same as the Einstein equations G_{mu nu} = 0.

    Returns the residual: Q - R - nabla_mu(Q^mu - Qtilde^mu)
    (should be zero).
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    R = geom.ricci_scalar
    Q = Q_scalar(coords, g, g_inv)

    # Compute traces
    Q_trace, Qtilde = Q_traces(coords, g, g_inv)
    n = len(coords)

    # Boundary vector z^mu = Q^mu - Qtilde^mu
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

    # Residual: Q - R - div_z  (should be zero)
    return _clean(Q - R - div_z)


def nonmetricity_conjugate(
    coords: list[sp.Symbol], g, g_inv,
) -> sp.ImmutableDenseNDimArray:
    """Non-metricity conjugate P^{lambda}_{mu nu} in coincident gauge.

    Following Beltran Jimenez, Heisenberg, Koivisto (2018) eq (2.11):
      P^lambda_{mu nu} = (1/4)[-2 L^lambda_{mu nu}
                            + Q^lambda g_{mu nu} - Qtilde^lambda g_{mu nu}
                            - delta^lambda_{(mu} Q_{nu)}]

    where L^lambda_{mu nu} is the disformation tensor:
      L^lambda_{mu nu} = (1/2)(Q^lambda_{mu nu}
                              - Q_mu^lambda_nu - Q_nu^lambda_mu)

    and delta^lambda_{(mu} Q_{nu)} = (1/2)(delta^lambda_mu Q_nu
                                          + delta^lambda_nu Q_mu)
    is the weight-1/2 symmetrization.
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

    # Disformation L^lambda_{mu nu}
    L = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                L[lam, mu, nu] = _clean(sp.Rational(1, 2) * (
                    Q_mixed1[lam, mu, nu]
                    - Q_mixed2[lam, mu, nu]
                    - Q_mixed3[lam, mu, nu]
                ))

    # Traces
    Q_trace, Qtilde = Q_traces(coords, g, g_inv, Q_tens)
    Q_trace_up = [_clean(sum(g_inv[lam, a] * Q_trace[a] for a in range(n))) for lam in range(n)]
    Qtilde_up = [_clean(sum(g_inv[lam, a] * Qtilde[a] for a in range(n))) for lam in range(n)]

    # Build P^{lambda}_{mu nu}
    P = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(mu, n):
                # -2 L^lambda_{mu nu}
                val = sp.Rational(-2, 1) * L[lam, mu, nu]
                # + Q^lambda g_{mu nu}
                val += Q_trace_up[lam] * g[mu, nu]
                # - Qtilde^lambda g_{mu nu}
                val -= Qtilde_up[lam] * g[mu, nu]
                # - delta^lambda_{(mu} Q_{nu)}
                # = -(1/2)(delta^lambda_mu Q_nu + delta^lambda_nu Q_mu)
                if lam == mu:
                    val -= Q_trace[nu] / 2
                if lam == nu:
                    val -= Q_trace[mu] / 2
                val = _clean(sp.Rational(1, 4) * val)
                P[lam, mu, nu] = val
                P[lam, nu, mu] = val  # symmetric in last pair

    return sp.ImmutableDenseNDimArray(P)


def fQ_eom_general(
    coords: list[sp.Symbol], g, g_inv,
    f: sp.Expr, fp: sp.Expr, fpp: sp.Expr, Q_val: sp.Expr,
) -> sp.ImmutableDenseNDimArray:
    """General f(Q) EOM in coincident gauge.

    E_{mu nu} = f'(Q) [G_{mu nu} - (1/2) g_{mu nu} Q]
              + 2 f''(Q) P_{mu nu}^{lambda} partial_lambda Q
              + (1/2) g_{mu nu} [f(Q) - Q f'(Q)]

    This form follows the standard f(Q) literature. For f(Q) = Q
    (f'=1, f''=0, f-Qf'=0): E_{mu nu} = G_{mu nu} - (1/2) g_{mu nu} Q,
    which vanishes on shell by the boundary-term identity.
    All quantities are computed from the metric.
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
            # Term 1: f'(Q) [G_{mu nu} - (1/2) g_{mu nu} Q]
            t1 = _clean(fp * (G[mu, nu] - sp.Rational(1, 2) * g[mu, nu] * Q_val))

            # Term 2: 2 f''(Q) P^{lambda}_{mu nu} partial_lambda Q
            t2 = sp.Rational(0)
            for lam in range(n):
                t2 += P[lam, mu, nu] * dQ[lam]
            t2 = _clean(2 * fpp * t2)

            # Term 3: (1/2) g_{mu nu} [f(Q) - Q f'(Q)]
            t3 = _clean(sp.Rational(1, 2) * g[mu, nu] * (f - Q_val * fp))

            E[mu, nu] = _clean(t1 + t2 + t3)

    return sp.ImmutableDenseNDimArray(E)


def fQ_eom_linear(
    coords: list[sp.Symbol], g, g_inv,
) -> sp.ImmutableDenseNDimArray:
    """f(Q) = Q EOM in coincident gauge.

    Since Q = R + boundary, the f(Q) = Q action equals the Einstein-
    Hilbert action plus a boundary term, so the metric EOM is
    G_{mu nu} = 0. Returns the Einstein tensor G_{mu nu} which
    should vanish on-shell.
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    return geom.einstein
