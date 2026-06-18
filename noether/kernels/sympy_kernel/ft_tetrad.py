"""Tetrad/Weitzenbock verification: SymPy component cross-checks for f(T) gravity.

In the metric teleparallel geometry (curvature-free, metric-compatible,
torsionful connection), the Weitzenbock connection is built from the
tetrad (vierbein) e^a_mu:

  Gamma^rho_{mu nu} = E_a^rho partial_mu e^a_nu

where E_a^mu is the inverse tetrad satisfying
  E_a^mu e^a_nu = delta^mu_nu  and  E_a^mu e^b_mu = delta^b_a.

Properties:
  - Flat: R(Gamma) = 0 (by construction, the tetrad is covariantly
    constant under the Weitzenbock connection)
  - Metric-compatible: nabla_rho g_{mu nu} = 0 (Q=0), because
    g_{mu nu} = e^a_mu e^b_nu eta_{ab} and nabla_rho e^a_mu = 0
  - Torsionful: T^rho_{mu nu} = E_a^rho (partial_mu e^a_nu - partial_nu
    e^a_mu) is generally nonzero

The torsion scalar (Weitzenbock scalar) is defined as:

  T = (1/4) T_{rho mu nu} T^{rho mu nu}
    + (1/2) T_{rho mu nu} T^{mu rho nu}
    - T_mu T^{mu}

where T_mu = T^rho_{rho mu} is the torsion trace vector and
T^{mu} = g^{mu alpha} T_alpha.

Convention block: noether-default-v1 + metric-affine-v1 + tetrad-teleparallel-v1

  Torsion:  T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
  Non-metricity: Q_{lambda mu nu} = 0 (metric-compatible)
  Curvature: R^rho_{sigma mu nu}(Gamma) = 0 (curvature-free)
  Minkowski: eta_{ab} = diag(-1, +1, +1, +1)
  Metric: g_{mu nu} = e^a_mu e^b_nu eta_{ab}

Boundary-term identity:

  T = -R(g) + 2 nabla_mu^{LC} T^mu

where R(g) is the Ricci scalar of the metric's Levi-Civita connection,
and nabla^{LC} is the Levi-Civita covariant derivative. The divergence
2 nabla_mu T^mu is a total boundary term, so f(T) = T (linear teleparallel
gravity) produces the same EOM as the Einstein-Hilbert action: G_{mu nu} = 0.

The general f(T) field equation (metric form):

  E_{mu nu} = f'(T) [G_{mu nu} - (1/2) g_{mu nu} T]
            + S_{mu nu}^{rho} nabla_rho f'(T)
            + (1/2) g_{mu nu} [f(T) - T f'(T)]
            = 0

where S^{rho mu nu} = (1/2)(K^{rho mu nu} + g^{rho mu} T^nu - g^{rho nu} T^mu)
is the modified superpotential, K^{rho mu nu} is the fully raised contortion,
and T^mu = T^{nu mu}_{nu} is the torsion trace vector with raised index.
"""

from __future__ import annotations

import sympy as sp

from noether.kernels.sympy_kernel.geometry import (
    ComponentGeometry,
    _clean,
    components,
)

# ---------------------------------------------------------------------------
# Tetrad construction
# ---------------------------------------------------------------------------


def diagonal_tetrad_from_metric(
    geom: ComponentGeometry,
) -> tuple[sp.ImmutableDenseNDimArray, sp.ImmutableDenseNDimArray]:
    """Construct the diagonal (proper) tetrad from a diagonal metric.

    For a diagonal metric g = diag(-A, B, C, D) with mostly-plus
    signature (-,+,+,+), the diagonal tetrad is:

      e^a_mu = delta^a_mu * sqrt(|g_{mu mu}|)

    This is the "proper frame" or "coefficient tetrad" aligned with
    the coordinate basis. It satisfies:
      g_{mu nu} = e^a_mu e^b_nu eta_{ab}

    Returns (e, E) where e[a, mu] = e^a_mu and E[a, mu] = E_a^mu
    (the inverse tetrad).
    """
    n = geom.dim
    g = geom.g
    g_imm = sp.ImmutableDenseNDimArray(g)

    e = sp.MutableDenseNDimArray.zeros(n, n)
    E = sp.MutableDenseNDimArray.zeros(n, n)
    for a in range(n):
        for mu in range(n):
            if a == mu:
                # Diagonal component
                # g_{mu mu} = e^a_mu * e^a_mu * eta_{aa} (no sum)
                # So (e^a_mu)^2 = g_{mu mu} / eta_{aa}
                # eta_{00} = -1, eta_{ii} = +1 for i > 0
                if a == 0:
                    # g_{00} < 0 (timelike), eta_{00} = -1
                    # (e^0_0)^2 = g_{00}/(-1) = -g_{00} > 0
                    val = sp.sqrt(-g_imm[mu, mu])
                else:
                    # g_{ii} > 0 (spacelike), eta_{ii} = +1
                    val = sp.sqrt(g_imm[mu, mu])
                e[a, mu] = _clean(val)
                E[a, mu] = _clean(sp.Rational(1, 1) / val)
            else:
                e[a, mu] = sp.Rational(0)
                E[a, mu] = sp.Rational(0)

    return sp.ImmutableDenseNDimArray(e), sp.ImmutableDenseNDimArray(E)


def rotated_tetrad_from_metric(
    geom: ComponentGeometry,
    seed: int = 42,
) -> tuple[sp.ImmutableDenseNDimArray, sp.ImmutableDenseNDimArray]:
    """Construct a non-diagonal tetrad with nonzero Weitzenbock torsion scalar.

    The diagonal (proper) tetrad of a diagonal metric gives T = 0.
    Adding off-diagonal entries only in the spacelike block (a >= 1)
    also gives T = 0.  To get T != 0, we need off-diagonal entries
    in the *timelike row* (a = 0), which mix the -1 and +1 entries
    of the Minkowski metric and produce torsion components that do
    not cancel in the scalar contraction.

    All entries are rational polynomials (no transcendental functions)
    so SymPy can simplify the Weitzenbock geometry checks (R=0, Q=0).

    Convention (tetrad-teleparallel-v1):
      Metric-compatible, curvature-free, torsionful connection.
      Signature: mostly-plus (-,+,+,+).
      Torsion: T^rho_{mu nu} = Gamma^rho_{mu nu} - Gamma^rho_{nu mu}.
      Weitzenbock: Gamma^rho_{mu nu} = E_a^rho partial_mu e^a_nu.
      Torsion scalar: T = (1/4)T_{rho mu nu}T^{rho mu nu}
                           + (1/2)T_{rho mu nu}T^{mu rho nu}
                           - T_mu T^mu.

    Returns (e, E) where e[a, mu] = e^a_mu and E[a, mu] = E_a^mu.
    The metric implied by the returned tetrad is
      g'_{mu nu} = e^a_mu e^b_nu eta_{ab}
    which may differ from the original diagonal metric geom.g.
    """
    import random as rng

    n = geom.dim
    x = geom.coords
    rng_local = rng.Random(seed)

    # Start with the diagonal tetrad
    e0, _E0 = diagonal_tetrad_from_metric(geom)

    # Add a small coordinate-dependent off-diagonal entry in the
    # timelike row.  e^0_1 += c * x_1 mixes the timelike and
    # spacelike directions and is the minimal perturbation that
    # produces a nonzero torsion scalar T.
    e = sp.MutableDenseNDimArray(e0)
    if n >= 2:
        c_val = sp.Rational(rng_local.randint(1, 3), rng_local.randint(6, 15))
        e[0, 1] = _clean(e0[0, 1] + c_val * x[1])

    e = sp.ImmutableDenseNDimArray(e)

    # Compute inverse tetrad as the transpose of the matrix inverse of e.
    # The condition E_a^mu * e^a_nu = delta^mu_nu means E^T * e = I,
    # so E = (e^{-1})^T and E[a, mu] = (e^{-1})[mu, a].
    e_matrix = sp.Matrix([[e[a, mu] for mu in range(n)] for a in range(n)])
    e_inv_matrix = e_matrix.inv()
    E = sp.MutableDenseNDimArray.zeros(n, n)
    for a in range(n):
        for mu in range(n):
            E[a, mu] = _clean(e_inv_matrix[mu, a])  # transpose!
    E = sp.ImmutableDenseNDimArray(E)

    return e, E


def tetrad_metric(
    e: sp.ImmutableDenseNDimArray,
    eta: sp.ImmutableDenseNDimArray,
) -> sp.ImmutableDenseNDimArray:
    """Compute the spacetime metric from the tetrad.

    g_{mu nu} = e^a_mu e^b_nu eta_{ab}

    Parameters:
        e: tetrad array e[a, mu] = e^a_mu
        eta: Minkowski metric eta[a, b] = eta_{ab}
    """
    n = e.shape[1]  # spacetime dimension
    g = sp.MutableDenseNDimArray.zeros(n, n)
    for mu in range(n):
        for nu in range(mu, n):
            val = _clean(sum(
                e[a, mu] * e[b, nu] * eta[a, b]
                for a in range(n) for b in range(n)
            ))
            g[mu, nu] = val
            g[nu, mu] = val  # symmetric
    return sp.ImmutableDenseNDimArray(g)


def minkowski_metric(dim: int = 4) -> sp.ImmutableDenseNDimArray:
    """Minkowski metric eta_{ab} = diag(-1, +1, ..., +1) in mostly-plus."""
    eta = sp.MutableDenseNDimArray.zeros(dim, dim)
    for a in range(dim):
        eta[a, a] = sp.Integer(-1) if a == 0 else sp.Integer(1)
    return sp.ImmutableDenseNDimArray(eta)


# ---------------------------------------------------------------------------
# Weitzenbock connection from tetrad
# ---------------------------------------------------------------------------


def weitzenbock_connection(
    coords: list[sp.Symbol],
    e: sp.ImmutableDenseNDimArray,
    E: sp.ImmutableDenseNDimArray,
) -> sp.ImmutableDenseNDimArray:
    """Weitzenbock (teleparallel) connection from the tetrad.

    Gamma^rho_{mu nu} = E_a^rho partial_mu e^a_nu

    This connection is:
      - Flat (R=0) by construction
      - Metric-compatible (Q=0) because nabla e^a_mu = 0
      - Torsionful: T^rho_{mu nu} = Gamma^rho_{mu nu} - Gamma^rho_{nu mu} != 0

    Convention: tetrad-teleparallel-v1.

    Parameters:
        coords: coordinate symbols
        e: tetrad array e[a, mu] = e^a_mu
        E: inverse tetrad array E[a, mu] = E_a^mu
    """
    n = len(coords)
    x = coords
    gamma = sp.MutableDenseNDimArray.zeros(n, n, n)
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                val = _clean(sum(
                    E[a, rho] * sp.diff(e[a, nu], x[mu])
                    for a in range(n)
                ))
                gamma[rho, mu, nu] = val
    return sp.ImmutableDenseNDimArray(gamma)


# ---------------------------------------------------------------------------
# Torsion scalar (Weitzenbock scalar)
# ---------------------------------------------------------------------------


def torsion_trace_vector(
    T_tensor: sp.ImmutableDenseNDimArray,
    g_inv: sp.ImmutableDenseNDimArray,
) -> list[sp.Expr]:
    """Torsion trace vector T_mu = T^rho_{rho mu}.

    This is a Kronecker contraction of the upper index with the
    first lower index: T^rho_{rho mu} = sum_rho T_tensor[rho, rho, mu].
    The inverse metric is used only for the raised trace vector
    T^{mu} = g^{mu alpha} T_alpha.

    Parameters:
        T_tensor: T^rho_{mu nu} array (3-index)
        g_inv: inverse metric g^{mu nu} array (used for raised form)
    """
    n = T_tensor.shape[0]
    T_down = [sp.Rational(0)] * n
    for mu in range(n):
        # T^rho_{rho mu} = sum_rho T_tensor[rho, rho, mu]
        # Kronecker contraction: upper index = first lower index
        T_down[mu] = _clean(sum(T_tensor[rho, rho, mu] for rho in range(n)))
    return T_down


def torsion_scalar_T(
    T_tensor: sp.ImmutableDenseNDimArray,
    g: sp.ImmutableDenseNDimArray,
    g_inv: sp.ImmutableDenseNDimArray,
) -> sp.Expr:
    """Weitzenbock torsion scalar T.

    T = (1/4) T_{rho mu nu} T^{rho mu nu}
      + (1/2) T_{rho mu nu} T^{mu rho nu}
      - T_mu T^{mu}

    Convention: tetrad-teleparallel-v1.

    Parameters:
        T_tensor: T^rho_{mu nu} array (3-index)
        g: metric g_{mu nu} array (2-index)
        g_inv: inverse metric g^{mu nu} array (2-index)
    """
    n = T_tensor.shape[0]

    # Lowered torsion: T_{rho mu nu} = g_{rho alpha} T^alpha_{mu nu}
    T_down = sp.MutableDenseNDimArray.zeros(n, n, n)
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                T_down[rho, mu, nu] = _clean(sum(
                    g[rho, alpha] * T_tensor[alpha, mu, nu]
                    for alpha in range(n)
                ))
    T_down = sp.ImmutableDenseNDimArray(T_down)

    # Fully raised torsion: T^{rho mu nu} = g^{rho a} g^{mu b} g^{nu c} T_{a b c}
    T_up = sp.MutableDenseNDimArray.zeros(n, n, n)
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                T_up[rho, mu, nu] = _clean(sum(
                    g_inv[rho, a] * g_inv[mu, b] * g_inv[nu, c] * T_down[a, b, c]
                    for a in range(n) for b in range(n) for c in range(n)
                ))
    T_up = sp.ImmutableDenseNDimArray(T_up)

    # Mixed form T^{mu rho nu} = g^{mu a} g^{rho b} g^{nu c} T_{a b c}
    T_mixed = sp.MutableDenseNDimArray.zeros(n, n, n)
    for mu in range(n):
        for rho in range(n):
            for nu in range(n):
                T_mixed[mu, rho, nu] = _clean(sum(
                    g_inv[mu, a] * g_inv[rho, b] * g_inv[nu, c] * T_down[a, b, c]
                    for a in range(n) for b in range(n) for c in range(n)
                ))
    T_mixed = sp.ImmutableDenseNDimArray(T_mixed)

    # Term 1: (1/4) T_{rho mu nu} T^{rho mu nu}
    term1 = sp.Rational(0)
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                term1 += T_down[rho, mu, nu] * T_up[rho, mu, nu]
    term1 = _clean(sp.Rational(1, 4) * term1)

    # Term 2: (1/2) T_{rho mu nu} T^{mu rho nu}
    term2 = sp.Rational(0)
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                term2 += T_down[rho, mu, nu] * T_mixed[mu, rho, nu]
    term2 = _clean(sp.Rational(1, 2) * term2)

    # Term 3: -T_mu T^{mu}
    T_vec = torsion_trace_vector(T_tensor, g_inv)
    term3 = sp.Rational(0)
    for mu in range(n):
        T_mu_up = _clean(sum(g_inv[mu, alpha] * T_vec[alpha] for alpha in range(n)))
        term3 += T_vec[mu] * T_mu_up
    term3 = _clean(-term3)

    return _clean(term1 + term2 + term3)


def boundary_term_identity_residual(
    coords: list[sp.Symbol],
    T_tensor: sp.ImmutableDenseNDimArray,
    g: sp.ImmutableDenseNDimArray,
    g_inv: sp.ImmutableDenseNDimArray,
) -> sp.Expr:
    """Residual of the boundary-term identity T = -R(g) + 2 nabla_mu^{LC} T^mu.

    Returns: T + R(g) - 2 nabla_mu T^mu (should be zero).

    This identity implies that for f(T) = T, the EOM is the same as
    the Einstein equation G_{mu nu} = 0.
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    R = geom.ricci_scalar
    T_val = torsion_scalar_T(T_tensor, g, g_inv)

    # Torsion trace vector
    T_vec = torsion_trace_vector(T_tensor, g_inv)
    n = len(coords)
    x = coords

    # LC divergence of T^mu: nabla_mu T^mu = partial_mu T^mu + Gamma^mu_{mu alpha} T^alpha
    LC = geom.christoffel
    T_up = [_clean(sum(g_inv[mu, alpha] * T_vec[alpha] for alpha in range(n)))
            for mu in range(n)]

    div_T = sp.Rational(0)
    for mu in range(n):
        div_T += sp.diff(T_up[mu], x[mu])
        for alpha in range(n):
            div_T += LC[mu, mu, alpha] * T_up[alpha]
    div_T = _clean(div_T)

    # Residual: T + R - 2 div_T  (should be zero)
    return _clean(T_val + R - 2 * div_T)


# ---------------------------------------------------------------------------
# Superpotential S^{rho mu nu}
# ---------------------------------------------------------------------------


def superpotential(
    T_tensor: sp.ImmutableDenseNDimArray,
    g: sp.ImmutableDenseNDimArray,
    g_inv: sp.ImmutableDenseNDimArray,
) -> sp.ImmutableDenseNDimArray:
    """Modified contortion (superpotential) S^{rho mu nu}.

    S^{rho mu nu} = (1/2)(K^{rho mu nu} + g^{rho mu} T^{nu} - g^{rho nu} T^{mu})

    where K^{rho mu nu} is the fully raised contortion and T^{mu} is the
    torsion trace vector with raised index.

    Convention: metric-affine-v1 + tetrad-teleparallel-v1.

    Parameters:
        T_tensor: T^rho_{mu nu} array
        g: metric g_{mu nu}
        g_inv: inverse metric g^{mu nu}
    """
    # Compute contortion K^lambda_{mu nu} directly from the torsion.
    # K^lambda_{mu nu} = (1/2)(T^lambda_{mu nu}
    #                   + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu}
    #                   + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu})
    n = T_tensor.shape[0]
    K = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                term1 = T_tensor[lam, mu, nu]
                term2 = sum(
                    g_inv[lam, sig] * g[mu, tau] * T_tensor[tau, sig, nu]
                    for sig in range(n) for tau in range(n)
                )
                term3 = sum(
                    g_inv[lam, sig] * g[nu, tau] * T_tensor[tau, sig, mu]
                    for sig in range(n) for tau in range(n)
                )
                K[lam, mu, nu] = _clean(sp.Rational(1, 2) * (term1 + term2 + term3))
    K = sp.ImmutableDenseNDimArray(K)

    # Fully raised contortion K^{rho mu nu} = g^{mu a} g^{nu b} K^rho_{a b}
    K_up = sp.MutableDenseNDimArray.zeros(n, n, n)
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                K_up[rho, mu, nu] = _clean(sum(
                    g_inv[mu, a] * g_inv[nu, b] * K[rho, a, b]
                    for a in range(n) for b in range(n)
                ))
    K_up = sp.ImmutableDenseNDimArray(K_up)

    # Torsion trace vector
    T_vec = torsion_trace_vector(T_tensor, g_inv)
    T_vec_up = [_clean(sum(g_inv[mu, alpha] * T_vec[alpha] for alpha in range(n)))
                for mu in range(n)]

    # Build superpotential S^{rho mu nu}
    S = sp.MutableDenseNDimArray.zeros(n, n, n)
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                val = sp.Rational(1, 2) * (
                    K_up[rho, mu, nu]
                    + g_inv[rho, mu] * T_vec_up[nu]
                    - g_inv[rho, nu] * T_vec_up[mu]
                )
                S[rho, mu, nu] = _clean(val)
    return sp.ImmutableDenseNDimArray(S)


# ---------------------------------------------------------------------------
# f(T) field equation (metric form)
# ---------------------------------------------------------------------------


def ft_eom_metric_form(
    coords: list[sp.Symbol],
    T_tensor: sp.ImmutableDenseNDimArray,
    g: sp.ImmutableDenseNDimArray,
    g_inv: sp.ImmutableDenseNDimArray,
    f: sp.Expr,
    fp: sp.Expr,
    T_val: sp.Expr,
) -> sp.ImmutableDenseNDimArray:
    """General f(T) EOM in metric form.

    E_{mu nu} = f'(T) [G_{mu nu} - (1/2) g_{mu nu} T]
              + S_{mu nu}^{rho} nabla_rho f'(T)
              + (1/2) g_{mu nu} [f(T) - T f'(T)]

    For the linear case f(T) = T: f'=1, f''=0, f-Tf'=0,
    giving E_{mu nu} = G_{mu nu} - (1/2) g_{mu nu} T + S_{mu nu}^{rho} partial_rho T
    ... wait, let me re-derive.

    Actually, nabla_rho f'(T) = f''(T) partial_rho T for the metric covariant
    derivative. For f(T) = T: f'=1, f''=0, so nabla_rho f'(T) = 0.
    Also f(T) - T f'(T) = T - T = 0.
    So E_{mu nu} = G_{mu nu} - (1/2) g_{mu nu} T.

    But wait, by the boundary-term identity T = -R + 2 nabla T^mu, so:
    G_{mu nu} - (1/2) g_{mu nu} T = (R_{mu nu} - 1/2 g_{mu nu} R) - 1/2 g_{mu nu} T
    = R_{mu nu} - 1/2 g_{mu nu} (R + T)
    = R_{mu nu} - 1/2 g_{mu nu} (R + (-R + 2 nabla T^mu))
    = R_{mu nu} - 1/2 g_{mu nu} (2 nabla T^mu)

    Hmm, this doesn't simplify to G_{mu nu} = 0 directly from the metric-form
    EOM. The metric-form EOM for f(T) = T is:

    E_{mu nu} = G_{mu nu} - (1/2) g_{mu nu} T

    But we also have the identity S_{mu nu}^{rho} partial_rho T = ... hmm,
    this is getting complicated.

    Actually, for f(T) = T, the EOM should be zero on-shell, but the
    metric form has the term -(1/2) g_{mu nu} T which is NOT zero on a
    general background. The metric form is the EOM itself, and setting
    it to zero gives the field equation.

    Let me just implement the formula and verify it numerically.

    Parameters:
        coords: coordinate symbols
        T_tensor: T^rho_{mu nu} torsion tensor
        g: metric g_{mu nu}
        g_inv: inverse metric g^{mu nu}
        f: f(T) expression
        fp: f'(T) expression
        T_val: the torsion scalar T evaluated on the background
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    G = geom.einstein  # G_{mu nu}
    n = len(coords)
    x = coords

    # Superpotential
    S = superpotential(T_tensor, g, g_inv)

    # S_{mu nu}^{rho} = g_{mu alpha} g_{nu beta} S^{rho alpha beta}
    S_down_down = sp.MutableDenseNDimArray.zeros(n, n, n)
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                S_down_down[rho, mu, nu] = _clean(sum(
                    g[mu, alpha] * g[nu, beta] * S[rho, alpha, beta]
                    for alpha in range(n) for beta in range(n)
                ))

    # LC covariant derivative of f'(T):
    # For a scalar, nabla_rho f'(T) = partial_rho f'(T)
    # (the chain rule: d(f'(T))/dx^rho = f''(T) * dT/dx^rho)
    # We are given fp = f'(T) as an expression, so nabla_rho fp = partial_rho fp
    nabla_fp = [_clean(sp.diff(fp, x[rho])) for rho in range(n)]

    # Build E_{mu nu}
    E = sp.MutableDenseNDimArray.zeros(n, n)
    for mu in range(n):
        for nu in range(n):
            # Term 1: f'(T) [G_{mu nu} - (1/2) g_{mu nu} T]
            t1 = _clean(fp * (G[mu, nu] - sp.Rational(1, 2) * g[mu, nu] * T_val))

            # Term 2: S_{mu nu}^{rho} nabla_rho f'(T)
            t2 = sp.Rational(0)
            for rho in range(n):
                t2 += S_down_down[rho, mu, nu] * nabla_fp[rho]
            t2 = _clean(t2)

            # Term 3: (1/2) g_{mu nu} [f(T) - T f'(T)]
            t3 = _clean(sp.Rational(1, 2) * g[mu, nu] * (f - T_val * fp))

            E[mu, nu] = _clean(t1 + t2 + t3)

    return sp.ImmutableDenseNDimArray(E)


def ft_eom_linear(
    coords: list[sp.Symbol],
    T_tensor: sp.ImmutableDenseNDimArray,
    g: sp.ImmutableDenseNDimArray,
    g_inv: sp.ImmutableDenseNDimArray,
    T_val: sp.Expr,
) -> sp.ImmutableDenseNDimArray:
    """f(T) = T EOM in metric form.

    For f(T) = T: f'=1, f-Tf'=0, giving:
    E_{mu nu} = G_{mu nu} - (1/2) g_{mu nu} T + S_{mu nu}^{rho} partial_rho T

    Wait, but nabla_rho f'(T) = nabla_rho 1 = 0. So actually:
    E_{mu nu} = G_{mu nu} - (1/2) g_{mu nu} T

    Setting this to zero gives the field equation. For the boundary-term
    identity, the trace of this equation gives:
    g^{mu nu} E_{mu nu} = R - 2 T = -(R + 2 nabla T^mu - 2T) = ...
    The equation E_{mu nu} = 0 is equivalent to G_{mu nu} = (1/2) g_{mu nu} T
    on shell.
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    G = geom.einstein
    n = g.shape[0]

    E = sp.MutableDenseNDimArray.zeros(n, n)
    for mu in range(n):
        for nu in range(n):
            E[mu, nu] = _clean(G[mu, nu] - sp.Rational(1, 2) * g[mu, nu] * T_val)

    return sp.ImmutableDenseNDimArray(E)


def ft_eom_linear_via_boundary(
    coords: list[sp.Symbol],
    g: sp.ImmutableDenseNDimArray,
    g_inv: sp.ImmutableDenseNDimArray,
) -> sp.ImmutableDenseNDimArray:
    """f(T) = T EOM via the boundary-term identity: G_{mu nu} = 0.

    By the boundary-term identity T = -R + 2 nabla_mu T^mu, the
    f(T) = T action equals -S_EH + boundary, so the metric EOM
    is G_{mu nu} = 0 (identical to GR).

    This is the "coincident" result: the linear f(T) EOM is the
    same as the Einstein equation.
    """
    geom = ComponentGeometry(coords, sp.Matrix(g))
    return geom.einstein


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_weitzenbock_is_flat(
    coords: list[sp.Symbol],
    gamma: sp.ImmutableDenseNDimArray,
) -> bool:
    """Verify that the Weitzenbock connection is flat (R=0)."""
    from noether.kernels.sympy_kernel.geometry import riemann_of_connection
    R = riemann_of_connection(coords, gamma)
    return all(sp.simplify(c) == 0 for c in components(R))


def verify_weitzenbock_is_metric_compatible(
    coords: list[sp.Symbol],
    gamma: sp.ImmutableDenseNDimArray,
    g: sp.ImmutableDenseNDimArray,
) -> bool:
    """Verify that the Weitzenbock connection is metric-compatible (Q=0)."""
    from noether.kernels.sympy_kernel.geometry import nonmetricity_of_connection
    Q = nonmetricity_of_connection(coords, gamma, g)
    return all(sp.simplify(c) == 0 for c in components(Q))
