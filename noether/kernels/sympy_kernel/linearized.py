"""Linearized curvature around Minkowski and the eval 3s spectrum checks.

The linearized geometry is computed from the SAME definitional formulas as
ComponentGeometry, with arithmetic truncated at O(eps^2): for g = eta + eps h
the inverse is eta - eps h (indices moved with eta) and the GammaGamma terms
of Riemann are O(eps^2) and drop. Nothing is taken from a textbook; every
coefficient below was pinned by kernel computation (the alternatives fail
with nonzero residue; see spectrum_checks):

  R^(1)[chi eta]            = -3 box chi
  eta^{mu nu} G^(1)[h]      = -R^(1)[h]
  mixing shift              h = hbar - (F1/F0) chi eta   (c = +F1/F0 FAILS)
  effective scalar kinetic  K_chi = (F0 + 3 F1^2) / F0

with F0 = F(phi_0), F1 = F'(phi_0) for the eval-3 theory
S = int sqrt(-g) [F(phi) R - 1/2 (d phi)^2 - V(phi)] around g = eta,
phi = phi_0, V(phi_0) = V'(phi_0) = 0. The linear equations of motion are the
eps-derivative of the FULL cadabra-verified eval-3 equations (anchor checks
recompute that on concrete component fields, exact ComponentGeometry, no
truncation).

Conventions: noether-default-v1 (4 dimensions, mostly plus).
"""

import itertools

import sympy as sp

N_DIM = 4
_T, _X, _Y, _Z = sp.symbols("t x y z")
COORDS = [_T, _X, _Y, _Z]
ETA = sp.diag(-1, 1, 1, 1)


def lin_christoffel(h: sp.Matrix) -> list[list[list[sp.Expr]]]:
    """Gamma^(1)a_bc = 1/2 eta^{ad}(d_b h_dc + d_c h_db - d_d h_bc)."""
    n = N_DIM
    out = [[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                out[a][b][c] = sp.Rational(1, 2) * sum(
                    ETA[a, d]
                    * (
                        sp.diff(h[d, c], COORDS[b])
                        + sp.diff(h[d, b], COORDS[c])
                        - sp.diff(h[b, c], COORDS[d])
                    )
                    for d in range(n)
                )
    return out


def lin_ricci(h: sp.Matrix) -> sp.Matrix:
    """R^(1)_{sigma nu} = d_lam Gamma^(1)lam_{nu sigma} - d_nu Gamma^(1)lam_{lam sigma}."""
    n = N_DIM
    gamma = lin_christoffel(h)
    return sp.Matrix(
        n,
        n,
        lambda s, nu: sum(
            sp.diff(gamma[lam][nu][s], COORDS[lam]) - sp.diff(gamma[lam][lam][s], COORDS[nu])
            for lam in range(n)
        ),
    )


def lin_ricci_scalar(h: sp.Matrix) -> sp.Expr:
    """R^(1) = eta^{mu nu} R^(1)_{mu nu} (the g^(1) R^(0) piece vanishes)."""
    ric = lin_ricci(h)
    n = N_DIM
    return sum(ETA[a, b] * ric[a, b] for a in range(n) for b in range(n))


def lin_einstein(h: sp.Matrix) -> sp.Matrix:
    n = N_DIM
    ric = lin_ricci(h)
    scal = sum(ETA[a, b] * ric[a, b] for a in range(n) for b in range(n))
    return sp.Matrix(n, n, lambda a, b: sp.expand(ric[a, b] - sp.Rational(1, 2) * ETA[a, b] * scal))


def box_flat(f: sp.Expr) -> sp.Expr:
    n = N_DIM
    return sum(ETA[a, b] * sp.diff(f, COORDS[a], COORDS[b]) for a in range(n) for b in range(n))


def _all_zero_matrix(m: sp.Matrix) -> bool:
    return all(sp.expand(comp) == 0 for comp in m)


# --------------------------------------------------------------------- checks


def _check_conformal_ricci() -> tuple[bool, str]:
    chi = sp.Function("chi")(*COORDS)
    residue = sp.expand(lin_ricci_scalar(ETA * chi) + 3 * box_flat(chi))
    ok = residue == 0
    return ok, "R1[chi eta] == -3 box chi" if ok else f"residue {residue}"


def _check_trace_identity() -> tuple[bool, str]:
    n = N_DIM
    h = sp.Matrix(n, n, lambda i, j: sp.Function(f"hb{min(i, j)}{max(i, j)}")(*COORDS))
    tr = sum(ETA[a, b] * lin_einstein(h)[a, b] for a in range(n) for b in range(n))
    ok = sp.expand(tr + lin_ricci_scalar(h)) == 0
    return ok, "eta^{mu nu} G1[h] == -R1[h]" if ok else "trace identity failed"


def _check_mixing_shift() -> tuple[bool, str]:
    """F0 G1[c chi eta] + (eta box - dd)(F1 chi) == 0 for c = -F1/F0 only."""
    n = N_DIM
    F0, F1 = sp.symbols("F0 F1", positive=True)
    chi = sp.Function("chi")(*COORDS)
    g1 = lin_einstein(ETA * chi)

    def residue(c):
        return [
            sp.expand(
                F0 * c * g1[i, j]
                + ETA[i, j] * box_flat(F1 * chi)
                - sp.diff(F1 * chi, COORDS[i], COORDS[j])
            )
            for i in range(n)
            for j in range(n)
        ]

    good = all(r == 0 for r in residue(-F1 / F0))
    bad_cancels = all(r == 0 for r in residue(F1 / F0))
    if not good:
        return False, "c = -F1/F0 does not cancel the mixing"
    if bad_cancels:
        return False, "falsifier failed: c = +F1/F0 also cancels (check degenerate)"
    return True, "h = hbar - (F1/F0) chi eta removes the kinetic mixing; +F1/F0 does not"


def _check_scalar_diagonalization() -> tuple[bool, str]:
    """box chi + F1 R1[hbar + c chi eta] - V2 chi
    == K_chi box chi + F1 R1[hbar] - V2 chi, K_chi = (F0 + 3 F1^2)/F0."""
    n = N_DIM
    F0, F1, V2 = sp.symbols("F0 F1 V2", positive=True)
    chi = sp.Function("chi")(*COORDS)
    hbar = sp.Matrix(n, n, lambda i, j: sp.Function(f"hb{min(i, j)}{max(i, j)}")(*COORDS))
    c = -F1 / F0
    k_chi = (F0 + 3 * F1**2) / F0
    lhs = box_flat(chi) + F1 * lin_ricci_scalar(hbar + c * chi * ETA) - V2 * chi
    rhs = k_chi * box_flat(chi) + F1 * lin_ricci_scalar(hbar) - V2 * chi
    residue = sp.expand(sp.together(lhs - rhs))
    if residue != 0:
        return False, f"diagonalization residue {residue}"
    mixing_real = sp.simplify(k_chi - 1) != 0
    if not mixing_real:
        return False, "falsifier failed: K_chi == 1, no mixing to diagonalize"
    return True, "scalar sector diagonalizes with K_chi = (F0 + 3 F1^2)/F0"


def _check_tt_null_wave() -> tuple[bool, str]:
    """A TT plane wave solves G1 == 0 iff its momentum is null (massless)."""
    n = N_DIM
    w, k = sp.symbols("omega kvec", positive=True)
    amp = sp.Symbol("A")
    pol = sp.zeros(n, n)
    pol[1, 2] = amp
    pol[2, 1] = amp  # TT polarization for propagation along z
    null = sp.Matrix(n, n, lambda i, j: pol[i, j] * sp.cos(w * _T - w * _Z))
    off = sp.Matrix(n, n, lambda i, j: pol[i, j] * sp.cos(w * _T - k * _Z))
    if not _all_zero_matrix(lin_einstein(null)):
        return False, "TT null wave does not solve G1 == 0"
    if all(
        sp.simplify(lin_einstein(off)[i, j]) == 0 for i, j in itertools.product(range(n), repeat=2)
    ):
        return False, "falsifier failed: non-null wave also solves G1 == 0"
    return True, "G1[TT wave] == 0 iff the momentum is null (massless graviton)"


def _full_eom_anchor() -> tuple[tuple[bool, str], tuple[bool, str]]:
    """The assembled linear operators equal the eps-derivative of the FULL
    cadabra-verified eval-3 equations on concrete component fields (exact
    ComponentGeometry, no truncated arithmetic)."""
    from noether.kernels.sympy_kernel.geometry import ComponentGeometry

    n = N_DIM
    eps = sp.Symbol("epsilon")
    hc = sp.zeros(n, n)
    hc[0, 0] = sp.Rational(1, 3) * _X
    hc[1, 1] = sp.Rational(1, 4) * _T
    hc[1, 2] = sp.Rational(1, 5) * _Z
    hc[2, 1] = hc[1, 2]
    hc[3, 3] = sp.Rational(1, 2) * _Y
    chi_c = sp.Rational(1, 7) * (_X * _T + _Y**2)

    phi0, s = sp.symbols("phi0 s")
    F, V = sp.Function("F"), sp.Function("V")
    phi = phi0 + eps * chi_c
    g = sp.Matrix(n, n, lambda i, j: ETA[i, j] + eps * hc[i, j])
    geom = ComponentGeometry(COORDS, g)
    ginv, gamma = geom.g_inv, geom.christoffel

    f_field = F(phi)
    df = [sp.diff(f_field, c) for c in COORDS]
    hess_f = sp.Matrix(
        n,
        n,
        lambda a, b: (
            sp.diff(f_field, COORDS[a], COORDS[b])
            - sum(gamma[lam, a, b] * df[lam] for lam in range(n))
        ),
    )
    box_f = sum(ginv[a, b] * hess_f[a, b] for a in range(n) for b in range(n))
    dphi = [sp.diff(phi, c) for c in COORDS]
    kin = sum(ginv[a, b] * dphi[a] * dphi[b] for a in range(n) for b in range(n))
    einst = geom.einstein

    F0s, F1s, V2s = sp.symbols("F0 F1 V2")
    rules = {
        F(phi0): F0s,
        sp.Derivative(F(phi0), phi0): F1s,
        V(phi0): 0,
        sp.Derivative(V(phi0), phi0): 0,
        sp.Derivative(V(phi0), (phi0, 2)): V2s,
    }

    def lin(expr):
        e = sp.diff(expr, eps).subs(eps, 0)
        return sp.expand(e.doit(deep=True).subs(rules))

    ok_metric = True
    for a in range(n):
        for b in range(n):
            full_comp = (
                f_field * einst[a, b]
                + g[a, b] * box_f
                - hess_f[a, b]
                - sp.Rational(1, 2) * dphi[a] * dphi[b]
                + sp.Rational(1, 4) * g[a, b] * kin
                + sp.Rational(1, 2) * g[a, b] * V(phi)
            )
            target = (
                F0s * sp.diff(einst[a, b], eps).subs(eps, 0)
                + ETA[a, b] * box_flat(F1s * chi_c)
                - sp.diff(F1s * chi_c, COORDS[a], COORDS[b])
            )
            if sp.simplify(lin(full_comp) - sp.expand(target)) != 0:
                ok_metric = False
    metric_detail = (
        "linear metric operator == d/deps of the full eval-3 metric EOM"
        if ok_metric
        else "metric anchor mismatch"
    )

    box_phi = sum(
        ginv[a, b]
        * (
            sp.diff(phi, COORDS[a], COORDS[b])
            - sum(gamma[lam, a, b] * dphi[lam] for lam in range(n))
        )
        for a in range(n)
        for b in range(n)
    )
    full_scalar = (
        box_phi + sp.diff(F(s), s).subs(s, phi) * geom.ricci_scalar - sp.diff(V(s), s).subs(s, phi)
    )
    r1_concrete = sp.diff(geom.ricci_scalar, eps).subs(eps, 0)
    target_scalar = box_flat(chi_c) + F1s * r1_concrete - V2s * chi_c
    ok_scalar = sp.simplify(lin(full_scalar) - sp.expand(target_scalar)) == 0
    scalar_detail = (
        "linear scalar operator == d/deps of the full eval-3 scalar EOM"
        if ok_scalar
        else "scalar anchor mismatch"
    )
    return (ok_metric, metric_detail), (ok_scalar, scalar_detail)


def spectrum_checks() -> dict[str, tuple[bool, str]]:
    metric_anchor, scalar_anchor = _full_eom_anchor()
    return {
        "conformal-ricci": _check_conformal_ricci(),
        "trace-identity": _check_trace_identity(),
        "mixing-shift": _check_mixing_shift(),
        "scalar-diagonalization": _check_scalar_diagonalization(),
        "tt-null-wave": _check_tt_null_wave(),
        "full-eom-anchor-metric": metric_anchor,
        "full-eom-anchor-scalar": scalar_anchor,
    }
