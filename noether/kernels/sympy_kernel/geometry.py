"""Component differential geometry under noether-default-v1 conventions.

Conventions (AGENTS.md section 5):
  Gamma^a_{bc} = 1/2 g^{ad} (d_b g_{dc} + d_c g_{db} - d_d g_{bc})
  R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
                        + Gamma^rho_{mu lam} Gamma^lam_{nu sigma}
                        - Gamma^rho_{nu lam} Gamma^lam_{mu sigma}
  R_{sigma nu} = R^lambda_{sigma lambda nu}
  G_{mu nu} = R_{mu nu} - 1/2 g_{mu nu} R
"""

import random
from functools import cached_property

import sympy as sp

Array = sp.ImmutableDenseNDimArray


def _clean(expr):
    """Fast exact normalization for intermediates.

    All geometry built from polynomial metrics is rational in the coordinates,
    where cancel() is canonical (zero iff numerator zero). Full simplify() is
    reserved for final zero-tests (see evaluator.all_zero) so hot loops stay
    polynomial-time. Correctness is unaffected: cancel never changes value.
    """
    return sp.cancel(sp.together(expr))


class ComponentGeometry:
    def __init__(self, coords: list[sp.Symbol], metric: sp.Matrix):
        if metric.shape != (len(coords), len(coords)):
            raise ValueError("metric shape does not match coordinate count")
        if sp.simplify(metric - metric.T) != sp.zeros(*metric.shape):
            raise ValueError("metric must be symmetric")
        self.coords = list(coords)
        self.dim = len(coords)
        self.g = sp.ImmutableMatrix(metric)

    @cached_property
    def g_inv(self) -> sp.ImmutableMatrix:
        return sp.ImmutableMatrix(self.g.inv().applyfunc(_clean))

    @cached_property
    def christoffel(self) -> Array:
        """Gamma[a][b][c] = Gamma^a_{bc}."""
        n, x, g, ginv = self.dim, self.coords, self.g, self.g_inv
        out = sp.MutableDenseNDimArray.zeros(n, n, n)
        for a in range(n):
            for b in range(n):
                for c in range(b, n):
                    val = sp.Rational(1, 2) * sum(
                        ginv[a, d]
                        * (sp.diff(g[d, c], x[b]) + sp.diff(g[d, b], x[c]) - sp.diff(g[b, c], x[d]))
                        for d in range(n)
                    )
                    val = _clean(val)
                    out[a, b, c] = val
                    out[a, c, b] = val  # symmetric in lower pair (Levi-Civita)
        return Array(out)

    @cached_property
    def riemann(self) -> Array:
        """R[rho][sigma][mu][nu] = R^rho_{sigma mu nu}."""
        n, x, Gm = self.dim, self.coords, self.christoffel
        out = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for rho in range(n):
            for sig in range(n):
                for mu in range(n):
                    for nu in range(mu + 1, n):
                        val = (
                            sp.diff(Gm[rho, nu, sig], x[mu])
                            - sp.diff(Gm[rho, mu, sig], x[nu])
                            + sum(
                                Gm[rho, mu, lam] * Gm[lam, nu, sig]
                                - Gm[rho, nu, lam] * Gm[lam, mu, sig]
                                for lam in range(n)
                            )
                        )
                        val = _clean(val)
                        out[rho, sig, mu, nu] = val
                        out[rho, sig, nu, mu] = -val  # antisymmetry in last pair
        return Array(out)

    @cached_property
    def ricci(self) -> Array:
        """R_{sigma nu} = R^lambda_{sigma lambda nu}."""
        n, Rm = self.dim, self.riemann
        out = sp.MutableDenseNDimArray.zeros(n, n)
        for sig in range(n):
            for nu in range(n):
                out[sig, nu] = _clean(sum(Rm[lam, sig, lam, nu] for lam in range(n)))
        return Array(out)

    @cached_property
    def ricci_scalar(self) -> sp.Expr:
        n = self.dim
        return _clean(sum(self.g_inv[a, b] * self.ricci[a, b] for a in range(n) for b in range(n)))

    @cached_property
    def einstein(self) -> Array:
        """G_{mu nu}, both indices down."""
        n, R = self.dim, self.ricci_scalar
        out = sp.MutableDenseNDimArray.zeros(n, n)
        for a in range(n):
            for b in range(n):
                out[a, b] = _clean(self.ricci[a, b] - sp.Rational(1, 2) * self.g[a, b] * R)
        return Array(out)

    @cached_property
    def riemann_down(self) -> Array:
        """R_{rho sigma mu nu}, all indices down."""
        n = self.dim
        out = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for s in range(n):
            for m in range(n):
                for nu in range(m + 1, n):
                    for r in range(n):
                        val = _clean(
                            sum(self.g[r, lam] * self.riemann[lam, s, m, nu] for lam in range(n))
                        )
                        out[r, s, m, nu] = val
                        out[r, s, nu, m] = -val
        return Array(out)

    @cached_property
    def _riemann_mixed(self) -> Array:
        """R_mu^{abc}: riemann_down with the last three axes raised."""
        arr = self.riemann_down
        for axis in (1, 2, 3):
            arr = self.raise_first_index(arr, axis)
        return arr

    @cached_property
    def gauss_bonnet_scalar(self) -> sp.Expr:
        """GB = R^2 - 4 R_{ab}R^{ab} + R_{abcd}R^{abcd}."""
        n = self.dim
        ric_up = self.raise_first_index(self.raise_first_index(self.ricci, 0), 1)
        ricric = sum(self.ricci[a, b] * ric_up[a, b] for a in range(n) for b in range(n))
        riem_up = self.raise_first_index(self._riemann_mixed, 0)
        riemriem = sum(self.riemann_down[idx] * riem_up[idx] for idx in _all_indices(n, 4))
        return _clean(self.ricci_scalar**2 - 4 * ricric + riemriem)

    @cached_property
    def gauss_bonnet(self) -> Array:
        """The Lanczos tensor H_{mu nu} (the Gauss-Bonnet field equation LHS):
        2( R R_{mu nu} - 2 R_{mu a}R^a_nu - 2 R^{ab} R_{mu a nu b}
           + R_mu^{abc} R_{nu abc} ) - 1/2 g_{mu nu} GB.
        Identically zero in dim 4 (Lovelock); divergence-free in any dim."""
        n = self.dim
        ric, scal = self.ricci, self.ricci_scalar
        ric_mixed = self.raise_first_index(ric, 0)  # R^a_b
        ric_up = self.raise_first_index(ric_mixed, 1)  # R^{ab}
        rdown, rmixed = self.riemann_down, self._riemann_mixed
        out = sp.MutableDenseNDimArray.zeros(n, n)
        # every component computed independently so the V1 symmetric check
        # genuinely exercises the pair symmetries of the Riemann contractions
        for m in range(n):
            for nu in range(n):
                a_term = sum(ric[m, a] * ric_mixed[a, nu] for a in range(n))
                b_term = sum(ric_up[a, b] * rdown[m, a, nu, b] for a in range(n) for b in range(n))
                c_term = sum(
                    rmixed[m, a, b, c] * rdown[nu, a, b, c]
                    for a in range(n)
                    for b in range(n)
                    for c in range(n)
                )
                out[m, nu] = _clean(
                    2 * (scal * ric[m, nu] - 2 * a_term - 2 * b_term + c_term)
                    - sp.Rational(1, 2) * self.g[m, nu] * self.gauss_bonnet_scalar
                )
        return Array(out)

    def covariant_derivative(self, arr, variances: list[str]):
        """nabla_a T: returns array with a new leading 'down' axis.

        `arr` is a scalar (rank 0) or an Array whose slots have the given
        variances ("up"/"down").
        """
        n, x, Gm = self.dim, self.coords, self.christoffel
        if not variances:
            return Array([_clean(sp.diff(arr, x[a])) for a in range(n)])
        rank = len(variances)
        shape = (n,) + (n,) * rank
        out = sp.MutableDenseNDimArray.zeros(*shape)
        for a in range(n):
            for idx in _all_indices(n, rank):
                val = sp.diff(arr[idx], x[a])
                for s, var in enumerate(variances):
                    for lam in range(n):
                        swapped = idx[:s] + (lam,) + idx[s + 1 :]
                        if var == "up":
                            val += Gm[idx[s], a, lam] * arr[swapped]
                        else:
                            val -= Gm[lam, a, idx[s]] * arr[swapped]
                out[(a, *idx)] = _clean(val)
        return Array(out)

    def raise_first_index(self, arr, axis: int):
        """Contract g^{ab} with the given 'down' axis, returning it as 'up'."""
        n = self.dim
        rank = len(arr.shape)
        out = sp.MutableDenseNDimArray.zeros(*arr.shape)
        for idx in _all_indices(n, rank):
            val = sum(
                self.g_inv[idx[axis], b] * arr[idx[:axis] + (b,) + idx[axis + 1 :]]
                for b in range(n)
            )
            out[idx] = _clean(val)
        return Array(out)

    def lower_index(self, arr, axis: int):
        n = self.dim
        rank = len(arr.shape)
        out = sp.MutableDenseNDimArray.zeros(*arr.shape)
        for idx in _all_indices(n, rank):
            val = sum(
                self.g[idx[axis], b] * arr[idx[:axis] + (b,) + idx[axis + 1 :]] for b in range(n)
            )
            out[idx] = _clean(val)
        return Array(out)


def _all_indices(n: int, rank: int):
    if rank == 0:
        yield ()
        return
    for first in range(n):
        for rest in _all_indices(n, rank - 1):
            yield (first, *rest)


def components(arr):
    """Iterate scalar components of an NDimArray (its own iterator yields
    subarrays, which silently breaks `all(c == 0 ...)` style checks)."""
    shape = getattr(arr, "shape", ())
    if not shape:
        yield arr
        return
    for idx in _all_indices(shape[0], len(shape)):
        yield arr[idx]


def two_sphere(radius: sp.Expr | None = None) -> ComponentGeometry:
    """Round 2-sphere; known results pin the convention signs in tests."""
    theta, phi = sp.symbols("theta phi", positive=True)
    a = radius if radius is not None else sp.Symbol("a", positive=True)
    g = sp.Matrix([[a**2, 0], [0, a**2 * sp.sin(theta) ** 2]])
    return ComponentGeometry([theta, phi], g)


def riemann_of_connection(coords: list[sp.Symbol], gamma) -> Array:
    """R^rho_{sigma mu nu} of a general affine connection gamma[a][b][c] =
    Gamma^a_{bc} (no symmetry assumed; torsion allowed). Same sign conventions
    as ComponentGeometry.riemann."""
    n, x = len(coords), coords
    out = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for rho in range(n):
        for sig in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    val = (
                        sp.diff(gamma[rho, nu, sig], x[mu])
                        - sp.diff(gamma[rho, mu, sig], x[nu])
                        + sum(
                            gamma[rho, mu, lam] * gamma[lam, nu, sig]
                            - gamma[rho, nu, lam] * gamma[lam, mu, sig]
                            for lam in range(n)
                        )
                    )
                    val = _clean(val)
                    out[rho, sig, mu, nu] = val
                    out[rho, sig, nu, mu] = -val
    return Array(out)


def ricci_of_connection(coords: list[sp.Symbol], gamma) -> Array:
    """R_{sigma nu} = R^lambda_{sigma lambda nu}; NOT symmetric in general."""
    n = len(coords)
    Rm = riemann_of_connection(coords, gamma)
    out = sp.MutableDenseNDimArray.zeros(n, n)
    for sig in range(n):
        for nu in range(n):
            out[sig, nu] = _clean(sum(Rm[lam, sig, lam, nu] for lam in range(n)))
    return Array(out)


def torsion_of_connection(gamma) -> Array:
    """T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}.

    Convention: noether-default-v1 (AGENTS.md section 5). The torsion tensor
    is antisymmetric in the lower pair: T^lambda_{mu nu} = -T^lambda_{nu mu}.
    """
    n = gamma.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                out[lam, mu, nu] = _clean(gamma[lam, mu, nu] - gamma[lam, nu, mu])
    return Array(out)


def torsion_trace_vector(gamma, g_inv=None) -> Array:
    """T_mu = T^lambda_{lambda mu} (the torsion trace vector).

    Returns a 1-form (all indices down). If g_inv is given, the trace is
    computed as g^{lambda kappa} T_{kappa mu ...}; otherwise T^lambda_{lambda mu}
    is taken from the upper-lower form directly.
    """
    T = torsion_of_connection(gamma)
    n = T.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n)
    for mu in range(n):
        out[mu] = _clean(sum(T[lam, lam, mu] for lam in range(n)))
    return Array(out)


def torsion_axial_vector(gamma, geom: ComponentGeometry) -> Array:
    """A^rho = epsilon^{rho sigma kappa lambda} T_{sigma kappa lambda} / 6.

    The axial vector is (1/6) times the Levi-Civita dual of the totally
    antisymmetric part of T_{lambda mu nu} (all indices lowered). Convention:
    noether-default-v1 with epsilon^{0123} = +1/sqrt(-g) for the contravariant
    Levi-Civita tensor.

    Returns a vector (one index up).
    """
    T = torsion_of_connection(gamma)
    n = geom.dim
    T_down = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                T_down[lam, mu, nu] = _clean(
                    sum(geom.g[lam, k] * T[k, mu, nu] for k in range(n))
                )
    T_down = Array(T_down)
    A_up = sp.MutableDenseNDimArray.zeros(n)
    for rho in range(n):
        val = sp.Integer(0)
        for sig in range(n):
            for kap in range(n):
                for lam in range(n):
                    eps = _levi_civita_value(rho, sig, kap, lam, n)
                    val += eps * T_down[sig, kap, lam]
        A_up[rho] = _clean(val * sp.Rational(1, 6))
    return Array(A_up)


def torsion_trace_part(gamma) -> Array:
    """The trace-vector irreducible part of the torsion:

    _(1)T^lambda_{mu nu} = (1/3)(delta^lambda_mu T_nu - delta^lambda_nu T_mu)

    where T_mu = T^rho_{rho mu} is the trace vector.
    """
    T_vec = torsion_trace_vector(gamma)
    n = T_vec.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                val = sp.Rational(1, 3) * (
                    (1 if lam == mu else 0) * T_vec[nu]
                    - (1 if lam == nu else 0) * T_vec[mu]
                )
                out[lam, mu, nu] = _clean(val)
    return Array(out)


def torsion_axial_part(gamma, geom: ComponentGeometry) -> Array:
    """The axial-vector irreducible part of the torsion:

    _(2)T^lambda_{mu nu} = -(1/6) epsilon^lambda_{mu nu rho} A^rho

    where A^rho is the axial vector.
    """
    A = torsion_axial_vector(gamma, geom)
    n = geom.dim
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for rho in range(n):
                    # epsilon^lam_{mu nu rho} = g^{lam kappa} epsilon_{kappa mu nu rho}
                    eps_up_first = _clean(
                        sum(
                            geom.g_inv[lam, k]
                            * _levi_civita_value(k, mu, nu, rho, n)
                            for k in range(n)
                        )
                    )
                    val += eps_up_first * A[rho]
                out[lam, mu, nu] = _clean(-sp.Rational(1, 6) * val)
    return Array(out)


def torsion_traceless_tensor(gamma, geom: ComponentGeometry) -> Array:
    """The traceless-tensor irreducible part of the torsion:

    q^lambda_{mu nu} = T^lambda_{mu nu} - _(1)T^lambda_{mu nu} - _(2)T^lambda_{mu nu}

    This is the remainder after subtracting the trace and axial parts.
    It is traceless (q^lambda_{lambda mu} = 0) and has no totally
    antisymmetric component.
    """
    T = torsion_of_connection(gamma)
    t1 = torsion_trace_part(gamma)
    t2 = torsion_axial_part(gamma, geom)
    n = T.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                out[lam, mu, nu] = _clean(T[lam, mu, nu] - t1[lam, mu, nu] - t2[lam, mu, nu])
    return Array(out)


def nonmetricity_of_connection(
    coords: list[sp.Symbol], gamma, g
) -> Array:
    """Q_{lambda mu nu} = nabla_lambda g_{mu nu} of a general connection.

    Convention: noether-default-v1 (architecture.md section 7).
    Q is symmetric in the last pair (mu, nu).  For the Levi-Civita
    connection (Christoffel symbols), Q = 0 (metric compatibility).

    Q_{lambda mu nu} = partial_lambda g_{mu nu}
                       - Gamma^rho_{lambda mu} g_{rho nu}
                       - Gamma^rho_{lambda nu} g_{rho mu}

    Parameters:
        coords: coordinate symbols
        gamma: affine connection array gamma[a][b][c] = Gamma^a_{bc}
        g: metric matrix (symmetric)
    """
    n, x = len(coords), coords
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(mu, n):
                val = sp.diff(g[mu, nu], x[lam])
                for rho in range(n):
                    val -= gamma[rho, lam, mu] * g[rho, nu]
                    val -= gamma[rho, lam, nu] * g[rho, mu]
                val = _clean(val)
                out[lam, mu, nu] = val
                out[lam, nu, mu] = val  # symmetric in last pair
    return Array(out)


def nonmetricity_weyl_trace(
    coords: list[sp.Symbol], gamma, g, g_inv
) -> Array:
    """omega_lambda = Q_{lambda mu nu} g^{mu nu} (the Weyl / first trace).

    Returns a 1-form (all indices down).  The Weyl trace is the contraction
    of Q on its symmetric pair.
    """
    Q = nonmetricity_of_connection(coords, gamma, g)
    n = Q.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n)
    for lam in range(n):
        out[lam] = _clean(
            sum(Q[lam, mu, nu] * g_inv[mu, nu] for mu in range(n) for nu in range(n))
        )
    return Array(out)


def nonmetricity_second_trace(
    coords: list[sp.Symbol], gamma, g, g_inv
) -> Array:
    """qtilde_mu = Q_{lambda mu nu} g^{lambda nu} (the second trace).

    Returns a 1-form (all indices down).  The second trace contracts the
    first and third indices of Q.
    """
    Q = nonmetricity_of_connection(coords, gamma, g)
    n = Q.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n)
    for mu in range(n):
        out[mu] = _clean(
            sum(Q[lam, mu, nu] * g_inv[lam, nu] for lam in range(n) for nu in range(n))
        )
    return Array(out)


def nonmetricity_weyl_part(
    coords: list[sp.Symbol], gamma, g, g_inv
) -> Array:
    """The Weyl-vector trace irreducible part of non-metricity:

    Q^(W)_{lambda mu nu} = (1/((n+2)(n-1)))
        [(n+1) omega_lambda g_{mu nu}
         - (omega_mu g_{lambda nu} + omega_nu g_{lambda mu})]

    where omega_lambda = Q_{lambda mu nu} g^{mu nu} is the Weyl trace.

    Properties:
      Trace A (g^{mu nu} Q^(W)_{lambda mu nu}) = omega_lambda
      Trace B (g^{lambda nu} Q^(W)_{lambda mu nu}) = 0
    """
    omega = nonmetricity_weyl_trace(coords, gamma, g, g_inv)
    n = omega.shape[0]
    denom = (n + 2) * (n - 1)
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(mu, n):
                val = (
                    (n + 1) * omega[lam] * g[mu, nu]
                    - omega[mu] * g[lam, nu]
                    - omega[nu] * g[lam, mu]
                ) / denom
                val = _clean(val)
                out[lam, mu, nu] = val
                out[lam, nu, mu] = val  # symmetric in last pair
    return Array(out)


def nonmetricity_second_trace_part(
    coords: list[sp.Symbol], gamma, g, g_inv
) -> Array:
    """The second-trace irreducible part of non-metricity:

    Q^(2T)_{lambda mu nu} = (1/((n+2)(n-1)))
        [-2 qtilde_lambda g_{mu nu}
         + n(qtilde_mu g_{lambda nu} + qtilde_nu g_{lambda mu})]

    where qtilde_mu = Q_{lambda mu nu} g^{lambda nu} is the second trace.

    Properties:
      Trace A (g^{mu nu} Q^(2T)_{lambda mu nu}) = 0
      Trace B (g^{lambda nu} Q^(2T)_{lambda mu nu}) = qtilde_mu
    """
    qtilde = nonmetricity_second_trace(coords, gamma, g, g_inv)
    n = qtilde.shape[0]
    denom = (n + 2) * (n - 1)
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(mu, n):
                val = (
                    -2 * qtilde[lam] * g[mu, nu]
                    + n * (qtilde[mu] * g[lam, nu] + qtilde[nu] * g[lam, mu])
                ) / denom
                val = _clean(val)
                out[lam, mu, nu] = val
                out[lam, nu, mu] = val  # symmetric in last pair
    return Array(out)


def nonmetricity_traceless_tensor(
    coords: list[sp.Symbol], gamma, g, g_inv
) -> Array:
    """The traceless-tensor irreducible part of non-metricity:

    Q^(TL)_{lambda mu nu} = Q_{lambda mu nu}
        - Q^(W)_{lambda mu nu} - Q^(2T)_{lambda mu nu}

    This is the remainder after subtracting the Weyl and second-trace parts.
    It is traceless in both senses:
      Trace A: g^{mu nu} Q^(TL)_{lambda mu nu} = 0
      Trace B: g^{lambda nu} Q^(TL)_{lambda mu nu} = 0
    """
    Q = nonmetricity_of_connection(coords, gamma, g)
    qw = nonmetricity_weyl_part(coords, gamma, g, g_inv)
    q2t = nonmetricity_second_trace_part(coords, gamma, g, g_inv)
    n = Q.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                out[lam, mu, nu] = _clean(
                    Q[lam, mu, nu] - qw[lam, mu, nu] - q2t[lam, mu, nu]
                )
    return Array(out)


def covariant_derivative_of_connection(
    coords: list[sp.Symbol], gamma, arr, variances: list[str]
) -> Array:
    """Covariant derivative of a tensor using a general (possibly asymmetric)
    affine connection gamma[a][b][c] = Gamma^a_{bc}.

    Convention matches ComponentGeometry.covariant_derivative:
      nabla_a T^{...}_{...} = partial_a T + (connection terms)

    For each upper index s:  + Gamma^{idx_s}_{a lam} T(..., lam, ...)
    For each lower index s:  - Gamma^{lam}_{a idx_s} T(..., lam, ...)

    The connection gamma may be asymmetric (torsionful). Returns an array
    with a new leading 'down' axis.
    """
    n, x = len(coords), coords
    if not variances:
        return Array([_clean(sp.diff(arr, x[a])) for a in range(n)])
    rank = len(variances)
    shape = (n,) + (n,) * rank
    out = sp.MutableDenseNDimArray.zeros(*shape)
    for a in range(n):
        for idx in _all_indices(n, rank):
            val = sp.diff(arr[idx], x[a])
            for s, var in enumerate(variances):
                for lam in range(n):
                    swapped = idx[:s] + (lam,) + idx[s + 1 :]
                    if var == "up":
                        val += gamma[idx[s], a, lam] * arr[swapped]
                    else:
                        val -= gamma[lam, a, idx[s]] * arr[swapped]
            out[(a, *idx)] = _clean(val)
    return Array(out)


def riemann_down_of_connection(
    coords: list[sp.Symbol], gamma, g
) -> Array:
    """R_{rho sigma mu nu} (all indices down) of a general connection.

    Computed as g_{rho alpha} R^alpha_{sigma mu nu}(Gamma).
    """
    n = len(coords)
    R_up = riemann_of_connection(coords, gamma)
    out = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for rho in range(n):
        for sig in range(n):
            for mu in range(n):
                for nu in range(n):
                    out[rho, sig, mu, nu] = _clean(
                        sum(g[rho, alpha] * R_up[alpha, sig, mu, nu] for alpha in range(n))
                    )
    return Array(out)


def _levi_civita_value(i: int, j: int, k: int, ll: int, n: int) -> int:
    """Value of the Levi-Civita symbol epsilon_{ijkl} (not tensor) in n dimensions.

    Returns +1 for even permutations, -1 for odd permutations, 0 if any index
    repeats. Only valid for n >= 4 and exactly 4 indices.
    """
    indices = (i, j, k, ll)
    if len(set(indices)) < 4:
        return 0
    # Count inversions to determine the sign
    perm = list(indices)
    inversions = 0
    for a in range(len(perm)):
        for b in range(a + 1, len(perm)):
            if perm[a] > perm[b]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def projective_connection(geom: ComponentGeometry, covector) -> Array:
    """Gamma^lam_{mu nu} = C^lam_{mu nu}(g) + delta^lam_nu A_mu."""
    n = geom.dim
    out = sp.MutableDenseNDimArray(geom.christoffel)
    for a in range(n):
        for b in range(n):
            out[a, b, a] = _clean(out[a, b, a] + covector[b])
    return Array(out)


def _random_poly(rng: random.Random, coords: list[sp.Symbol]) -> sp.Expr:
    c = sp.Rational(rng.randint(1, 3), rng.randint(2, 5))
    return c * coords[rng.randrange(len(coords))]


def random_scalar_field(seed: int, coords: list[sp.Symbol]) -> sp.Expr:
    return _random_poly(random.Random(seed), coords)


def random_covector(seed: int, coords: list[sp.Symbol]) -> Array:
    rng = random.Random(seed)
    return Array([_random_poly(rng, coords) for _ in coords])


def random_antisymmetric(seed: int, coords: list[sp.Symbol]) -> Array:
    rng = random.Random(seed)
    n = len(coords)
    out = sp.MutableDenseNDimArray.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            p = _random_poly(rng, coords)
            out[i, j] = p
            out[j, i] = -p
    return Array(out)


def warped_product_4d() -> ComponentGeometry:
    """Deterministic warped-product 4-metric with NONZERO Gauss-Bonnet scalar,
    so the D=4 vanishing of the Lanczos tensor is a genuine cancellation
    between its quadratic-curvature pieces, not an artifact of GB = 0."""
    t, x, y, z = sp.symbols("t x y z")
    g = sp.diag(-(1 + x), 1, (1 + x) * (1 + y), 1 + y)
    return ComponentGeometry([t, x, y, z], g)


def sparse_diagonal_metric(seed: int, dim: int = 4, curved: int = 3) -> ComponentGeometry:
    """Seeded diagonal metric with only `curved` perturbed entries.

    Same Lorentzian mostly-plus shape as random_diagonal_metric, but the
    remaining entries stay constant, keeping the Riemann tensor sparse. Used
    for quartic-curvature checks (Gauss-Bonnet) where full random metrics
    make exact rational arithmetic prohibitively slow."""
    rng = random.Random(seed)
    names = ["t", "x", "y", "z", "w", "v"][:dim]
    coords = [sp.Symbol(nm) for nm in names]
    slots = sorted(rng.sample(range(dim), k=min(curved, dim)))
    entries = []
    for i in range(dim):
        if i in slots:
            c = sp.Rational(rng.randint(1, 3), rng.randint(2, 5))
            var = coords[rng.randrange(dim)]
            p = 1 + c * var
        else:
            p = sp.Integer(1)
        entries.append(-p if i == 0 else p)
    g = sp.diag(*entries)
    return ComponentGeometry(coords, g)


def random_diagonal_metric(seed: int, dim: int = 4) -> ComponentGeometry:
    """Seeded curved diagonal metric with polynomial entries.

    Lorentzian, mostly-plus: g = diag(-(1+p0), 1+p1, ..., 1+p_{dim-1}) with
    small random polynomials p_i in the coordinates. Deterministic per seed,
    which the provenance bundle records.
    """
    rng = random.Random(seed)
    names = ["t", "x", "y", "z", "w", "v"][:dim]
    coords = [sp.Symbol(nm) for nm in names]
    entries = []
    for i in range(dim):
        c = sp.Rational(rng.randint(1, 3), rng.randint(2, 5))
        var = coords[rng.randrange(dim)]
        deg = rng.randint(1, 2)
        p = c * var**deg
        entries.append(-(1 + p) if i == 0 else (1 + p))
    g = sp.diag(*entries)
    return ComponentGeometry(coords, g)


def random_affine_connection(
    seed: int, coords: list[sp.Symbol], *, symmetric: bool = False
) -> Array:
    """Seeded random affine connection gamma[a][b][c] = Gamma^a_{bc}.

    Each component is a small random polynomial in the coordinates.
    When ``symmetric=False`` (default) the lower pair (b,c) is NOT
    forced symmetric, so torsion T^a_{bc} = gamma[a,b,c] - gamma[a,c,b]
    is generically nonzero.  When ``symmetric=True`` the connection is
    symmetric in the lower pair (Levi-Civita-like, torsion-free).

    Deterministic per seed, so cross-checks reproduce exactly.
    """
    rng = random.Random(seed)
    n = len(coords)
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                p = _random_poly(rng, coords)
                out[a, b, c] = p
                if symmetric:
                    out[a, c, b] = p
                elif c != b:
                    q = _random_poly(rng, coords)
                    out[a, c, b] = q
    return Array(out)
