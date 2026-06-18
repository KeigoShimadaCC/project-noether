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
                T_down[lam, mu, nu] = _clean(sum(geom.g[lam, k] * T[k, mu, nu] for k in range(n)))
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
                    (1 if lam == mu else 0) * T_vec[nu] - (1 if lam == nu else 0) * T_vec[mu]
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
                            geom.g_inv[lam, k] * _levi_civita_value(k, mu, nu, rho, n)
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


def nonmetricity_of_connection(coords: list[sp.Symbol], gamma, g) -> Array:
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


def nonmetricity_weyl_trace(coords: list[sp.Symbol], gamma, g, g_inv) -> Array:
    """omega_lambda = Q_{lambda mu nu} g^{mu nu} (the Weyl / first trace).

    Returns a 1-form (all indices down).  The Weyl trace is the contraction
    of Q on its symmetric pair.
    """
    Q = nonmetricity_of_connection(coords, gamma, g)
    n = Q.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n)
    for lam in range(n):
        out[lam] = _clean(sum(Q[lam, mu, nu] * g_inv[mu, nu] for mu in range(n) for nu in range(n)))
    return Array(out)


def nonmetricity_second_trace(coords: list[sp.Symbol], gamma, g, g_inv) -> Array:
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


def nonmetricity_weyl_part(coords: list[sp.Symbol], gamma, g, g_inv) -> Array:
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


def nonmetricity_second_trace_part(coords: list[sp.Symbol], gamma, g, g_inv) -> Array:
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


def nonmetricity_traceless_tensor(coords: list[sp.Symbol], gamma, g, g_inv) -> Array:
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
                out[lam, mu, nu] = _clean(Q[lam, mu, nu] - qw[lam, mu, nu] - q2t[lam, mu, nu])
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


def riemann_down_of_connection(coords: list[sp.Symbol], gamma, g) -> Array:
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


def christoffel_of_metric(coords: list[sp.Symbol], g, g_inv) -> Array:
    """Levi-Civita (Christoffel) connection of a metric.

    Gamma^a_{bc} = (1/2) g^{ad} (d_b g_{dc} + d_c g_{db} - d_d g_{bc})

    Convention: noether-default-v1.  The Christoffel symbols are symmetric
    in the lower pair (b,c).

    Parameters:
        coords: coordinate symbols
        g: metric matrix (symmetric, n x n)
        g_inv: inverse metric matrix (n x n)
    """
    n, x = len(coords), coords
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                val = sp.Rational(1, 2) * sum(
                    g_inv[a, d]
                    * (sp.diff(g[d, c], x[b]) + sp.diff(g[d, b], x[c]) - sp.diff(g[b, c], x[d]))
                    for d in range(n)
                )
                val = _clean(val)
                out[a, b, c] = val
                out[a, c, b] = val  # symmetric in lower pair
    return Array(out)


def contortion_of_torsion(gamma, g, g_inv) -> Array:
    """Contortion tensor K^lambda_{mu nu} from the torsion of a connection.

    K^lambda_{mu nu} = (1/2)(T^lambda_{mu nu}
                          + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu}
                          + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu})

    Convention: metric-affine-v1.  The contortion is the torsion-dependent
    part of the post-Riemannian decomposition Gamma = LC + K(T) + L(Q).
    It inverts to the torsion: K^lambda_{mu nu} - K^lambda_{nu mu} = T^lambda_{mu nu}.

    Parameters:
        gamma: affine connection array gamma[a][b][c] = Gamma^a_{bc}
        g: metric matrix (symmetric, n x n)
        g_inv: inverse metric matrix (n x n)
    """
    T = torsion_of_connection(gamma)
    n = T.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                term1 = T[lam, mu, nu]
                # g^{lam sig} g_{mu tau} T^tau_{sig nu}
                term2 = sum(
                    g_inv[lam, sig] * g[mu, tau] * T[tau, sig, nu]
                    for sig in range(n)
                    for tau in range(n)
                )
                # g^{lam sig} g_{nu tau} T^tau_{sig mu}
                term3 = sum(
                    g_inv[lam, sig] * g[nu, tau] * T[tau, sig, mu]
                    for sig in range(n)
                    for tau in range(n)
                )
                out[lam, mu, nu] = _clean(sp.Rational(1, 2) * (term1 + term2 + term3))
    return Array(out)


def disformation_of_nonmetricity(coords: list[sp.Symbol], gamma, g, g_inv) -> Array:
    """Disformation tensor L^lambda_{mu nu} from the non-metricity.

    L^lambda_{mu nu} = (1/2) g^{lambda rho}(-Q_{mu nu rho}
                                        - Q_{nu rho mu} + Q_{rho mu nu})

    Convention: metric-affine-v1.  The disformation is the non-metricity-
    dependent part of the post-Riemannian decomposition
    Gamma = LC + K(T) + L(Q).  It inverts to the non-metricity:
    when T=0, Q_{lambda mu nu} = -(L^rho_{lambda mu} g_{rho nu}
                                   + L^rho_{lambda nu} g_{rho mu}).

    The disformation is symmetric in the lower pair (mu, nu), reflecting
    that it originates from non-metricity (which is symmetric in its
    last pair) rather than torsion.

    Parameters:
        coords: coordinate symbols (unused, kept for API consistency)
        gamma: affine connection array gamma[a][b][c] = Gamma^a_{bc}
        g: metric matrix (symmetric, n x n)
        g_inv: inverse metric matrix (n x n)
    """
    Q = nonmetricity_of_connection(coords, gamma, g)
    n = Q.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                val = sum(
                    g_inv[lam, rho] * (-Q[mu, nu, rho] - Q[nu, rho, mu] + Q[rho, mu, nu])
                    for rho in range(n)
                )
                out[lam, mu, nu] = _clean(sp.Rational(1, 2) * val)
    return Array(out)


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


# ---------------------------------------------------------------------------
# Modified Bianchi identity oracles (SymPy cross-check layer).
#
# These functions compute the residual (LHS - RHS) of the modified Bianchi
# identities for a general affine connection, returning an array that should
# be identically zero when the identity holds.  They serve as the SymPy
# oracle for the dual-gate verification model (Cadabra residue + SymPy
# component cross-check).
#
# Convention: noether-default-v1 + metric-affine-v1.
# ---------------------------------------------------------------------------


def first_bianchi_residual(coords: list[sp.Symbol], gamma: Array, g: Array, g_inv: Array) -> Array:
    """Residual of the modified first Bianchi identity for a general
    connection carrying torsion.

    R^rho_{sigma mu nu} + R^rho_{mu nu sigma} + R^rho_{nu sigma mu}
      - (nabla_sigma T^rho_{mu nu} + nabla_mu T^rho_{nu sigma}
         + nabla_nu T^rho_{sigma mu}
         + T^rho_{alpha sigma} T^alpha_{mu nu}
         + T^rho_{alpha mu} T^alpha_{nu sigma}
         + T^rho_{alpha nu} T^alpha_{sigma mu})

    Returns a (n, n, n, n) array.  When the identity holds every
    component is zero.  On a Levi-Civita connection (T=0) the RHS
    vanishes and the residual is just the cyclic sum of the Riemann
    tensor, which is zero for any metric-compatible torsion-free
    connection.

    Convention: noether-default-v1 + metric-affine-v1.
    """
    n = len(coords)
    R_up = riemann_of_connection(coords, gamma)
    T = torsion_of_connection(gamma)
    # riemann_of_connection already returns R^rho_{sigma mu nu} (raised)
    # torsion_of_connection already returns T^rho_{mu nu} (raised)

    # nabla_sigma T^rho_{mu nu} (covariant derivative of (1,2) tensor)
    nabla_T = covariant_derivative_of_connection(coords, gamma, T, variances=["up", "down", "down"])
    # T^rho_{alpha sigma} T^alpha_{mu nu}
    TT = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for rho in range(n):
        for sig in range(n):
            for mu in range(n):
                for nu in range(n):
                    TT[rho, sig, mu, nu] = _clean(
                        sum(T[rho, alpha, sig] * T[alpha, mu, nu] for alpha in range(n))
                    )
    residual = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for rho in range(n):
        for sig in range(n):
            for mu in range(n):
                for nu in range(n):
                    lhs = R_up[rho, sig, mu, nu] + R_up[rho, mu, nu, sig] + R_up[rho, nu, sig, mu]
                    rhs = (
                        nabla_T[sig, rho, mu, nu]
                        + nabla_T[mu, rho, nu, sig]
                        + nabla_T[nu, rho, sig, mu]
                        + TT[rho, sig, mu, nu]
                        + TT[rho, mu, nu, sig]
                        + TT[rho, nu, sig, mu]
                    )
                    residual[rho, sig, mu, nu] = _clean(lhs - rhs)
    return Array(residual)


def contracted_second_bianchi_residual(
    coords: list[sp.Symbol], gamma: Array, g: Array, g_inv: Array
) -> Array:
    """Residual of the modified contracted second Bianchi identity for a
    general connection carrying torsion.

    nabla_rho R^rho_{sigma mu nu}
      - nabla_mu R_{sigma nu}
      + nabla_nu R_{sigma mu}
      + (R^rho_{sigma alpha mu} T^alpha_{nu rho}
         + R^rho_{sigma alpha nu} T^alpha_{rho mu})
      - R_{sigma alpha} T^alpha_{mu nu}

    Returns a (n, n, n) array indexed [sigma, mu, nu].  When the identity
    holds every component is zero.  On a Levi-Civita connection (T=0) the
    correction terms vanish and the residual is just the LC contracted
    second Bianchi identity.

    The sign structure of the correction terms reflects the contraction:
    R^rho_{sigma alpha rho} = -R_{sigma alpha} (antisymmetry of the last
    pair of the Riemann tensor), so the double negation yields
    +R_{sigma alpha} T^alpha_{mu nu} in the identity.  Moving the RHS to
    the LHS gives +R^rho...T terms and -R_{sigma alpha} T^alpha.

    **Caveat:** This simplified form is valid on metric-compatible (Q=0)
    backgrounds where nabla commutes with index contraction.  For Q != 0,
    use the uncontracted second Bianchi and contract numerically.

    Convention: noether-default-v1 + metric-affine-v1.
    """
    n = len(coords)
    R_up = riemann_of_connection(coords, gamma)
    Ric = ricci_of_connection(coords, gamma)
    T = torsion_of_connection(gamma)

    # nabla_rho R^rho_{sigma mu nu} (divergence on first index)
    nabla_R = covariant_derivative_of_connection(
        coords, gamma, R_up, variances=["up", "down", "down", "down"]
    )
    div_R = sp.MutableDenseNDimArray.zeros(n, n, n)
    for sig in range(n):
        for mu in range(n):
            for nu in range(n):
                div_R[sig, mu, nu] = _clean(sum(nabla_R[rho, rho, sig, mu, nu] for rho in range(n)))

    nabla_Ric = covariant_derivative_of_connection(coords, gamma, Ric, variances=["down", "down"])

    # Correction terms (moving RHS to LHS, so signs flip):
    # Identity: LHS = -(corr1 + corr2) + corr3
    # So residual = LHS - (-(corr1+corr2) + corr3) = LHS + corr1 + corr2 - corr3

    # corr1: R^rho_{sigma alpha mu} T^alpha_{nu rho}
    corr1 = sp.MutableDenseNDimArray.zeros(n, n, n)
    for sig in range(n):
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for rho in range(n):
                    for alp in range(n):
                        val += R_up[rho, sig, alp, mu] * T[alp, nu, rho]
                corr1[sig, mu, nu] = _clean(val)

    # corr2: R^rho_{sigma alpha nu} T^alpha_{rho mu}
    corr2 = sp.MutableDenseNDimArray.zeros(n, n, n)
    for sig in range(n):
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for rho in range(n):
                    for alp in range(n):
                        val += R_up[rho, sig, alp, nu] * T[alp, rho, mu]
                corr2[sig, mu, nu] = _clean(val)

    # corr3: R_{sigma alpha} T^alpha_{mu nu} (positive in the identity)
    corr3 = sp.MutableDenseNDimArray.zeros(n, n, n)
    for sig in range(n):
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for alp in range(n):
                    val += Ric[sig, alp] * T[alp, mu, nu]
                corr3[sig, mu, nu] = _clean(val)

    # Residual = LHS - RHS = LHS - (-(corr1+corr2) + corr3)
    #          = LHS + corr1 + corr2 - corr3
    residual = sp.MutableDenseNDimArray.zeros(n, n, n)
    for sig in range(n):
        for mu in range(n):
            for nu in range(n):
                lhs = div_R[sig, mu, nu] - nabla_Ric[mu, sig, nu] + nabla_Ric[nu, sig, mu]
                rhs = -(corr1[sig, mu, nu] + corr2[sig, mu, nu]) + corr3[sig, mu, nu]
                residual[sig, mu, nu] = _clean(lhs - rhs)
    return Array(residual)


def exterior_derivative_of_1form(coords: list[sp.Symbol], A) -> Array:
    """Exterior derivative of a 1-form: (dA)_{mu nu} = partial_mu A_nu - partial_nu A_mu.

    Returns an antisymmetric (n, n) array.  The exterior derivative is
    defined purely in terms of partial derivatives, independent of any
    connection.  It is the standard field-strength definition for an
    abelian gauge potential (Maxwell F = dA).

    Convention: coordinate-independent (no connection involved).
    """
    n, x = len(coords), coords
    out = sp.MutableDenseNDimArray.zeros(n, n)
    for mu in range(n):
        for nu in range(mu + 1, n):
            val = _clean(sp.diff(A[nu], x[mu]) - sp.diff(A[mu], x[nu]))
            out[mu, nu] = val
            out[nu, mu] = _clean(-val)
    return Array(out)


def covariant_curl_of_1form(
    coords: list[sp.Symbol], gamma, A, variances_A: list[str] | None = None
) -> Array:
    """Covariant curl of a 1-form: (nabla_mu A_nu - nabla_nu A_mu)_{mu nu}.

    Returns an antisymmetric (n, n) array.  The covariant curl uses the
    full affine connection (possibly torsionful) to compute the covariant
    derivative of A, then antisymmetrizes.

    Parameters:
        coords: coordinate symbols
        gamma: affine connection array gamma[a][b][c] = Gamma^a_{bc}
        A: 1-form (covector) array with n components (all indices down)
        variances_A: variances of A indices, defaults to ["down"] for a
            covector / 1-form.

    Convention: noether-default-v1 + metric-affine-v1.

    The covariant curl equals the exterior derivative minus the torsion
    term: covariant_curl = dA - T^lam_{mu nu} A_lam.
    """
    if variances_A is None:
        variances_A = ["down"]
    n = len(coords)
    # nabla_mu A_nu: covariant derivative of the 1-form A
    nabla_A = covariant_derivative_of_connection(coords, gamma, A, variances=variances_A)
    # nabla_A[mu, nu] = nabla_mu A_nu (n, n) array
    out = sp.MutableDenseNDimArray.zeros(n, n)
    for mu in range(n):
        for nu in range(mu + 1, n):
            val = _clean(nabla_A[mu, nu] - nabla_A[nu, mu])
            out[mu, nu] = val
            out[nu, mu] = _clean(-val)
    return Array(out)


# ---------------------------------------------------------------------------
# Hypermomentum decomposition oracles (SymPy cross-check layer).
#
# The hypermomentum Delta^lambda_{mu nu} decomposes under GL(n) into
# spin (antisymmetric in first pair), dilation (trace), and shear
# (traceless symmetric in first pair).  These functions compute each
# piece and verify the reconstruction and trace properties on explicit
# random backgrounds.
#
# Convention: noether-default-v1 + metric-affine-v1.
# ---------------------------------------------------------------------------


def _kronecker_delta(i: int, j: int) -> sp.Expr:
    """Kronecker delta as a SymPy expression (1 if i==j, 0 otherwise)."""
    return sp.Integer(1) if i == j else sp.Integer(0)


def hypermomentum_spin(Delta: Array, g: Array, g_inv: Array) -> Array:
    """Spin (antisymmetric) part of the hypermomentum:

    tau^lambda_{mu nu} = (1/2)(Delta^lambda_{mu nu}
                          - g^{lambda rho} g_{mu sig} Delta^{sig}_{rho nu})

    The spin part is antisymmetric in the first pair (lambda, mu) and
    traceless (tau^lambda_{lambda nu} = 0).

    Parameters:
        Delta: hypermomentum tensor Delta^{lam}_{mu nu} (n, n, n) array
        g: metric matrix (n, n)
        g_inv: inverse metric matrix (n, n)

    Convention: metric-affine-v1."""
    n = Delta.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                # g^{lam rho} g_{mu sig} Delta^{sig}_{rho nu}
                swapped = sp.Integer(0)
                for rho in range(n):
                    for sig in range(n):
                        swapped += g_inv[lam, rho] * g[mu, sig] * Delta[sig, rho, nu]
                out[lam, mu, nu] = _clean(sp.Rational(1, 2) * (Delta[lam, mu, nu] - swapped))
    return Array(out)


def hypermomentum_dilation_trace(Delta: Array) -> Array:
    """Dilation (trace) vector of the hypermomentum:

    Delta_nu = Delta^{lambda}_{lambda nu}

    The dilation carries the full trace (both spin and shear are
    traceless).

    Parameters:
        Delta: hypermomentum tensor Delta^{lam}_{mu nu} (n, n, n) array

    Convention: metric-affine-v1."""
    n = Delta.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n)
    for nu in range(n):
        out[nu] = _clean(sum(Delta[lam, lam, nu] for lam in range(n)))
    return Array(out)


def hypermomentum_shear(Delta: Array, g: Array, g_inv: Array) -> Array:
    """Shear (traceless symmetric) part of the hypermomentum:

    sigma^lambda_{mu nu} = (1/2)(Delta^lambda_{mu nu}
                            + g^{lambda rho} g_{mu sig} Delta^{sig}_{rho nu})
                            - (1/n) delta^lambda_mu Delta^{kappa}_{kappa nu}

    The shear part is symmetric in the first pair (lambda, mu) and
    traceless (sigma^lambda_{lambda nu} = 0).

    Parameters:
        Delta: hypermomentum tensor Delta^{lam}_{mu nu} (n, n, n) array
        g: metric matrix (n, n)
        g_inv: inverse metric matrix (n, n)

    Convention: metric-affine-v1."""
    n = Delta.shape[0]
    trace = hypermomentum_dilation_trace(Delta)
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                # g^{lam rho} g_{mu sig} Delta^{sig}_{rho nu}
                swapped = sp.Integer(0)
                for rho in range(n):
                    for sig in range(n):
                        swapped += g_inv[lam, rho] * g[mu, sig] * Delta[sig, rho, nu]
                trace_part = _kronecker_delta(lam, mu) * trace[nu]
                out[lam, mu, nu] = _clean(
                    sp.Rational(1, 2) * (Delta[lam, mu, nu] + swapped)
                    - sp.Rational(1, n) * trace_part
                )
    return Array(out)


def hypermomentum_reconstruction_residual(Delta: Array, g: Array, g_inv: Array) -> Array:
    """Residual of the hypermomentum reconstruction identity:

    Delta - (tau + (1/n) delta^{lam}_{mu} Delta_nu + sigma)

    Should be zero componentwise for any Delta.  Verifies the
    algebraic decomposition Delta = spin + dilation + shear.

    Parameters:
        Delta: hypermomentum tensor Delta^{lam}_{mu nu} (n, n, n) array
        g: metric matrix (n, n)
        g_inv: inverse metric matrix (n, n)

    Convention: metric-affine-v1."""
    n = Delta.shape[0]
    tau = hypermomentum_spin(Delta, g, g_inv)
    trace = hypermomentum_dilation_trace(Delta)
    sigma = hypermomentum_shear(Delta, g, g_inv)
    residual = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                dilation_part = _kronecker_delta(lam, mu) * trace[nu] / n
                recon = tau[lam, mu, nu] + dilation_part + sigma[lam, mu, nu]
                residual[lam, mu, nu] = _clean(Delta[lam, mu, nu] - recon)
    return Array(residual)


def hypermomentum_spin_trace_residual(Delta: Array, g: Array, g_inv: Array) -> Array:
    """Residual of the spin trace identity: tau^{lambda}_{lambda nu} = 0.

    Should be zero componentwise for any Delta.

    Parameters:
        Delta: hypermomentum tensor Delta^{lam}_{mu nu} (n, n, n) array
        g: metric matrix (n, n)
        g_inv: inverse metric matrix (n, n)

    Convention: metric-affine-v1."""
    tau = hypermomentum_spin(Delta, g, g_inv)
    n = Delta.shape[0]
    residual = sp.MutableDenseNDimArray.zeros(n)
    for nu in range(n):
        residual[nu] = _clean(sum(tau[lam, lam, nu] for lam in range(n)))
    return Array(residual)


def hypermomentum_shear_trace_residual(Delta: Array, g: Array, g_inv: Array) -> Array:
    """Residual of the shear trace identity: sigma^{lambda}_{lambda nu} = 0.

    Should be zero componentwise for any Delta.

    Parameters:
        Delta: hypermomentum tensor Delta^{lam}_{mu nu} (n, n, n) array
        g: metric matrix (n, n)
        g_inv: inverse metric matrix (n, n)

    Convention: metric-affine-v1."""
    sigma = hypermomentum_shear(Delta, g, g_inv)
    n = Delta.shape[0]
    residual = sp.MutableDenseNDimArray.zeros(n)
    for nu in range(n):
        residual[nu] = _clean(sum(sigma[lam, lam, nu] for lam in range(n)))
    return Array(residual)


def hypermomentum_spin_antisym_residual(Delta: Array, g: Array, g_inv: Array) -> Array:
    """Residual of the spin antisymmetry: tau_{lam mu nu} + tau_{mu lam nu} = 0.

    The spin part is antisymmetric in the first pair when both indices
    are lowered: g_{lam alpha} tau^alpha_{mu nu} + g_{mu alpha} tau^alpha_{lam nu} = 0.

    Parameters:
        Delta: hypermomentum tensor Delta^{lam}_{mu nu} (n, n, n) array
        g: metric matrix (n, n)
        g_inv: inverse metric matrix (n, n)

    Convention: metric-affine-v1."""
    tau = hypermomentum_spin(Delta, g, g_inv)
    n = Delta.shape[0]
    residual = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                # Lower the first index of tau^alpha_{mu nu}
                # g_{lam alpha} tau^alpha_{mu nu}
                lowered1 = sum(g[lam, alpha] * tau[alpha, mu, nu] for alpha in range(n))
                # g_{mu alpha} tau^alpha_{lam nu}
                lowered2 = sum(g[mu, alpha] * tau[alpha, lam, nu] for alpha in range(n))
                residual[lam, mu, nu] = _clean(lowered1 + lowered2)
    return Array(residual)


def hypermomentum_shear_sym_residual(Delta: Array, g: Array, g_inv: Array) -> Array:
    """Residual of the shear symmetry: sigma_{lam mu nu} - sigma_{mu lam nu} = 0.

    The shear part is symmetric in the first pair when both indices
    are lowered: g_{lam alpha} sigma^alpha_{mu nu} - g_{mu alpha} sigma^alpha_{lam nu} = 0.

    Parameters:
        Delta: hypermomentum tensor Delta^{lam}_{mu nu} (n, n, n) array
        g: metric matrix (n, n)
        g_inv: inverse metric matrix (n, n)

    Convention: metric-affine-v1."""
    sigma = hypermomentum_shear(Delta, g, g_inv)
    n = Delta.shape[0]
    residual = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                # Lower the first index of sigma^alpha_{mu nu}
                lowered1 = sum(g[lam, alpha] * sigma[alpha, mu, nu] for alpha in range(n))
                lowered2 = sum(g[mu, alpha] * sigma[alpha, lam, nu] for alpha in range(n))
                residual[lam, mu, nu] = _clean(lowered1 - lowered2)
    return Array(residual)


def random_hypermomentum(seed: int, dim: int = 3) -> Array:
    """Seeded random hypermomentum tensor Delta^{lam}_{mu nu}.

    Each component is a small random integer.  Deterministic per seed.
    Used for cross-checking the decomposition identities on explicit
    backgrounds."""
    rng = random.Random(seed)
    n = dim
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                out[lam, mu, nu] = sp.Integer(rng.randint(-3, 3))
    return Array(out)


def palatini_connection_eom(
    coords: list[sp.Symbol], gamma: Array, g: Array, g_inv: Array
) -> Array:
    """Palatini connection EOM (coefficient of delta Gamma^alpha_{beta gamma})
    for the action S = -int sqrt(-g) g^{sigma nu} R_{nu sigma}(Gamma).

    The Euler-Lagrange equation is:

      E^alpha_{beta gamma} = partial_alpha(sg g^{gamma beta})
                            - delta^alpha_beta partial_rho(sg g^{gamma rho})
                            - sg delta^alpha_beta (g^{sigma nu} Gamma^gamma_{nu sigma})
                            - sg g^{beta gamma} Gamma^lambda_{lambda alpha}
                            + sg g^{sigma beta} Gamma^gamma_{alpha sigma}
                            + sg g^{gamma nu} Gamma^beta_{nu alpha}

    where sg = sqrt(-g).  The first two terms depend only on the metric
    and its derivatives (not on Gamma).  The remaining terms are purely
    algebraic in Gamma.  This structure is what makes the Palatini
    connection equation algebraic in the contortion K when Gamma = LC + K
    on a metric-compatible (Q=0) background: the partial(sg g) terms are
    the same for Gamma and LC, and the difference involves only K without
    derivatives of K.

    Returns a (n, n, n) array E^alpha_{beta gamma}.  On the Palatini
    solution (Gamma = LC + projective mode), this evaluates to zero.

    Convention: noether-default-v1 + metric-affine-v1."""
    n = len(coords)
    x = coords
    g_mat = sp.Matrix([[g[i, j] for j in range(n)] for i in range(n)])
    sg = sp.sqrt(-sp.det(g_mat))

    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                # partial_a(sg g^{c b})
                d_sg = _clean(sp.diff(sg, x[a]))
                d_ginv = _clean(sp.diff(g_inv[c, b], x[a]))
                term1 = _clean(d_sg * g_inv[c, b] + sg * d_ginv)

                # -delta^a_b partial_rho(sg g^{c rho})
                if a == b:
                    term2 = sp.Integer(0)
                    for rho in range(n):
                        d_sg_rho = _clean(sp.diff(sg, x[rho]))
                        d_ginv_rho = _clean(sp.diff(g_inv[c, rho], x[rho]))
                        term2 -= _clean(d_sg_rho * g_inv[c, rho] + sg * d_ginv_rho)
                else:
                    term2 = sp.Integer(0)

                # -sg delta^a_b (g^{sigma nu} Gamma^c_{nu sigma})
                if a == b:
                    term3 = sp.Integer(0)
                    for sig in range(n):
                        for nu in range(n):
                            term3 -= sg * g_inv[sig, nu] * gamma[c, nu, sig]
                    term3 = _clean(term3)
                else:
                    term3 = sp.Integer(0)

                # -sg g^{b c} Gamma^lam_{lam a}
                term4 = _clean(
                    -sg * g_inv[b, c] * sum(gamma[lam, lam, a] for lam in range(n))
                )

                # +sg g^{sigma b} Gamma^c_{a sigma}
                term5 = _clean(
                    sg * sum(g_inv[sig, b] * gamma[c, a, sig] for sig in range(n))
                )

                # +sg g^{c nu} Gamma^b_{nu a}
                term6 = _clean(
                    sg * sum(g_inv[c, nu] * gamma[b, nu, a] for nu in range(n))
                )

                out[a, b, c] = _clean(term1 + term2 + term3 + term4 + term5 + term6)

    return Array(out)


def einstein_cartan_algebraic_in_K_residual(
    coords: list[sp.Symbol], gamma: Array, g: Array, g_inv: Array
) -> Array:
    """Residual verifying the Palatini connection EOM is algebraic in K.

    On a metric-compatible (Q=0) torsionful background where
    Gamma = LC(g) + K(T), the Palatini connection EOM splits as:

      E(Gamma) = E_metric_part + E_Gamma_part(LC + K)

    The metric part (partial_a(sg g^{c b}) - delta^a_b ...) is the same
    whether Gamma or LC is used.  The Gamma part is linear in Gamma, so
    E(Gamma) - E(LC) is purely algebraic in K:

      E(Gamma) - E(LC) = sg [-delta^a_b g^{sigma nu} K^c_{nu sigma}
                             - g^{b c} K^lam_{lam a}
                             + g^{sigma b} K^c_{a sigma}
                             + g^{c nu} K^b_{nu a}]

    This residual should be zero componentwise, confirming:
    (1) The EOM difference between Gamma = LC + K and Gamma = LC equals
        the expected algebraic K expression.
    (2) The K expression has no derivative-of-K terms (it is manifestly
        algebraic in K, not proportional to partial K).

    This is the SymPy cross-check for VAL-EOM-011: the Palatini
    connection EOM is algebraic in the contortion K, meaning torsion is
    algebraically determined by any spin source rather than propagating
    as an independent degree of freedom.

    Parameters:
        coords: coordinate symbols
        gamma: affine connection array (must be metric-compatible, Q=0)
        g: metric matrix (symmetric)
        g_inv: inverse metric matrix

    Returns a (n, n, n) array.  Should be zero componentwise on
    metric-compatible (Q=0) torsionful backgrounds.

    Convention: noether-default-v1 + metric-affine-v1."""
    n = len(coords)
    g_mat = sp.Matrix([[g[i, j] for j in range(n)] for i in range(n)])
    sg = sp.sqrt(-sp.det(g_mat))

    LC = christoffel_of_metric(coords, g, g_inv)
    K = contortion_of_torsion(gamma, g, g_inv)

    # Compute the Palatini EOM for the full connection and for LC alone
    E_full = palatini_connection_eom(coords, gamma, g, g_inv)
    E_lc = palatini_connection_eom(coords, LC, g, g_inv)

    # Expected algebraic K contribution (linear in K, no derivative-of-K):
    # sg * [-delta^a_b g^{sigma nu} K^c_{nu sigma}
    #       - g^{b c} K^lam_{lam a}
    #       + g^{sigma b} K^c_{a sigma}
    #       + g^{c nu} K^b_{nu a}]
    K_contribution = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                # -delta^a_b sg g^{sigma nu} K^c_{nu sigma}
                if a == b:
                    term_delta = sp.Integer(0)
                    for sig in range(n):
                        for nu in range(n):
                            term_delta -= sg * g_inv[sig, nu] * K[c, nu, sig]
                    term_delta = _clean(term_delta)
                else:
                    term_delta = sp.Integer(0)

                # -sg g^{b c} K^lam_{lam a}
                term_trace = _clean(
                    -sg * g_inv[b, c] * sum(K[lam, lam, a] for lam in range(n))
                )

                # +sg g^{sigma b} K^c_{a sigma}
                term_raise1 = _clean(
                    sg * sum(g_inv[sig, b] * K[c, a, sig] for sig in range(n))
                )

                # +sg g^{c nu} K^b_{nu a}
                term_raise2 = _clean(
                    sg * sum(g_inv[c, nu] * K[b, nu, a] for nu in range(n))
                )

                K_contribution[a, b, c] = _clean(
                    term_delta + term_trace + term_raise1 + term_raise2
                )

    # Residual: (E_full - E_lc) - K_contribution should be zero
    residual = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                residual[a, b, c] = _clean(
                    E_full[a, b, c] - E_lc[a, b, c] - K_contribution[a, b, c]
                )

    return Array(residual)


def lc_contracted_bianchi_residual(
    coords: list[sp.Symbol], gamma: Array, g: Array, g_inv: Array
) -> Array:
    """Residual of the Levi-Civita contracted Bianchi identity
    (divergence form) on a general connection.

    g^{mu nu} nabla_mu R_{nu beta} - 1/2 nabla_beta R

    Returns a (n,) array.  On a Levi-Civita connection this is zero
    componentwise; on a connection with torsion or non-metricity it
    is generically nonzero (the torsion trap).  This function is used
    for the trap-guard test: reusing the LC contracted_bianchi under
    torsion would be caught.

    Convention: noether-default-v1.
    """
    n = len(coords)
    Ric = ricci_of_connection(coords, gamma)
    R_scalar = _clean(sum(g_inv[a, b] * Ric[a, b] for a in range(n) for b in range(n)))
    nabla_Ric = covariant_derivative_of_connection(coords, gamma, Ric, variances=["down", "down"])
    nabla_R = covariant_derivative_of_connection(coords, gamma, R_scalar, variances=[])
    residual = sp.MutableDenseNDimArray.zeros(n)
    for beta in range(n):
        div_Ric = sp.Integer(0)
        for mu in range(n):
            for nu in range(n):
                div_Ric += g_inv[mu, nu] * nabla_Ric[mu, nu, beta]
        residual[beta] = _clean(div_Ric - sp.Rational(1, 2) * nabla_R[beta])
    return Array(residual)
