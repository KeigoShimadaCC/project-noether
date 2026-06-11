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
