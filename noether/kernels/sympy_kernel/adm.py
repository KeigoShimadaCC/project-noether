"""ADM (3+1) component verification under noether-default-v1 conventions.

Everything here was determined by kernel computation, not asserted from
memory: each sign and factor below was found by probing the alternatives on a
nondegenerate background and keeping the one with exactly zero residue
(see evals/eval1s_adm.py for the acceptance gate that re-runs them).

Kernel-verified statements (mostly-plus signature, t-constant spacelike
slices, future-pointing unit normal n_mu = (-N, 0, ..., 0)):

  (A) K_ij = (d_t h_ij - D_i N_j - D_j N_i) / (2N)  equals  +nabla_i n_j
      (the expansion-positive extrinsic-curvature convention).
  (B) sqrt(-g) R = N sqrt(h) (R3 + K_ij K^ij - K^2)
                   - 2 d_mu( sqrt(-g) (n^nu nabla_nu n^mu - n^mu nabla_nu n^nu) ).
  (C) 2 G_{mu nu} n^mu n^nu = R3 + K^2 - K_ij K^ij   (Hamiltonian constraint).
  (D) G_{mu i} n^mu = D_j (K^j_i - delta^j_i K)      (momentum constraint).
  (E) d/dN [ N (R3 + K_ij K^ij - K^2) ] = R3 + K^2 - K_ij K^ij
      (the lapse Euler-Lagrange equation reproduces (C); N enters the bulk
      undifferentiated, so EL_N is a plain partial derivative).

All residues are reduced to rational functions by dividing out sqrt(h)
analytically (d_mu(sqrt(h) w) = sqrt(h)(d_mu w + w d_mu det h / (2 det h))),
so cancel() is a canonical, fast zero test.
"""

from functools import cached_property

import sympy as sp

from noether.kernels.sympy_kernel.geometry import ComponentGeometry

Array = sp.ImmutableDenseNDimArray


def _zero(expr: sp.Expr) -> bool:
    return sp.cancel(sp.together(expr)) == 0


class ADMGeometry:
    """A foliated spacetime in lapse/shift/spatial-metric variables."""

    def __init__(
        self,
        time_coord: sp.Symbol,
        spatial_coords: list[sp.Symbol],
        lapse: sp.Expr,
        shift: list[sp.Expr],
        spatial_metric: sp.Matrix,
    ) -> None:
        self.t = time_coord
        self.xs = list(spatial_coords)
        self.d = len(self.xs)
        self.coords = [self.t, *self.xs]
        self.N = lapse
        self.shift = list(shift)
        self.h = sp.ImmutableMatrix(spatial_metric)

    @cached_property
    def spatial(self) -> ComponentGeometry:
        return ComponentGeometry(self.xs, sp.Matrix(self.h))

    @cached_property
    def shift_down(self) -> list[sp.Expr]:
        d = self.d
        return [sp.cancel(sum(self.h[i, j] * self.shift[j] for j in range(d))) for i in range(d)]

    @cached_property
    def full(self) -> ComponentGeometry:
        d = self.d
        g = sp.zeros(d + 1, d + 1)
        g[0, 0] = -(self.N**2) + sum(self.shift[i] * self.shift_down[i] for i in range(d))
        for i in range(d):
            g[0, i + 1] = self.shift_down[i]
            g[i + 1, 0] = self.shift_down[i]
            for j in range(d):
                g[i + 1, j + 1] = self.h[i, j]
        return ComponentGeometry(self.coords, sp.Matrix(g))

    @cached_property
    def extrinsic(self) -> Array:
        """K_ij from the ADM formula (convention verified in check (A))."""
        d = self.d
        G3 = self.spatial.christoffel
        DN = sp.MutableDenseNDimArray.zeros(d, d)
        for i in range(d):
            for j in range(d):
                DN[i, j] = sp.diff(self.shift_down[j], self.xs[i]) - sum(
                    G3[k, i, j] * self.shift_down[k] for k in range(d)
                )
        K = sp.MutableDenseNDimArray.zeros(d, d)
        for i in range(d):
            for j in range(d):
                K[i, j] = sp.cancel(
                    (sp.diff(self.h[i, j], self.t) - DN[i, j] - DN[j, i]) / (2 * self.N)
                )
        return Array(K)

    @cached_property
    def trace_K(self) -> sp.Expr:
        d, hinv, K = self.d, self.spatial.g_inv, self.extrinsic
        return sp.cancel(sum(hinv[i, j] * K[i, j] for i in range(d) for j in range(d)))

    @cached_property
    def K_squared(self) -> sp.Expr:
        """K_ij K^ij."""
        d, hinv, K = self.d, self.spatial.g_inv, self.extrinsic
        return sp.cancel(
            sum(
                hinv[i, k] * hinv[j, m] * K[i, j] * K[k, m]
                for i in range(d)
                for j in range(d)
                for k in range(d)
                for m in range(d)
            )
        )

    @cached_property
    def hamiltonian_form(self) -> sp.Expr:
        """R3 + K^2 - K_ij K^ij (the vacuum Hamiltonian constraint density
        over sqrt(h), kernel-verified in checks (C) and (E))."""
        return sp.cancel(self.spatial.ricci_scalar + self.trace_K**2 - self.K_squared)

    @cached_property
    def momentum_form(self) -> list[sp.Expr]:
        """D_j (K^j_i - delta^j_i K) per spatial i (verified in check (D))."""
        d, hinv, K = self.d, self.spatial.g_inv, self.extrinsic
        G3 = self.spatial.christoffel
        T = sp.MutableDenseNDimArray.zeros(d, d)  # T^j_i
        for j in range(d):
            for i in range(d):
                up = sum(hinv[j, k] * K[k, i] for k in range(d))
                T[j, i] = sp.cancel(up - (self.trace_K if i == j else 0))
        out = []
        for i in range(d):
            val = sp.Integer(0)
            for j in range(d):
                val += sp.diff(T[j, i], self.xs[j])
                for k in range(d):
                    val += G3[j, j, k] * T[k, i] - G3[k, j, i] * T[j, k]
            out.append(sp.cancel(val))
        return out

    @cached_property
    def normal_up(self) -> list[sp.Expr]:
        n_down = [-self.N] + [sp.Integer(0)] * self.d
        ginv = self.full.g_inv
        return [
            sp.cancel(sum(ginv[a, c] * n_down[c] for c in range(self.d + 1)))
            for a in range(self.d + 1)
        ]

    # ------------------------------------------------------------------ checks

    def check_background_nondegenerate(self) -> tuple[bool, str]:
        """Falsifier hygiene: every structural feature of the split must be
        switched on, otherwise the zero residues prove nothing."""
        feats = {
            "R3 != 0": not _zero(self.spatial.ricci_scalar),
            "trK != 0": not _zero(self.trace_K),
            "ham_form != 0": not _zero(self.hamiltonian_form),
            "some momentum_form != 0": any(not _zero(m) for m in self.momentum_form),
            "shift != 0": any(not _zero(b) for b in self.shift),
            "dN != 0": any(not _zero(sp.diff(self.N, c)) for c in self.coords),
            "dt h != 0": any(
                not _zero(sp.diff(self.h[i, j], self.t))
                for i in range(self.d)
                for j in range(self.d)
            ),
            "h off-diagonal": any(
                not _zero(self.h[i, j]) for i in range(self.d) for j in range(self.d) if i != j
            ),
        }
        ok = all(feats.values())
        return ok, "; ".join(f"{k}: {v}" for k, v in feats.items())

    def check_normal_gradient(self) -> tuple[bool, str]:
        """(A) K_ij(ADM formula) == +nabla_i n_j."""
        n_down = Array([-self.N] + [sp.Integer(0)] * self.d)
        grad = self.full.covariant_derivative(n_down, ["down"])  # [a, b] = nabla_a n_b
        ok = all(
            _zero(grad[i + 1, j + 1] - self.extrinsic[i, j])
            for i in range(self.d)
            for j in range(self.d)
        )
        return ok, "K_ij == +nabla_i n_j componentwise" if ok else "mismatch"

    def check_lagrangian_split(self) -> tuple[bool, str]:
        """(B) sqrt(-g) R == N sqrt(h)(R3 + KK - K^2) - 2 d_mu(sqrt(-g) v^mu)."""
        D = self.d + 1
        n_up = Array(self.normal_up)
        grad_up = self.full.covariant_derivative(n_up, ["up"])  # [a, b] = nabla_a n^b
        theta = sp.cancel(sum(grad_up[a, a] for a in range(D)))
        accel = [
            sp.cancel(sum(self.normal_up[a] * grad_up[a, b] for a in range(D))) for b in range(D)
        ]
        v = [sp.cancel(accel[a] - self.normal_up[a] * theta) for a in range(D)]
        dh = sp.cancel(sp.Matrix(self.h).det())
        div_over_sqrth = sum(
            sp.diff(self.N * v[a], self.coords[a])
            + self.N * v[a] * sp.diff(dh, self.coords[a]) / (2 * dh)
            for a in range(D)
        )
        bulk = self.N * (self.spatial.ricci_scalar + self.K_squared - self.trace_K**2)
        residue = self.N * self.full.ricci_scalar - bulk - (-2) * div_over_sqrth
        ok = _zero(residue)
        return ok, (
            "sqrt(-g)R - N sqrt(h)(R3+KK-K^2) + 2 d_mu(sqrt(-g) v^mu) == 0 (over sqrt h)"
            if ok
            else "split residue nonzero"
        )

    def check_hamiltonian_projection(self) -> tuple[bool, str]:
        """(C) 2 G_nn == R3 + K^2 - KK."""
        D = self.d + 1
        G = self.full.einstein
        n = self.normal_up
        gnn = sp.cancel(sum(G[a, c] * n[a] * n[c] for a in range(D) for c in range(D)))
        ok = _zero(2 * gnn - self.hamiltonian_form)
        return ok, "2 G_nn == R3 + trK^2 - K_ij K^ij" if ok else "Hamiltonian projection mismatch"

    def check_momentum_projection(self) -> tuple[bool, str]:
        """(D) G_(n, i) == D_j(K^j_i - delta^j_i K), each spatial i."""
        D = self.d + 1
        G = self.full.einstein
        n = self.normal_up
        for i in range(self.d):
            gni = sp.cancel(sum(G[a, i + 1] * n[a] for a in range(D)))
            if not _zero(gni - self.momentum_form[i]):
                return False, f"momentum projection mismatch at i={i}"
        return True, "G_(n,i) == D_j(K^j_i - delta^j_i trK) for every i"

    def check_lapse_euler_lagrange(self) -> tuple[bool, str]:
        """(E) dLambda/dN == R3 + trK^2 - KK, Lambda = N (R3 + KK - K^2).

        K is proportional to 1/N with an N-independent numerator (manifest in
        the ADM formula), so the symbolic-lapse bulk is obtained exactly by
        K -> K N / Ns."""
        Ns = sp.Symbol("Nlapse_symbolic", positive=True)
        scale = self.N / Ns
        trKs = self.trace_K * scale
        KKs = self.K_squared * scale**2
        R3 = self.spatial.ricci_scalar
        lam = Ns * (R3 + KKs - trKs**2)
        residue = sp.diff(lam, Ns) - (R3 + trKs**2 - KKs)
        ok = _zero(residue)
        return ok, (
            "EL_N of the bulk == Hamiltonian constraint density" if ok else "lapse EL mismatch"
        )

    def run_all(self) -> dict[str, tuple[bool, str]]:
        return {
            "background-nondegenerate": self.check_background_nondegenerate(),
            "extrinsic-curvature-normal-gradient": self.check_normal_gradient(),
            "lagrangian-split": self.check_lagrangian_split(),
            "hamiltonian-projection": self.check_hamiltonian_projection(),
            "momentum-projection": self.check_momentum_projection(),
            "lapse-euler-lagrange": self.check_lapse_euler_lagrange(),
        }


def adm_sample_1p2() -> ADMGeometry:
    """Deterministic nondegenerate 1+2 background: time-dependent curved
    slice, off-diagonal h, nonzero shift, nonconstant lapse. Small enough
    that every residue stays a fast exact rational computation."""
    t, x, y = sp.symbols("t x y")
    lapse = 1 + sp.Rational(1, 2) * x
    shift = [sp.Rational(1, 3) * y, sp.Integer(0)]
    h = sp.Matrix(
        [
            [1 + sp.Rational(1, 3) * t + sp.Rational(1, 4) * y, sp.Rational(1, 5) * x],
            [sp.Rational(1, 5) * x, 1 + sp.Rational(1, 3) * x],
        ]
    )
    return ADMGeometry(t, [x, y], lapse, shift, h)
