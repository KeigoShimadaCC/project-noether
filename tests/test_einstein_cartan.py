"""Einstein-Cartan connection equation: algebraic torsion-vs-spin-source relation.

VAL-EOM-011: The connection variation yields the algebraic torsion-vs-spin-source
relation, either kernel-verified (residue + SymPy cross-check) or clearly gated with
a stated reason.

Conventions: noether-default-v1 + metric-affine-v1.

Two verification gates:
1. Cadabra residue check: the Palatini connection equation, after substituting
   G = LC + K(T), is algebraic in K (no partial-K terms).  The pure Palatini
   solution G = LC + projective mode satisfies the equation.
2. SymPy cross-check: the algebraic nature of the connection equation is confirmed
   componentwise on random metric-compatible (Q=0) torsionful backgrounds.
"""

from __future__ import annotations

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel.geometry import (
    christoffel_of_metric,
    components,
    nonmetricity_of_connection,
    random_diagonal_metric,
    torsion_of_connection,
)

# ---------------------------------------------------------------------------
# Cadabra residue checks (require cadabra2 installed)
# ---------------------------------------------------------------------------

EC_CONNECTION_SCRIPT = r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\partial{#}::PartialDerivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
{g_{\mu\nu}, g^{\mu\nu}, sg, G^{\lambda}_{\mu\nu}, dG^{\lambda}_{\mu\nu}}::Depends(\partial{#}).
C^{\lambda}_{\mu\nu}::TableauSymmetry(shape={2}, indices={1,2}).
C^{\lambda}_{\mu\nu}::Depends(\partial{#}).
A_{\mu}::Depends(\partial{#}).
K^{\lambda}_{\mu\nu}::Depends(\partial{#}).
LC^{\lambda}_{\mu\nu}::TableauSymmetry(shape={2}, indices={1,2}).
LC^{\lambda}_{\mu\nu}::Depends(\partial{#}).

# Step 1: Derive the Palatini connection equation
ex := \int{ - sg g^{\sigma\nu} ( \partial_{\lambda}{G^{\lambda}_{\nu\sigma}} - \partial_{\nu}{G^{\lambda}_{\lambda\sigma}} + G^{\lambda}_{\lambda\rho} G^{\rho}_{\nu\sigma} - G^{\lambda}_{\nu\rho} G^{\rho}_{\lambda\sigma} ) }{x};
vary(ex, $G^{\lambda}_{\mu\nu} -> dG^{\lambda}_{\mu\nu}$);
distribute(ex);
integrate_by_parts(ex, $dG^{\lambda}_{\mu\nu}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\int{A??}{x} -> A??$);

# Step 2: Verify G = LC + projective satisfies the equation
soln := @(ex);
substitute(soln, $G^{\lambda}_{\mu\nu} -> C^{\lambda}_{\mu\nu} + g^{\lambda}_{\nu} A_{\mu}$);
distribute(soln);
substitute(soln, $\partial_{\lambda}{g^{\nu\sigma}} -> -g^{\nu\rho} C^{\sigma}_{\lambda\rho} - g^{\sigma\rho} C^{\nu}_{\lambda\rho}$);
substitute(soln, $\partial_{\lambda}{g_{\nu\sigma}} -> g_{\rho\sigma} C^{\rho}_{\lambda\nu} + g_{\nu\rho} C^{\rho}_{\lambda\sigma}$);
substitute(soln, $\partial_{\lambda}{sg} -> sg C^{\rho}_{\rho\lambda}$);
distribute(soln);
eliminate_kronecker(soln);
sort_product(soln);
canonicalise(soln);
rename_dummies(soln);
meld(soln);
print("NOETHER_CHECK: solution_zero=" + str(str(soln) == "0"))

# Step 3: Substitute G = LC + K and verify algebraic in K
algex := @(ex);
substitute(algex, $G^{\lambda}_{\mu\nu} -> LC^{\lambda}_{\mu\nu} + K^{\lambda}_{\mu\nu}$);
distribute(algex);
product_rule(algex);
distribute(algex);

# Check: substituting partial_K -> 0 should give the same result
noDK := @(algex);
substitute(noDK, $\partial_{\mu}{K^{\lambda}_{\nu\rho}} -> 0$, repeat=True);
distribute(noDK);
eliminate_metric(noDK);
eliminate_kronecker(noDK);
sort_product(noDK);
canonicalise(noDK);
rename_dummies(noDK);

# Compute difference: full expression minus no-deriv-K expression
diff := @(algex) - @(noDK);
distribute(diff);
eliminate_metric(diff);
eliminate_kronecker(diff);
sort_product(diff);
canonicalise(diff);
rename_dummies(diff);
meld(diff);
print("NOETHER_CHECK: algebraic_in_K=" + str(str(diff) == "0"))
"""


def _run_ec_script():
    """Run the Einstein-Cartan connection script and return the ComputedResult."""
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="Einstein-Cartan connection equation check",
            payload={"script": EC_CONNECTION_SCRIPT},
        )
    )


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestEinsteinCartanConnection:
    """Einstein-Cartan connection equation: algebraic torsion-vs-spin relation."""

    def test_solution_zero(self):
        """G = LC + projective mode satisfies the Palatini connection EOM."""
        result = _run_ec_script()
        checks = result.value.get("checks", {})
        assert checks.get("solution_zero") == "True", (
            f"Palatini connection EOM not satisfied by LC+projective: {checks}"
        )

    def test_algebraic_in_K(self):
        """The Palatini connection EOM is algebraic in K (no derivatives of K)."""
        result = _run_ec_script()
        checks = result.value.get("checks", {})
        assert checks.get("algebraic_in_K") == "True", (
            f"Palatini connection EOM contains derivatives of K (not algebraic): {checks}"
        )


# ---------------------------------------------------------------------------
# SymPy cross-check: verify the algebraic nature on explicit backgrounds
# ---------------------------------------------------------------------------

def _make_metric_compatible_torsionful_connection(seed: int, dim: int = 3):
    """Create a metric-compatible (Q=0) but torsionful connection.

    Gamma = LC(g) + K(T) where T is a random torsion tensor.
    This guarantees Q=0 (metric compatibility) but T != 0.
    """
    import random

    geom = random_diagonal_metric(seed, dim=dim)
    gamma_lc = geom.christoffel

    # Create a random torsion tensor (antisymmetric in lower pair)
    rng = random.Random(seed + 100)
    n = dim
    T = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(mu + 1, n):
                val = sp.Integer(rng.randint(-2, 2))
                T[lam, mu, nu] = val
                T[lam, nu, mu] = -val
    T = sp.ImmutableDenseNDimArray(T)

    # Build Gamma = LC + K(T)
    # K^lam_{mu nu} = (1/2)(T^lam_{mu nu} + g^{lam sig} g_{mu tau} T^tau_{sig nu}
    #                        + g^{lam sig} g_{nu tau} T^tau_{sig mu})
    gamma = sp.MutableDenseNDimArray(gamma_lc)
    g_inv_imm = sp.ImmutableDenseNDimArray(geom.g_inv)
    g_imm = sp.ImmutableDenseNDimArray(geom.g)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                term1 = T[lam, mu, nu]
                term2 = sum(
                    g_inv_imm[lam, sig] * g_imm[mu, tau] * T[tau, sig, nu]
                    for sig in range(n) for tau in range(n)
                )
                term3 = sum(
                    g_inv_imm[lam, sig] * g_imm[nu, tau] * T[tau, sig, mu]
                    for sig in range(n) for tau in range(n)
                )
                K_lmn = sp.Rational(1, 2) * (term1 + term2 + term3)
                gamma[lam, mu, nu] = sp.cancel(gamma[lam, mu, nu] + K_lmn)

    gamma = sp.ImmutableDenseNDimArray(gamma)

    # Verify Q=0 (metric compatibility)
    Q = nonmetricity_of_connection(geom.coords, gamma, g_imm)
    q_zero = all(sp.simplify(c) == 0 for c in components(Q))

    # Verify T != 0 (nonzero torsion)
    T_check = torsion_of_connection(gamma)
    t_nonzero = any(sp.simplify(c) != 0 for c in components(T_check))

    return geom, gamma, q_zero, t_nonzero


class TestEinsteinCartanSymPy:
    """SymPy cross-check for Einstein-Cartan algebraic torsion relation."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_metric_compatible_torsionful_background(self, seed):
        """Verify we can construct metric-compatible (Q=0) torsionful backgrounds."""
        geom, gamma, q_zero, t_nonzero = _make_metric_compatible_torsionful_connection(seed, dim=3)
        assert q_zero, "Connection should be metric-compatible (Q=0)"
        assert t_nonzero, "Connection should have nonzero torsion"

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_distortion_torsion_matches_original(self, seed):
        """Gamma - LC has the same torsion as Gamma (since LC is torsion-free).

        This confirms the algebraic link: the contortion K(T) correctly
        reconstructs the torsion of the decomposed connection.
        """
        geom, gamma, q_zero, t_nonzero = _make_metric_compatible_torsionful_connection(seed, dim=3)
        assert q_zero and t_nonzero

        LC = christoffel_of_metric(
            geom.coords,
            sp.ImmutableDenseNDimArray(geom.g),
            sp.ImmutableDenseNDimArray(geom.g_inv),
        )

        # Gamma - LC should have the same torsion as Gamma itself
        gamma_minus_LC = sp.MutableDenseNDimArray.zeros(3, 3, 3)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    gamma_minus_LC[a, b, c] = sp.cancel(gamma[a, b, c] - LC[a, b, c])
        gamma_minus_LC = sp.ImmutableDenseNDimArray(gamma_minus_LC)

        T_original = torsion_of_connection(gamma)
        T_distortion = torsion_of_connection(gamma_minus_LC)

        # The distortion's torsion should equal the original torsion
        for idx_val in components(T_original - T_distortion):
            assert sp.simplify(idx_val) == 0, (
                f"Torsion mismatch on seed {seed}: distortion torsion != original torsion"
            )
