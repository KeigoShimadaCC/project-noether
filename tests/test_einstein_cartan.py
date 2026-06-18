"""Einstein-Cartan connection equation: algebraic torsion-vs-spin-source relation.

VAL-EOM-011: The connection variation yields the algebraic torsion-vs-spin-source
relation, either kernel-verified (residue + SymPy cross-check) or clearly gated with
a stated reason.

Conventions: noether-default-v1 + metric-affine-v1.

Two verification gates (dual-gate requirement, architecture.md section 3.2):
1. Cadabra residue check: the Palatini connection equation, after substituting
   G = LC + K(T), is algebraic in K (no partial-K terms).  The pure Palatini
   solution G = LC + projective mode satisfies the equation.
2. SymPy cross-check: the Palatini connection EOM on metric-compatible (Q=0)
   torsionful backgrounds is algebraic in the contortion K: the difference
   E(Gamma=LC+K) - E(Gamma=LC) equals the expected algebraic K expression
   with no derivative-of-K terms (einstein_cartan_algebraic_in_K_residual
   returns zero componentwise on multiple seeded backgrounds).
"""

from __future__ import annotations

import random

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.templates import get as get_template
from noether.kernels.sympy_kernel.geometry import (
    christoffel_of_metric,
    components,
    contortion_of_torsion,
    einstein_cartan_algebraic_in_K_residual,
    nonmetricity_of_connection,
    palatini_connection_eom,
    random_diagonal_metric,
    torsion_of_connection,
)

# ---------------------------------------------------------------------------
# Cadabra residue checks (require cadabra2 installed)
# Template registered in templates.py as 'ec_connection_algebraic_in_K'.
# ---------------------------------------------------------------------------

_EC_TEMPLATE_NAME = "ec_connection_algebraic_in_K"


def _run_ec_script():
    """Run the Einstein-Cartan connection script and return the ComputedResult."""
    script = get_template(_EC_TEMPLATE_NAME)
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="Einstein-Cartan connection equation check",
            payload={"script": script},
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
    """SymPy cross-check for Einstein-Cartan algebraic torsion relation.

    Dual-gate requirement (architecture.md section 3.2): the Cadabra residue
    check alone is insufficient because of the torsion trap.  These SymPy
    component checks verify the core physics claim on explicit backgrounds.
    """

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

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_palatini_eom_algebraic_in_K(self, seed):
        """The Palatini connection EOM is algebraic in K (no derivative-of-K
        terms) on metric-compatible torsionful backgrounds.

        This is the core SymPy cross-check for VAL-EOM-011.  The
        einstein_cartan_algebraic_in_K_residual computes the Palatini
        connection EOM for both Gamma = LC + K and Gamma = LC, then
        verifies the difference equals the expected algebraic K expression
        (no derivatives of K).  A zero residual confirms the EOM is
        algebraic in K, meaning torsion is algebraically determined by
        any spin source rather than propagating independently.
        """
        geom, gamma, q_zero, t_nonzero = _make_metric_compatible_torsionful_connection(seed, dim=3)
        assert q_zero, "Background must be metric-compatible (Q=0)"
        assert t_nonzero, "Background must have nonzero torsion"

        g_imm = sp.ImmutableDenseNDimArray(geom.g)
        g_inv_imm = sp.ImmutableDenseNDimArray(geom.g_inv)

        residual = einstein_cartan_algebraic_in_K_residual(
            geom.coords, gamma, g_imm, g_inv_imm
        )
        for c in components(residual):
            assert sp.simplify(c) == 0, (
                f"Algebraic-in-K residual nonzero on seed {seed}: "
                f"Palatini EOM has derivative-of-K terms (not algebraic)"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_palatini_eom_lc_solution(self, seed):
        """The Palatini connection EOM vanishes when Gamma = LC (plus
        projective mode) on metric-compatible torsionful backgrounds.

        This verifies the E(LC) = 0 condition: on a metric-compatible
        background, the Levi-Civita connection plus the projective mode
        (delta^lam_nu A_mu) is a solution of the Palatini connection
        equation.  This is the SymPy analogue of the Cadabra
        solution_zero check.
        """
        geom, gamma, q_zero, t_nonzero = _make_metric_compatible_torsionful_connection(seed, dim=3)
        assert q_zero and t_nonzero

        g_imm = sp.ImmutableDenseNDimArray(geom.g)
        g_inv_imm = sp.ImmutableDenseNDimArray(geom.g_inv)
        LC = christoffel_of_metric(geom.coords, g_imm, g_inv_imm)

        # Compute the Palatini EOM at Gamma = LC
        E_lc = palatini_connection_eom(geom.coords, LC, g_imm, g_inv_imm)

        # On a metric-compatible background, E(LC) should vanish
        # (LC is the Palatini solution up to projective mode)
        for c in components(E_lc):
            assert sp.simplify(c) == 0, (
                f"Palatini EOM at LC nonzero on seed {seed}: {c}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_K_has_coordinate_dependence(self, seed):
        """The contortion K on a diagonal metric with constant T has
        nonzero coordinate derivatives (since K involves g^{-1} and g),
        yet the Palatini EOM contains no derivative-of-K terms.

        This is a negative-control: it shows the algebraic-in-K
        property is non-trivial (K is NOT constant, so the absence
        of derivative-of-K terms in the EOM is a real cancellation).
        """
        geom, gamma, q_zero, t_nonzero = _make_metric_compatible_torsionful_connection(seed, dim=3)
        assert q_zero and t_nonzero

        g_imm = sp.ImmutableDenseNDimArray(geom.g)
        g_inv_imm = sp.ImmutableDenseNDimArray(geom.g_inv)
        K = contortion_of_torsion(gamma, g_imm, g_inv_imm)

        # Verify that K has nonzero coordinate derivatives
        n = len(geom.coords)
        has_dK = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    for a in range(n):
                        dK = sp.diff(K[lam, mu, nu], geom.coords[a])
                        if sp.simplify(dK) != 0:
                            has_dK = True
                            break
                    if has_dK:
                        break
                if has_dK:
                    break
            if has_dK:
                break
        assert has_dK, (
            f"Contortion K should have nonzero coordinate derivatives "
            f"on seed {seed} (makes the algebraic-in-K property non-trivial)"
        )
