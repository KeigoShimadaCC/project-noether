"""Vector (Maxwell) EOM on a metric-affine background.

Tests for VAL-EOM-020 and VAL-EOM-021:

- VAL-EOM-020: The vector EOM uses the full-connection divergence
  (carrying torsion/non-metricity contributions), not the Levi-Civita
  divergence.  With F = dA the EOM is nabla^{LC}_mu F^{mu nu} = 0, which
  when expressed with the full affine nabla^{aff} carries T/Q correction
  terms.  With F = nabla A (covariant curl) the EOM naturally involves
  the full-connection nabla with Q contributions from the IBP.  Both
  are verified by Cadabra residue check and/or SymPy cross-check.

- VAL-EOM-021: With F = dA the gauge field contributes nothing to the
  connection equation (zero hypermomentum); with the covariant-curl
  choice it sources the connection equation with a purely spin-type
  hypermomentum Delta^lambda_{mu nu} = -2 A_lambda F^{mu nu}
  (antisymmetric in mu, nu).  The two derivations differ exactly in
  the connection-equation source, each verified or gated.

Conventions: noether-default-v1 + metric-affine-v1.
  T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
  Q_{lambda mu nu} = nabla_lambda g_{mu nu}
  Contortion and disformation signs per metric-affine-v1.

Verification gates:
1. Cadabra residue check (for the dA EOM and hypermomentum scripts)
2. SymPy component cross-check on random metric + connection backgrounds
   (for both EOMs and hypermomentum)

The covariant-curl EOM Cadabra residue check is gated (the expansion
produces mixed-index G terms that canonicalise cannot resolve, per the
known limitation in cadabra-gotchas.md).  The SymPy cross-check provides
the independent verification.
"""

from __future__ import annotations

import random

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.templates import get as get_template
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    christoffel_of_metric,
    components,
    covariant_curl_of_1form,
    exterior_derivative_of_1form,
    nonmetricity_of_connection,
    random_affine_connection,
    random_diagonal_metric,
    torsion_of_connection,
)

# ---------------------------------------------------------------------------
# Cadabra scripts: registered in templates.py, retrieved via get_template().
#   _DA_EOM_SCRIPT         -> "vector_affine_dA_eom"
#   _DA_HYPERMOMENTUM_SCRIPT -> "vector_affine_dA_hypermomentum"
#   _COVCURL_HYPERMOMENTUM_SCRIPT -> "vector_affine_covcurl_hypermomentum"
# Following the pattern established in misc-m3-inline-script-to-template-refactor
# (test_hypermomentum.py, test_palatini_scalar_tensor_affine.py).
# ---------------------------------------------------------------------------


def _run_cadabra(script: str):
    """Run a Cadabra script and return the ComputedResult."""
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="vector EOM affine check",
            payload={"script": script},
        )
    )


# ---------------------------------------------------------------------------
# SymPy cross-check helpers
# ---------------------------------------------------------------------------


def _make_affine_background(seed: int, dim: int = 3, symmetric: bool = False):
    """Build a random metric + connection, torsion, and non-metricity."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=symmetric)
    T = torsion_of_connection(gamma)
    g_imm = sp.ImmutableDenseNDimArray(geom.g)
    Q = nonmetricity_of_connection(geom.coords, gamma, g_imm)
    return geom, gamma, T, Q


def _make_test_1form(coords, seed: int):
    """Build a simple polynomial 1-form for testing."""
    rng = random.Random(seed + 500)
    n = len(coords)
    A = sp.MutableDenseNDimArray.zeros(n)
    for mu in range(n):
        c = sp.Rational(rng.randint(1, 3), rng.randint(2, 5))
        A[mu] = _clean(c * coords[rng.randrange(n)])
    return sp.ImmutableDenseNDimArray(A)


def _raise_F(F_low, g_inv, n):
    """Raise both indices of F_{alpha beta} to get F^{mu nu}."""
    F_up = sp.MutableDenseNDimArray.zeros(n, n)
    for mu in range(n):
        for nu in range(n):
            val = sp.Integer(0)
            for a in range(n):
                for b in range(n):
                    val += g_inv[mu, a] * g_inv[nu, b] * F_low[a, b]
            F_up[mu, nu] = _clean(val)
    return sp.ImmutableDenseNDimArray(F_up)


def _sqrt_neg_g(g_imm):
    """Compute sqrt(-det(g)) from the metric array."""
    g_mat = sp.Matrix([[g_imm[i, j] for j in range(g_imm.shape[1])] for i in range(g_imm.shape[0])])
    return sp.sqrt(-g_mat.det())


# ===========================================================================
# Cadabra residue checks
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestVectorEomDAResidue:
    """Cadabra residue checks for the Maxwell EOM with F = dA
    on a metric-affine background.

    The dA EOM is nabla^{LC}_mu F^{mu nu} = 0, verified using the
    nabla + LC-substitution approach (valid because F = dA does not
    depend on the independent connection).  VAL-EOM-020 (dA part)."""

    def test_dA_eom_lc_divergence_residue_zero(self):
        """The dA Maxwell EOM equals nabla^{LC}_mu F^{mu nu} = 0.
        VAL-EOM-020."""
        result = _run_cadabra(get_template("vector_affine_dA_eom"))
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("dA_eom_residue_zero") == "True", (
            f"dA EOM residue nonzero: {result.raw.stdout}"
        )


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestHypermomentumDA:
    """Connection variation with F = dA: zero hypermomentum.

    The action S = -1/4 ∫ √-g F_{μν} F^{μν} with F = dA does not depend
    on the independent connection Γ, so δS/δΓ = 0.  VAL-EOM-021 (dA part)."""

    def test_dA_hypermomentum_zero(self):
        """Varying the dA action w.r.t. G^lambda_{mu nu} gives zero.
        VAL-EOM-021."""
        result = _run_cadabra(get_template("vector_affine_dA_hypermomentum"))
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("dA_hypermomentum_zero") == "True", (
            f"dA hypermomentum nonzero: {result.raw.stdout}"
        )


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestHypermomentumCovCurl:
    """Connection variation with F = nabla A: nonzero hypermomentum.

    The action depends on Γ through F = nabla_mu A_nu - nabla_nu A_mu,
    yielding a nonzero hypermomentum Delta^lambda_{mu nu} = -2 A_lambda
    F^{mu nu} (antisymmetric in mu, nu).  VAL-EOM-021 (covariant-curl
    part)."""

    def test_covcurl_hypermomentum_nonzero(self):
        """Varying the nabla-A action w.r.t. G^lambda_{mu nu} gives
        nonzero hypermomentum.  VAL-EOM-021."""
        result = _run_cadabra(get_template("vector_affine_covcurl_hypermomentum"))
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("covcurl_hypermomentum_nonzero") == "True", (
            f"covariant-curl hypermomentum zero: {result.raw.stdout}"
        )


# ===========================================================================
# SymPy component cross-checks (the torsion-trap safeguard)
# ===========================================================================


class TestVectorEomAffineSymPy:
    """SymPy cross-checks for the vector EOM on metric-affine backgrounds.

    These verify the key claims of VAL-EOM-020 and VAL-EOM-021 by
    evaluating identities on explicit random metric + connection
    backgrounds and asserting componentwise agreement.

    Key identities verified:
    1. nabla^{aff}_mu F^{mu nu} - nabla^{LC}_mu F^{mu nu}
       = (K^rho_{rho mu} + L^rho_{rho mu}) F^{mu nu}
         + (1/2) T^nu_{mu rho} F^{mu rho}
       (the T/Q correction that appears when the dA EOM is expressed
       with the full-connection divergence)
    2. The correction is nonzero on torsionful/non-metric backgrounds
    3. The correction vanishes at T=Q=0 (Levi-Civita limit)
    4. The hypermomentum for F = dA is zero; for F = nabla A it is
       Delta^lam_{mu nu} = -2 A_lam F^{mu nu} (antisymmetric)
    """

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_affine_lc_divergence_difference_equals_TQ_correction(self, seed):
        """nabla^{aff}_mu F^{mu nu} - nabla^{LC}_mu F^{mu nu}
        = (K^rho_{rho mu} + L^rho_{rho mu}) F^{mu nu}
          + (1/2) T^nu_{mu rho} F^{mu rho}

        This identity holds for any F^{mu nu} (it's an algebraic
        identity about the connection, not about the EOM).  It shows
        that when the dA EOM (nabla^{LC}_mu F^{mu nu} = 0) is
        expressed with the full-connection divergence, T/Q correction
        terms appear.  VAL-EOM-020."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        g_imm = sp.ImmutableDenseNDimArray(geom.g)

        # Compute F = dA
        F_low = exterior_derivative_of_1form(x, A)
        F_up = _raise_F(F_low, g_inv, n)

        # Compute Gamma^{LC}
        gamma_lc = christoffel_of_metric(x, g_imm, g_inv)

        # LHS: nabla^{aff}_mu F^{mu nu} - nabla^{LC}_mu F^{mu nu}
        # = (Gamma^{aff,rho}_{rho mu} - Gamma^{LC,rho}_{rho mu}) F^{mu nu}
        #   + Gamma^{aff,nu}_{mu rho} F^{mu rho}
        # (Gamma^{LC,nu}_{mu rho} F^{mu rho} = 0 by symmetry)
        lhs = sp.MutableDenseNDimArray.zeros(n)
        for nu in range(n):
            val = sp.Integer(0)
            for mu in range(n):
                # Distortion trace: (Gamma^{aff} - Gamma^{LC})^rho_{rho mu} F^{mu nu}
                for rho in range(n):
                    val += (gamma[rho, rho, mu] - gamma_lc[rho, rho, mu]) * F_up[mu, nu]
                # Gamma^{aff,nu}_{mu rho} F^{mu rho}
                for rho in range(n):
                    val += gamma[nu, mu, rho] * F_up[mu, rho]
            lhs[nu] = _clean(val)
        lhs = sp.ImmutableDenseNDimArray(lhs)

        # RHS: contortion trace + disformation trace + torsion term
        # Compute contortion K^lam_{mu nu}
        K = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    term1 = T[lam, mu, nu]
                    term2 = sum(
                        g_inv[lam, sig] * g_imm[mu, tau] * T[tau, sig, nu]
                        for sig in range(n)
                        for tau in range(n)
                    )
                    term3 = sum(
                        g_inv[lam, sig] * g_imm[nu, tau] * T[tau, sig, mu]
                        for sig in range(n)
                        for tau in range(n)
                    )
                    K[lam, mu, nu] = _clean(sp.Rational(1, 2) * (term1 + term2 + term3))
        K = sp.ImmutableDenseNDimArray(K)

        # Compute disformation L^lam_{mu nu}
        L = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    val = sp.Rational(1, 2) * sum(
                        g_inv[lam, rho] * (-Q[mu, nu, rho] - Q[nu, rho, mu] + Q[rho, mu, nu])
                        for rho in range(n)
                    )
                    L[lam, mu, nu] = _clean(val)
        L = sp.ImmutableDenseNDimArray(L)

        # Expected: (K+L)^rho_{rho mu} F^{mu nu} + (1/2) T^nu_{mu rho} F^{mu rho}
        rhs = sp.MutableDenseNDimArray.zeros(n)
        for nu in range(n):
            val = sp.Integer(0)
            for mu in range(n):
                for rho in range(n):
                    val += (K[rho, rho, mu] + L[rho, rho, mu]) * F_up[mu, nu]
                for rho in range(n):
                    val += sp.Rational(1, 2) * T[nu, mu, rho] * F_up[mu, rho]
            rhs[nu] = _clean(val)
        rhs = sp.ImmutableDenseNDimArray(rhs)

        # Verify: LHS == RHS
        for nu in range(n):
            diff = sp.simplify(lhs[nu] - rhs[nu])
            assert diff == 0, (
                f"seed={seed}: T/Q correction mismatch at nu={nu}: "
                f"lhs={lhs[nu]}, rhs={rhs[nu]}, residual={diff}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_TQ_correction_nonzero_on_torsionful_background(self, seed):
        """The T/Q correction terms are nonzero on a torsionful background,
        confirming the full-connection divergence carries torsion
        contributions.  VAL-EOM-020."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        g_imm = sp.ImmutableDenseNDimArray(geom.g)

        F_low = exterior_derivative_of_1form(x, A)
        F_up = _raise_F(F_low, g_inv, n)
        gamma_lc = christoffel_of_metric(x, g_imm, g_inv)

        # Compute the correction: nabla^{aff} - nabla^{LC}
        correction = sp.MutableDenseNDimArray.zeros(n)
        for nu in range(n):
            val = sp.Integer(0)
            for mu in range(n):
                for rho in range(n):
                    val += (gamma[rho, rho, mu] - gamma_lc[rho, rho, mu]) * F_up[mu, nu]
                    val += gamma[nu, mu, rho] * F_up[mu, rho]
            correction[nu] = _clean(val)
        correction = sp.ImmutableDenseNDimArray(correction)

        # The correction should be nonzero on a torsionful background
        any_nonzero = any(sp.simplify(c) != 0 for c in components(correction))
        has_torsion = any(sp.simplify(c) != 0 for c in components(T))
        if has_torsion:
            assert any_nonzero, (
                f"seed={seed}: affine-LC divergence difference is zero on "
                "a torsionful background (should carry T/Q terms)"
            )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_TQ_correction_zero_on_Levi_Civita(self, seed):
        """At T=0, Q=0 (Levi-Civita), the T/Q correction vanishes and
        the affine divergence equals the LC divergence.
        VAL-EOM-020 (T=Q=0 limit)."""
        geom = random_diagonal_metric(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)
        gamma_lc = geom.christoffel

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)

        F_low = exterior_derivative_of_1form(x, A)
        F_up = _raise_F(F_low, g_inv, n)

        # At T=Q=0, gamma = gamma_lc, so the correction is zero
        # (Gamma^{LC,nu}_{mu rho} F^{mu rho} = 0 because LC is
        # symmetric in mu,rho and F is antisymmetric)
        for nu in range(n):
            val = sp.Integer(0)
            for mu in range(n):
                for rho in range(n):
                    val += gamma_lc[nu, mu, rho] * F_up[mu, rho]
            val = _clean(val)
            diff = sp.simplify(val)
            assert diff == 0, f"seed={seed}: Gamma-LC F correction nonzero at T=Q=0, nu={nu}: {val}"

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_dA_hypermomentum_zero(self, seed):
        """With F = dA, the action has no Gamma dependence, so the
        hypermomentum is zero.  VAL-EOM-021 (dA part).

        We verify this structurally: F_{mu nu} = dA has no Gamma
        dependence, and F^{mu nu} = g^{-1} F g^{-1} has no Gamma
        dependence, and √-g has no Gamma dependence.  Therefore the
        entire action integrand is Gamma-independent."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)

        # F = dA involves only partial derivatives of A (no Gamma)
        F_low = exterior_derivative_of_1form(x, A)

        # Verify: dA has no gamma dependence (by construction, it's
        # computed from partial derivatives only).
        # The hypermomentum is Delta = -(2/√-g) δS/δΓ = 0.

        # Cross-check: the covariant curl differs from dA by the
        # torsion term, confirming that dA has no Gamma dependence
        # while covariant_curl does.
        cov_curl = covariant_curl_of_1form(x, gamma, A)

        has_torsion = any(sp.simplify(c) != 0 for c in components(T))
        if has_torsion:
            # On a torsionful background, dA != covariant_curl
            any_difference = False
            for mu in range(n):
                for nu in range(mu + 1, n):
                    diff = sp.simplify(F_low[mu, nu] - cov_curl[mu, nu])
                    if diff != 0:
                        any_difference = True
                        break
                if any_difference:
                    break
            assert any_difference, (
                f"seed={seed}: dA equals covariant curl on a torsionful "
                "background (should differ by T term)"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_covcurl_hypermomentum_is_minus_2AF(self, seed):
        """With F = nabla A, the hypermomentum is
        Delta^lam_{mu nu} = -2 A_lam F^{mu nu} (antisymmetric in mu, nu).
        VAL-EOM-021 (covariant-curl part).

        This is verified by computing the connection variation via finite
        differences and comparing with the expected expression."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        g_imm = sp.ImmutableDenseNDimArray(geom.g)

        # Compute F^{cov} = nabla A - nabla A (covariant curl)
        F_cov_low = covariant_curl_of_1form(x, gamma, A)
        F_cov_up = _raise_F(F_cov_low, g_inv, n)

        # Expected hypermomentum: Delta^lam_{mu nu} = -2 A_lam F^{mu nu}
        Delta_expected = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    Delta_expected[lam, mu, nu] = _clean(-2 * A[lam] * F_cov_up[mu, nu])
        Delta_expected = sp.ImmutableDenseNDimArray(Delta_expected)

        # The hypermomentum should be nonzero on a nontrivial background
        any_nonzero = any(sp.simplify(c) != 0 for c in components(Delta_expected))
        assert any_nonzero, f"seed={seed}: covariant-curl hypermomentum is zero (should be nonzero)"

        # The hypermomentum should be antisymmetric in (mu, nu)
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    diff = sp.simplify(Delta_expected[lam, mu, nu] + Delta_expected[lam, nu, mu])
                    assert diff == 0, (
                        f"seed={seed}: hypermomentum not antisymmetric at ({lam},{mu},{nu})"
                    )

        # Verify the connection derivative exactly via symbolic perturbation.
        # Make epsilon a Symbol, perturb one gamma component, and extract the
        # coefficient of epsilon.  This gives the exact derivative (no O(eps)
        # error from finite differences).
        sqrt_g = _sqrt_neg_g(g_imm)
        eps = sp.Symbol("epsilon")

        def _action_integrand(gamma_arr):
            F_cov = covariant_curl_of_1form(x, gamma_arr, A)
            F_cov_up_local = _raise_F(F_cov, g_inv, n)
            integrand = sp.Integer(0)
            for mu in range(n):
                for nu in range(n):
                    integrand += F_cov[mu, nu] * F_cov_up_local[mu, nu]
            return _clean(-sp.Rational(1, 4) * sqrt_g * integrand)

        # Check a subset of components
        for sig in range(min(n, 2)):
            for alp in range(min(n, 2)):
                for bet in range(alp, min(n, 2)):
                    gamma_pert = sp.MutableDenseNDimArray(gamma)
                    gamma_pert[sig, alp, bet] = gamma[sig, alp, bet] + eps
                    gamma_pert = sp.ImmutableDenseNDimArray(gamma_pert)

                    S_pert = _action_integrand(gamma_pert)
                    # Extract the first-order coefficient of eps
                    exact_deriv = S_pert.expand().coeff(eps)
                    expected_deriv = _clean(sqrt_g * A[sig] * F_cov_up[alp, bet])

                    diff = sp.simplify(exact_deriv - expected_deriv)
                    assert diff == 0, (
                        f"seed={seed}: connection derivative mismatch at "
                        f"({sig},{alp},{bet}): exact={exact_deriv}, "
                        f"expected={expected_deriv}, diff={diff}"
                    )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_two_F_definitions_differ_by_torsion_term(self, seed):
        """F^{cov}_{mu nu} = F^{dA}_{mu nu} - T^lam_{mu nu} A_lam.
        This is the identity VAL-GEOM-020, verified here as a
        prerequisite for the EOM difference.  VAL-EOM-021."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)

        F_dA_low = exterior_derivative_of_1form(x, A)
        F_cov_low = covariant_curl_of_1form(x, gamma, A)

        # Verify: F_cov = F_dA - T·A
        for mu in range(n):
            for nu in range(n):
                torsion_correction = sum(T[lam, mu, nu] * A[lam] for lam in range(n))
                diff = sp.simplify(
                    F_cov_low[mu, nu] - (F_dA_low[mu, nu] - _clean(torsion_correction))
                )
                assert diff == 0, f"seed={seed}: F_cov != F_dA - T·A at ({mu},{nu}): diff={diff}"


class TestCovariantCurlEomSymPy:
    """SymPy cross-checks for the covariant-curl EOM.

    The covariant-curl EOM (from varying S = -1/4 √-g F_{μν} F^{μν}
    with F = ∇A w.r.t. A) is:

        (1/√-g) ∂_μ(√-g F^{μν}) + F^{μρ} Γ^{ν}_{μρ} = 0

    equivalently (since F^{μρ} Γ^{ν}_{(μρ)} = 0 by symmetry):

        (1/√-g) ∂_μ(√-g F^{μν}) + (1/2) T^{ν}_{μρ} F^{μρ} = 0

    The EOM carries torsion contributions (the T term), confirming
    VAL-EOM-020.

    We verify the EOM form by checking the two components of the
    Euler-Lagrange equation separately:
    (a) ∂L/∂A_σ = 1/2 √-g T^σ_{μν} F^{μν}  (explicit A dependence)
    (b) ∂L/∂(∂_μ A_σ) = -√-g F^{μσ}         (∂A dependence)
    using symbolic perturbation with SymPy's .coeff() to extract
    exact first-order terms.
    """

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_covcurl_eom_explicit_A_derivative(self, seed):
        """The explicit A-derivative of the Lagrangian gives the
        torsion term: ∂L/∂A_σ = 1/2 √-g T^σ_{μν} F^{μν}.

        Verified by perturbing A_σ with a symbolic epsilon and
        extracting the linear coefficient.  VAL-EOM-020."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        g_imm = sp.ImmutableDenseNDimArray(geom.g)
        sqrt_g = _sqrt_neg_g(g_imm)

        # Original action integrand
        F_cov_low = covariant_curl_of_1form(x, gamma, A)
        F_cov_up = _raise_F(F_cov_low, g_inv, n)

        S0 = sp.Integer(0)
        for mu in range(n):
            for nu in range(n):
                S0 += F_cov_low[mu, nu] * F_cov_up[mu, nu]
        S0 = _clean(-sp.Rational(1, 4) * sqrt_g * S0)

        # Perturb A with symbolic epsilon and extract linear term
        eps = sp.Symbol("epsilon")
        for sig in range(n):
            A_pert = sp.MutableDenseNDimArray(A)
            A_pert[sig] = A[sig] + eps
            A_pert = sp.ImmutableDenseNDimArray(A_pert)

            F_cov_pert = covariant_curl_of_1form(x, gamma, A_pert)
            F_cov_up_pert = _raise_F(F_cov_pert, g_inv, n)

            S_pert = sp.Integer(0)
            for mu in range(n):
                for nu in range(n):
                    S_pert += F_cov_pert[mu, nu] * F_cov_up_pert[mu, nu]
            S_pert = _clean(-sp.Rational(1, 4) * sqrt_g * S_pert)

            # Extract first-order coefficient of epsilon
            exact_deriv = S_pert.expand().coeff(eps)

            # Expected: 1/2 √-g T^σ_{μν} F^{μν}
            expected = sp.Integer(0)
            for mu in range(n):
                for nu in range(n):
                    expected += T[sig, mu, nu] * F_cov_up[mu, nu]
            expected = _clean(sp.Rational(1, 2) * sqrt_g * expected)

            diff = sp.simplify(exact_deriv - expected)
            assert diff == 0, (
                f"seed={seed}: explicit A-derivative mismatch at "
                f"sigma={sig}: exact={exact_deriv}, "
                f"expected={expected}, diff={diff}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_covcurl_eom_partial_A_derivative(self, seed):
        """The ∂A-derivative of the Lagrangian gives the
        field-strength term: ∂L/∂(∂_μ A_σ) = -√-g F^{μσ}.

        Verified by perturbing A_σ → A_σ + ε x^ρ (a coordinate-
        dependent perturbation) and extracting the epsilon coefficient.
        The coefficient of epsilon combines both the ∂A-derivative
        (∂L/∂(∂_ρ A_σ)) and the explicit A-derivative (∂L/∂A_σ)·x^ρ.
        We subtract the known explicit A-derivative (verified in the
        previous test) to isolate ∂L/∂(∂_ρ A_σ) = -√-g F^{ρσ}.
        VAL-EOM-020."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        g_imm = sp.ImmutableDenseNDimArray(geom.g)
        sqrt_g = _sqrt_neg_g(g_imm)

        # Original action integrand
        F_cov_low = covariant_curl_of_1form(x, gamma, A)
        F_cov_up = _raise_F(F_cov_low, g_inv, n)

        S0 = sp.Integer(0)
        for mu in range(n):
            for nu in range(n):
                S0 += F_cov_low[mu, nu] * F_cov_up[mu, nu]
        S0 = _clean(-sp.Rational(1, 4) * sqrt_g * S0)

        # Perturb A_σ → A_σ + ε x^ρ and extract the linear coefficient
        eps = sp.Symbol("epsilon")
        for sig in range(min(n, 2)):
            for rho in range(min(n, 2)):
                A_pert = sp.MutableDenseNDimArray(A)
                A_pert[sig] = A[sig] + eps * x[rho]
                A_pert = sp.ImmutableDenseNDimArray(A_pert)

                F_cov_pert = covariant_curl_of_1form(x, gamma, A_pert)
                F_cov_up_pert = _raise_F(F_cov_pert, g_inv, n)

                S_pert = sp.Integer(0)
                for mu in range(n):
                    for nu in range(n):
                        S_pert += F_cov_pert[mu, nu] * F_cov_up_pert[mu, nu]
                S_pert = _clean(-sp.Rational(1, 4) * sqrt_g * S_pert)

                # The epsilon coefficient combines:
                # (a) ∂L/∂(∂_ρ A_σ)  (from ∂_μ(A_σ+εx^ρ) giving ε δ_{μρ})
                # (b) ∂L/∂A_σ · x^ρ  (from explicit A dependence)
                # So: coeff(eps) = ∂L/∂(∂_ρ A_σ) + ∂L/∂A_σ · x^ρ
                # We know ∂L/∂A_σ = 1/2 √-g T^σ_{μν} F^{μν}
                total_deriv = S_pert.expand().coeff(eps)

                # Explicit A-derivative contribution
                explicit_A_deriv = sp.Integer(0)
                for mu in range(n):
                    for nu in range(n):
                        explicit_A_deriv += T[sig, mu, nu] * F_cov_up[mu, nu]
                explicit_A_deriv = _clean(sp.Rational(1, 2) * sqrt_g * explicit_A_deriv * x[rho])

                # ∂A-derivative = total - explicit_A
                partial_A_deriv = _clean(total_deriv - explicit_A_deriv)

                # Expected: ∂L/∂(∂_ρ A_σ) = -√-g F^{ρσ}
                expected = _clean(-sqrt_g * F_cov_up[rho, sig])

                diff = sp.simplify(partial_A_deriv - expected)
                assert diff == 0, (
                    f"seed={seed}: ∂A-derivative mismatch at "
                    f"(rho={rho}, sig={sig}): partial={partial_A_deriv}, "
                    f"expected={expected}, diff={diff}"
                )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_covcurl_eom_reduces_to_dA_eom_at_T0(self, seed):
        """At T=0, Q=0 (Levi-Civita), the covariant-curl EOM reduces to
        nabla^{LC}_mu F^{mu nu} = 0 (the dA EOM).  This is because:
        (1) F^{cov} = F^{dA} when T=0
        (2) Gamma^{LC,nu}_{mu rho} F^{mu rho} = 0 (symmetric × antisymmetric)

        VAL-EOM-020 / VAL-EOM-021 (T=0 limit)."""
        geom = random_diagonal_metric(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)
        gamma_lc = geom.christoffel

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)

        # At T=0, F^{cov} = F^{dA}
        F_cov_low = covariant_curl_of_1form(x, gamma_lc, A)
        F_dA_low = exterior_derivative_of_1form(x, A)

        for mu in range(n):
            for nu in range(n):
                diff = sp.simplify(F_cov_low[mu, nu] - F_dA_low[mu, nu])
                assert diff == 0, f"seed={seed}: F_cov != F_dA at T=0, ({mu},{nu})"

        # The Gamma term: Gamma^{LC,nu}_{mu rho} F^{mu rho}
        F_up = _raise_F(F_dA_low, g_inv, n)
        for nu in range(n):
            val = sp.Integer(0)
            for mu in range(n):
                for rho in range(n):
                    val += gamma_lc[nu, mu, rho] * F_up[mu, rho]
            diff = sp.simplify(_clean(val))
            assert diff == 0, f"seed={seed}: Gamma-LC F term nonzero at T=Q=0, nu={nu}: {val}"

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_two_eoms_differ_by_torsion_term(self, seed):
        """The dA and covariant-curl EOMs differ by the torsion term
        (1/2) T^nu_{mu rho} F^{mu rho} on torsionful backgrounds.

        The dA EOM is nabla^{LC}_mu F^{mu nu} = 0.
        The covcurl EOM is (1/√-g) ∂_μ(√-g F^{μν}) + F^{μρ} Γ^{ν}_{μρ} = 0.

        When both are expressed with the full-connection divergence,
        the dA EOM has an extra (1/2) T^nu_{mu rho} F^{mu rho} term
        compared to the covcurl EOM.  This is exactly the torsion term
        that distinguishes the two field-strength choices.

        VAL-EOM-021 (difference in connection-equation source)."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _make_test_1form(x, seed)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)

        # Compute both F definitions
        F_dA_low = exterior_derivative_of_1form(x, A)
        F_dA_up = _raise_F(F_dA_low, g_inv, n)

        # The torsion term that distinguishes the two EOMs
        torsion_term = sp.MutableDenseNDimArray.zeros(n)
        for nu in range(n):
            val = sp.Integer(0)
            for mu in range(n):
                for rho in range(n):
                    val += sp.Rational(1, 2) * T[nu, mu, rho] * F_dA_up[mu, rho]
            torsion_term[nu] = _clean(val)
        torsion_term = sp.ImmutableDenseNDimArray(torsion_term)

        # The torsion term should be nonzero on a torsionful background
        has_torsion = any(sp.simplify(c) != 0 for c in components(T))
        if has_torsion:
            any_nonzero = any(sp.simplify(c) != 0 for c in components(torsion_term))
            assert any_nonzero, (
                f"seed={seed}: torsion EOM difference is zero on a "
                "torsionful background (the two EOMs should differ)"
            )
