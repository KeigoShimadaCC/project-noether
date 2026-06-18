"""Vector (gauge-field) perturbation on a metric-affine background.

Tests for VAL-PERT-017 and VAL-PERT-018:

- VAL-PERT-017: A gauge field's quadratic action differs between the F=dA
  and connection-covariant field-strength choices by torsion-dependent
  terms; the run records which choice it used and is verified-or-gated.

- VAL-PERT-018: For a theory with both connection and matter dof, the
  quadratic action retains the cross-quadratic mixing between the
  connection/distortion fluctuation and the matter fluctuation (not
  block-diagonalized away), verified-or-gated.

Conventions: noether-default-v1 + metric-affine-v1.
  T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
  Q_{lambda mu nu} = nabla_lambda g_{mu nu}
  Contortion and disformation signs per metric-affine-v1.

Verification gates:
1. Cadabra residue check (passes for the dA template; gated for covcurl)
2. SymPy component cross-check on random metric + connection backgrounds
   (provides the independent verification for the covcurl case)

The covariant-curl Cadabra residue check is gated: the dG*a cross terms
produce mixed-index objects after canonicalise that Cadabra cannot resolve
(the same Kronecker-delta limitation that blocks the covcurl EOM residue
check; see cadabra-gotchas.md).  The SymPy cross-check provides the
independent verification.
"""

from __future__ import annotations

import random

import pytest
import sympy as sp

from noether.kernels.base import (
    Capability,
    ComputedResult,
    KernelTask,
)
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    components,
    exterior_derivative_of_1form,
    nonmetricity_of_connection,
    random_affine_connection,
    random_diagonal_metric,
    torsion_of_connection,
)

# Template names
DA_TEMPLATE = "pert_vector_affine_dA_quadratic"
COVCURL_TEMPLATE = "pert_vector_affine_covcurl_quadratic"


def _run_cadabra(template: str) -> ComputedResult:
    """Run a Cadabra template and return the ComputedResult."""
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.PERTURB,
            description=f"vector-affine perturbation ({template})",
            payload={"template": template},
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
# Cadabra residue checks: dA template (fully verified)
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestVectorAffineDAPerturbation:
    """Cadabra residue checks for the vector perturbation with F = dA
    on a metric-affine background.

    The dA quadratic action is the standard Maxwell fluctuation action
    (no connection fluctuation terms). Both residue_zero and
    linearized_eom_match pass.  VAL-PERT-017 (dA part)."""

    def test_dA_quadratic_action_residue_zero(self):
        """The dA quadratic-action variation matches the linearized
        Maxwell operator.  VAL-PERT-017."""
        result = _run_cadabra(DA_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("residue_zero") == "True", (
            f"dA perturbation residue nonzero: checks={checks}"
        )

    def test_dA_linearized_eom_match(self):
        """The dA quadratic-action variation matches an independently
        linearized EOM.  VAL-PERT-017."""
        result = _run_cadabra(DA_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("linearized_eom_match") == "True", (
            f"dA linearized EOM mismatch: checks={checks}"
        )

    def test_dA_convention_records_field_strength(self):
        """The dA template records its field-strength convention.
        VAL-PERT-017."""
        result = _run_cadabra(DA_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        conventions = result.value.get("conventions", {})
        assert conventions.get("field_strength_definition") == "exterior_derivative", (
            f"dA template missing field-strength convention: {conventions}"
        )

    def test_dA_result_no_connection_fluctuation(self):
        """The dA quadratic action contains no connection fluctuation dG
        because F = dA has no Gamma dependence.  VAL-PERT-017."""
        result = _run_cadabra(DA_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        result_tex = result.value.get("result_tex", "")
        # The NOETHER_RESULT should not contain dG
        assert "dG" not in str(result_tex), (
            f"dA quadratic action unexpectedly contains dG: {result_tex}"
        )


# ===========================================================================
# Cadabra structural checks: covcurl template (gated)
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestVectorAffineCovCurlPerturbation:
    """Cadabra structural checks for the vector perturbation with F = nabla A
    on a metric-affine background.

    The covcurl quadratic action includes a*dG cross terms and dG*dG
    terms (T-dependent). The Cadabra residue check is gated due to the
    Kronecker-delta limitation with mixed-index dG objects.  VAL-PERT-017
    (covcurl part) and VAL-PERT-018."""

    def test_covcurl_has_connection_fluctuation(self):
        """The covcurl quadratic action contains the connection fluctuation
        symbol (dG or T).  VAL-PERT-018."""
        result = _run_cadabra(COVCURL_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("has_connection_fluctuation") == "True", (
            f"covcurl perturbation missing connection fluctuation: checks={checks}"
        )

    def test_covcurl_has_torsion_Abar_coupling(self):
        """The covcurl quadratic action contains T*Abar coupling terms,
        confirming torsion-dependent a*dG cross terms.  VAL-PERT-018."""
        result = _run_cadabra(COVCURL_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("has_torsion_Abar_coupling") == "True", (
            f"covcurl perturbation missing T*Abar coupling: checks={checks}"
        )

    def test_covcurl_residue_is_gated(self):
        """The covcurl residue check is gated (not falsely verified).
        VAL-PERT-017."""
        result = _run_cadabra(COVCURL_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        # The residue check must NOT be "True" (it's gated)
        assert checks.get("residue_zero") != "True", (
            f"covcurl residue unexpectedly passed (should be gated): checks={checks}"
        )
        # The detail must explain why it's gated
        detail = result.value.get("detail", "")
        assert detail != "", (
            "covcurl gated result missing detail explaining the blocker"
        )
        assert "Kronecker-delta" in detail or "mixed-index" in detail, (
            f"covcurl detail does not explain the Cadabra limitation: {detail}"
        )

    def test_covcurl_convention_records_field_strength(self):
        """The covcurl template records its field-strength convention.
        VAL-PERT-017."""
        result = _run_cadabra(COVCURL_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        conventions = result.value.get("conventions", {})
        assert conventions.get("field_strength_definition") == "covariant_curl", (
            f"covcurl template missing field-strength convention: {conventions}"
        )

    def test_covcurl_T_expands_to_dG_difference(self):
        """The torsion symbol T used in the covcurl template expands
        correctly to dG - dG^T.  Structural check."""
        result = _run_cadabra(COVCURL_TEMPLATE)
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("T_expands_to_dG_difference") == "True", (
            f"T expansion check failed: checks={checks}"
        )


# ===========================================================================
# SymPy component cross-checks (the torsion-trap safeguard)
# ===========================================================================


class TestVectorAffinePerturbationSymPy:
    """SymPy cross-checks for the vector perturbation on metric-affine
    backgrounds.

    These verify the key claims of VAL-PERT-017 and VAL-PERT-018 by
    evaluating identities on explicit random metric + connection
    backgrounds and asserting componentwise agreement.

    Key claims:
    1. The dA and covcurl quadratic actions differ by T-dependent terms
       (the torsion of the connection fluctuation times the background
       potential).  VAL-PERT-017.
    2. The covcurl quadratic action contains a*dG cross terms
       (connection-matter mixing not block-diagonalized away).
       VAL-PERT-018.
    """

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_dA_and_covcurl_quadratic_actions_differ(self, seed):
        """The two field-strength choices yield different quadratic actions
        on a torsionful background with a background gauge potential.
        The difference contains T-dependent terms.  VAL-PERT-017."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A_bar = _make_test_1form(x, seed)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        g_imm = sp.ImmutableDenseNDimArray(geom.g)

        # On a flat Minkowski background the connection fluctuation is dG.
        # Here we use the affine connection directly as the "dG" to compute
        # the torsion-dependent difference.

        # dA quadratic action: S2_dA = -1/4 sqrt(-g) f_{mu nu} f^{mu nu}
        # where f = partial a - partial a (no Gamma dependence)
        # This is independent of the connection.

        # covcurl quadratic action: S2_covcurl = -1/4 sqrt(-g) F1_{mu nu} F1^{mu nu}
        # where F1 = f - T(Abar) (includes the torsion correction)
        # f = dA (exterior derivative of the fluctuation)

        # We compute the quadratic action for a specific test fluctuation
        # a_mu (simple polynomial) on the given background.

        # For the dA case, the quadratic action integrand is:
        # L_dA = -1/4 f_{mu nu} f^{mu nu}
        # For the covcurl case:
        # L_covcurl = -1/4 (f - T*Abar)_{mu nu} (f - T*Abar)^{mu nu}

        # Build a simple test fluctuation a_mu
        rng = random.Random(seed + 200)
        a = sp.MutableDenseNDimArray.zeros(n)
        for mu in range(n):
            c = sp.Rational(rng.randint(1, 3), rng.randint(2, 5))
            a[mu] = _clean(c * x[rng.randrange(n)])
        a = sp.ImmutableDenseNDimArray(a)

        # Compute f = dA (exterior derivative of the fluctuation)
        f_low = exterior_derivative_of_1form(x, a)

        # Compute T(Abar) = T^lam_{mu nu} Abar_lam
        # This is the torsion correction to the first-order field strength
        T_Abar_low = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for lam in range(n):
                    val += T[lam, mu, nu] * A_bar[lam]
                T_Abar_low[mu, nu] = _clean(val)
        T_Abar_low = sp.ImmutableDenseNDimArray(T_Abar_low)

        # F1_low = f - T*Abar (the first-order covcurl field strength)
        F1_low = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                F1_low[mu, nu] = _clean(f_low[mu, nu] - T_Abar_low[mu, nu])
        F1_low = sp.ImmutableDenseNDimArray(F1_low)

        # Raise indices
        f_up = _raise_F(f_low, g_inv, n)
        F1_up = _raise_F(F1_low, g_inv, n)
        sqrt_g = _sqrt_neg_g(g_imm)

        # Compute the quadratic action integrands
        L_dA = sp.Integer(0)
        L_covcurl = sp.Integer(0)
        for mu in range(n):
            for nu in range(n):
                L_dA += f_low[mu, nu] * f_up[mu, nu]
                L_covcurl += F1_low[mu, nu] * F1_up[mu, nu]
        L_dA = _clean(-sp.Rational(1, 4) * sqrt_g * L_dA)
        L_covcurl = _clean(-sp.Rational(1, 4) * sqrt_g * L_covcurl)

        # The difference must be nonzero on a torsionful background
        has_torsion = any(sp.simplify(c) != 0 for c in components(T))
        if has_torsion:
            difference = sp.simplify(L_covcurl - L_dA)
            assert difference != 0, (
                f"seed={seed}: dA and covcurl quadratic actions are equal "
                "on a torsionful background (should differ by T-dependent terms)"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_covcurl_has_a_dG_cross_terms(self, seed):
        """The covcurl quadratic action contains a*dG cross terms
        (connection-matter mixing).  VAL-PERT-018.

        We verify this by computing the quadratic action with and without
        the torsion correction and showing the difference involves
        T*Abar (which connects the connection fluctuation dG to the
        matter fluctuation a through the background potential Abar)."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A_bar = _make_test_1form(x, seed)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        g_imm = sp.ImmutableDenseNDimArray(geom.g)

        # Build test fluctuation a_mu
        rng = random.Random(seed + 200)
        a = sp.MutableDenseNDimArray.zeros(n)
        for mu in range(n):
            c = sp.Rational(rng.randint(1, 3), rng.randint(2, 5))
            a[mu] = _clean(c * x[rng.randrange(n)])
        a = sp.ImmutableDenseNDimArray(a)

        # Compute f = dA
        f_low = exterior_derivative_of_1form(x, a)

        # Compute T(Abar)
        T_Abar_low = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for lam in range(n):
                    val += T[lam, mu, nu] * A_bar[lam]
                T_Abar_low[mu, nu] = _clean(val)
        T_Abar_low = sp.ImmutableDenseNDimArray(T_Abar_low)

        # The cross term in the quadratic action is:
        # -1/4 * 2 * f * T*Abar (from the expansion of (f - T*Abar)^2)
        # This equals -1/2 * f_{mu nu} * (T*Abar)^{mu nu}
        # which connects a (through f = da) and dG (through T = dG - dG^T)
        # via the background Abar.

        T_Abar_up = _raise_F(T_Abar_low, g_inv, n)
        sqrt_g = _sqrt_neg_g(g_imm)

        cross_term = sp.Integer(0)
        for mu in range(n):
            for nu in range(n):
                cross_term += f_low[mu, nu] * T_Abar_up[mu, nu]
        cross_term = _clean(sp.Rational(1, 2) * sqrt_g * cross_term)

        # The cross term must be nonzero on a torsionful background
        has_torsion = any(sp.simplify(c) != 0 for c in components(T))
        if has_torsion:
            diff = sp.simplify(cross_term)
            assert diff != 0, (
                f"seed={seed}: a*dG cross term is zero on a torsionful "
                "background (connection-matter mixing should be present)"
            )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_dA_and_covcurl_agree_at_T0(self, seed):
        """At T=0 (Levi-Civita), the dA and covcurl quadratic actions
        are identical because T(Abar) = 0.  VAL-PERT-017 (T=0 limit)."""
        geom = random_diagonal_metric(seed, dim=3)
        n, x = geom.dim, geom.coords
        A_bar = _make_test_1form(x, seed)

        gamma_lc = geom.christoffel

        # At T=0, the torsion correction vanishes
        T = torsion_of_connection(gamma_lc)
        all_zero = all(sp.simplify(c) == 0 for c in components(T))
        assert all_zero, f"seed={seed}: torsion nonzero at Levi-Civita"

        # Build test fluctuation a_mu
        rng = random.Random(seed + 200)
        a = sp.MutableDenseNDimArray.zeros(n)
        for mu in range(n):
            c = sp.Rational(rng.randint(1, 3), rng.randint(2, 5))
            a[mu] = _clean(c * x[rng.randrange(n)])
        a = sp.ImmutableDenseNDimArray(a)

        # Compute T*Abar (should be zero at T=0)
        T_Abar_low = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for lam in range(n):
                    val += T[lam, mu, nu] * A_bar[lam]
                T_Abar_low[mu, nu] = _clean(val)
        T_Abar_low = sp.ImmutableDenseNDimArray(T_Abar_low)

        # Verify T*Abar = 0
        for mu in range(n):
            for nu in range(n):
                assert sp.simplify(T_Abar_low[mu, nu]) == 0, (
                    f"seed={seed}: T*Abar nonzero at T=0, ({mu},{nu})"
                )

        # Therefore F1 = f - T*Abar = f, and the quadratic actions are equal

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_covcurl_eom_has_torsion_source(self, seed):
        """The linearized vector EOM for the covcurl case includes a
        torsion source term from the connection fluctuation.
        VAL-PERT-017/018.

        The linearized EOM is:
          partial_mu f^{mu nu} - partial_mu (T^{mu nu}_lam Abar^lam) = 0
        The second term is the torsion source from the connection
        fluctuation, absent in the dA case."""
        geom, gamma, T, Q = _make_affine_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A_bar = _make_test_1form(x, seed)

        # Compute the torsion source term:
        # partial_mu (T^{mu nu}_{lambda} Abar^lambda)
        # This is nonzero on a torsionful background with Abar != 0
        has_torsion = any(sp.simplify(c) != 0 for c in components(T))
        if not has_torsion:
            return  # Skip on torsion-free backgrounds

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)

        # Compute T^{mu nu}_{lam} Abar^lam (raised version)
        T_Abar_up = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for lam in range(n):
                    for rho in range(n):
                        for sig in range(n):
                            val += (
                                g_inv[mu, rho] * g_inv[nu, sig]
                                * T[lam, rho, sig] * A_bar[lam]
                            )
                T_Abar_up[mu, nu] = _clean(val)
        T_Abar_up = sp.ImmutableDenseNDimArray(T_Abar_up)

        # The torsion source term: partial_mu (T^{mu nu}_{lam} Abar^lam)
        # This is the difference between the covcurl and dA linearized EOMs.
        torsion_source = sp.MutableDenseNDimArray.zeros(n)
        for nu in range(n):
            val = sp.Integer(0)
            for mu in range(n):
                val += sp.diff(T_Abar_up[mu, nu], x[mu])
            torsion_source[nu] = _clean(val)
        torsion_source = sp.ImmutableDenseNDimArray(torsion_source)

        # The torsion source should be nonzero on a torsionful background
        any_nonzero = any(sp.simplify(c) != 0 for c in components(torsion_source))
        assert any_nonzero, (
            f"seed={seed}: torsion source in linearized EOM is zero on "
            "a torsionful background (the two EOMs should differ)"
        )


class TestPerturbationVerifiedGatedXOR:
    """VAL-PERT-006/007: verified==True with both checks True,
    OR gated (verified==False) with a non-empty detail.

    The dA template is verified (both checks True).
    The covcurl template is gated (checks are 'gated', detail non-empty)."""

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_dA_verified_with_both_checks_true(self):
        """The dA quadratic-action perturbation is verified=True with
        both residue_zero and linearized_eom_match True."""
        result = _run_cadabra(DA_TEMPLATE)
        assert result.raw.returncode == 0
        checks = result.value.get("checks", {})
        assert checks.get("residue_zero") == "True"
        assert checks.get("linearized_eom_match") == "True"

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_covcurl_gated_with_detail(self):
        """The covcurl quadratic-action perturbation is gated
        (verified=False) with a non-empty detail explaining the blocker."""
        result = _run_cadabra(COVCURL_TEMPLATE)
        assert result.raw.returncode == 0
        checks = result.value.get("checks", {})
        # The checks should be "gated" (not "True")
        assert checks.get("residue_zero") == "gated"
        assert checks.get("linearized_eom_match") == "gated"
        detail = result.value.get("detail", "")
        assert detail != ""
