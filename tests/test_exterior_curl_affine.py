"""Pin the exterior derivative vs covariant curl identity for a 1-form.

Tests for VAL-GEOM-020:

- The identity ``2 partial_{[mu} A_{nu]} = 2 nabla_{[mu} A_{nu]} + T^lambda_{mu nu} A_lambda``
  is residue-pinned and SymPy-confirmed.
- The covariant curl equals ``dA - T.A`` on random backgrounds.
- The torsion term is nonzero when T != 0 and zero when T = 0.

Physics: For a 1-form A_mu under a general affine connection Gamma^lambda_{mu nu}:

  nabla_mu A_nu = partial_mu A_nu - Gamma^lambda_{mu nu} A_lambda
  nabla_nu A_mu = partial_nu A_mu - Gamma^lambda_{nu mu} A_lambda

  2 nabla_{[mu} A_{nu]} = (partial_mu A_nu - partial_nu A_mu)
                           - (Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}) A_lambda
                         = dA - T^lambda_{mu nu} A_lambda

  Rearranging: dA = covariant curl + T.A

This is the primitive behind the gauge field-strength subtlety used in M3:
on a Levi-Civita background (T=0) the exterior derivative and covariant
curl coincide, but under torsion they differ.  LC code that equates
F = dA with F = nabla A is invalid under torsion.

Convention: noether-default-v1 + metric-affine-v1.
  T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
"""

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.curvature import (
    AFFINE_CONNECTION_DEPENDS,
    AFFINE_CONNECTION_DEPENDS_NABLA,
    AFFINE_CURVATURE_DECL,
    TORSION_DECL,
    TORSION_DECL_NABLA,
    curl_vs_exterior_affine,
    define_torsion,
)
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    covariant_curl_of_1form,
    exterior_derivative_of_1form,
    random_affine_connection,
    random_diagonal_metric,
    torsion_of_connection,
)

# ---------------------------------------------------------------------------
# Cadabra script builders
# ---------------------------------------------------------------------------

_BASE_DECL_AFFINE = (
    r"{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon}"
    r"::Indices(position=fixed)."
    "\n"
    r"{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon}"
    r"::Integer(range=0..3)."
    "\n"
    r"\partial{#}::PartialDerivative."
    "\n"
    r"g_{\mu\nu}::Metric."
    "\n"
    r"g^{\mu\nu}::InverseMetric."
    "\n"
    r"g^{\mu}_{\nu}::KroneckerDelta."
    "\n"
    r"g_{\mu}^{\nu}::KroneckerDelta."
    "\n"
)

_BASE_DECL_NABLA = (
    r"{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon}"
    r"::Indices(position=fixed)."
    "\n"
    r"{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon}"
    r"::Integer(range=0..3)."
    "\n"
    r"\nabla{#}::Derivative."
    "\n"
    r"\partial{#}::PartialDerivative."
    "\n"
    r"g_{\mu\nu}::Metric."
    "\n"
    r"g^{\mu\nu}::InverseMetric."
    "\n"
    r"g^{\mu}_{\nu}::KroneckerDelta."
    "\n"
    r"g_{\mu}^{\nu}::KroneckerDelta."
    "\n"
)


def _affine_script(body: str) -> str:
    return (
        _BASE_DECL_AFFINE
        + AFFINE_CURVATURE_DECL
        + "\n"
        + AFFINE_CONNECTION_DEPENDS
        + "\n"
        + TORSION_DECL
        + "\n"
        # A is a 1-form depending on partial derivatives
        + r"{A_{\mu}}::Depends(\partial{#})."
        + "\n"
        + body
    )


def _nabla_script(body: str) -> str:
    return (
        _BASE_DECL_NABLA
        + AFFINE_CURVATURE_DECL
        + "\n"
        + AFFINE_CONNECTION_DEPENDS_NABLA
        + "\n"
        + TORSION_DECL_NABLA
        + "\n"
        # A is a 1-form depending on both nabla and partial
        + r"{A_{\mu}}::Depends(\nabla{#}, \partial{#})."
        + "\n"
        + body
    )


def _run_affine(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="affine exterior-vs-curl residue check",
            payload={"script": _affine_script(body)},
        )
    )


def _run_nabla(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="affine nabla curl substitution check",
            payload={"script": _nabla_script(body)},
        )
    )


# ---------------------------------------------------------------------------
# SymPy cross-check helpers
# ---------------------------------------------------------------------------


def _sympy_background(seed: int, dim: int = 3, symmetric: bool = False):
    """Build a random metric + connection, torsion, and test 1-form."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(
        seed + 1000, geom.coords, symmetric=symmetric
    )
    T = torsion_of_connection(gamma)
    return geom, gamma, T


def _test_1form(coords, seed: int):
    """Build a simple polynomial 1-form for testing."""
    rng = __import__("random").Random(seed + 500)
    n = len(coords)
    A = sp.MutableDenseNDimArray.zeros(n)
    for mu in range(n):
        c = sp.Rational(rng.randint(1, 3), rng.randint(2, 5))
        A[mu] = _clean(c * coords[rng.randrange(n)])
    return sp.ImmutableDenseNDimArray(A)


# ===========================================================================
# Cadabra residue checks
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestExteriorCurlResidue:
    """Cadabra residue checks for the exterior-vs-covariant-curl identity.

    The Cadabra residue checks verify that the identity holds by expanding
    both sides from the definition of the covariant derivative and torsion.
    The physics verification (that the identity is correct for a general
    torsionful connection) is additionally cross-checked by the SymPy
    component oracle on explicit random backgrounds.
    """

    def test_curl_identity_residue_zero(self):
        """2 partial_{[mu} A_{nu]} = 2 nabla_{[mu} A_{nu]} + T^lam_{mu nu} A_lam
        verified by expanding both sides from the connection definition.
        VAL-GEOM-020 (residue pin).

        LHS = partial_mu A_nu - partial_nu A_mu  (exterior derivative dA)
        RHS = (partial_mu A_nu - G^lam_{mu nu} A_lam)
              - (partial_nu A_mu - G^lam_{nu mu} A_lam)
              + T^lam_{mu nu} A_lam
            = dA - (G^lam_{mu nu} - G^lam_{nu mu}) A_lam + T^lam_{mu nu} A_lam
            = dA - T^lam_{mu nu} A_lam + T^lam_{mu nu} A_lam
            = dA = LHS  ✓
        """
        body = (
            # LHS: exterior derivative dA
            r"lhs := \partial_{\mu}{A_{\nu}} - \partial_{\nu}{A_{\mu}};"
            "\n"
            "distribute(lhs); canonicalise(lhs); rename_dummies(lhs);\n"
            # RHS: covariant curl + T.A
            #   covariant curl = nabla_mu A_nu - nabla_nu A_mu
            #   = (partial_mu A_nu - G^lam_{mu nu} A_lam)
            #     - (partial_nu A_mu - G^lam_{nu mu} A_lam)
            #   T^lam_{mu nu} A_lam = (G^lam_{mu nu} - G^lam_{nu mu}) A_lam
            r"rhs := (\partial_{\mu}{A_{\nu}} - G^{\lambda}_{\mu\nu} A_{\lambda})"
            r" - (\partial_{\nu}{A_{\mu}} - G^{\lambda}_{\nu\mu} A_{\lambda})"
            r" + T^{\lambda}_{\mu\nu} A_{\lambda};"
            "\n"
            # Expand T from the connection
            + define_torsion("G", "rhs")
            + "\n"
            "distribute(rhs); canonicalise(rhs); rename_dummies(rhs);\n"
            # Residue: LHS - RHS should be 0
            "residue := @(lhs) - @(rhs);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: curl_identity_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("curl_identity_zero") == "True", (
            result.raw.stdout
        )

    def test_curl_substitution_fires(self):
        """The curl_vs_exterior_affine substitution fires correctly:
        nabla_mu A_nu - nabla_nu A_mu -> partial_mu A_nu - partial_nu A_mu
        - T^lam_{mu nu} A_lam.
        VAL-GEOM-020 (substitution machinery)."""
        body = (
            r"ex := \nabla_{\mu}{A_{\nu}} - \nabla_{\nu}{A_{\mu}};"
            "\n"
            + curl_vs_exterior_affine("A", "ex")
            + "\n"
            "canonicalise(ex);\n"
            # Check both partial and torsion terms appear in the result
            'has_both = "partial" in str(ex) and "T" in str(ex)\n'
            'print("NOETHER_CHECK: curl_subst_fires=" + str(has_both))'
        )
        result = _run_nabla(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("curl_subst_fires") == "True", (
            result.raw.stdout
        )

    def test_covariant_curl_equals_dA_at_T0(self):
        """When T=0 (symmetric connection), the covariant curl equals dA.
        VAL-GEOM-020 (T=0 limit, Cadabra residue).

        nabla_mu A_nu - nabla_nu A_mu = partial_mu A_nu - partial_nu A_mu
        when G is symmetric in the lower pair."""
        body = (
            # Build the covariant curl from definition with symmetric G.
            # Use simple name (no underscore) for the @() reference operator.
            r"ccurl := (\partial_{\mu}{A_{\nu}} - G^{\lambda}_{\mu\nu} A_{\lambda})"
            r" - (\partial_{\nu}{A_{\mu}} - G^{\lambda}_{\nu\mu} A_{\lambda});"
            "\n"
            "distribute(ccurl); canonicalise(ccurl); "
            "rename_dummies(ccurl); meld(ccurl);\n"
            # When G is symmetric, G^lam_{mu nu} = G^lam_{nu mu}, so the
            # Gamma terms cancel and we get dA
            r"target := \partial_{\mu}{A_{\nu}} - \partial_{\nu}{A_{\mu}};"
            "\n"
            "distribute(target); canonicalise(target); "
            "rename_dummies(target);\n"
            "residue := @(ccurl) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: curl_T0_zero=" '
            '+ str(str(residue) == "0"))'
        )
        # Use affine script with symmetric G
        script = (
            _BASE_DECL_AFFINE
            + AFFINE_CURVATURE_DECL
            + "\n"
            # Override: G is symmetric (no torsion)
            r"G^{\lambda}_{\mu\nu}::Symmetric(\mu,\nu)."
            "\n"
            + r"{G^{\lambda}_{\mu\nu}, g_{\mu\nu}, g^{\mu\nu}}::Depends(\partial{#})."
            + "\n"
            + r"{A_{\mu}}::Depends(\partial{#})."
            + "\n"
            + body
        )
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="curl T=0 check",
                payload={"script": script},
            )
        )
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("curl_T0_zero") == "True", (
            result.raw.stdout
        )


# ===========================================================================
# SymPy component cross-checks (the torsion-trap safeguard)
# ===========================================================================


class TestExteriorCurlSymPyCrossCheck:
    """Cross-check the exterior-vs-covariant-curl identity against the SymPy
    oracle on explicit random backgrounds.  This is the independent
    verification that catches the torsion trap (architecture.md section 3.2)."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_covariant_curl_equals_dA_minus_TA(self, seed):
        """Covariant curl = dA - T^lam_{mu nu} A_lam on a torsionful
        background.  VAL-GEOM-020 (SymPy cross-check, main identity).

        This verifies the identity:
          nabla_mu A_nu - nabla_nu A_mu = (partial_mu A_nu - partial_nu A_mu)
                                          - T^lam_{mu nu} A_lam
        componentwise on a random metric + connection background.
        """
        geom, gamma, T = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _test_1form(x, seed)

        # Compute the covariant curl
        cov_curl = covariant_curl_of_1form(x, gamma, A)

        # Compute the exterior derivative
        dA = exterior_derivative_of_1form(x, A)

        # Compute the torsion correction: T^lam_{mu nu} A_lam
        T_correction = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for lam in range(n):
                    val += T[lam, mu, nu] * A[lam]
                T_correction[mu, nu] = _clean(val)
        T_correction = sp.ImmutableDenseNDimArray(T_correction)

        # Verify: covariant_curl = dA - T.A
        for mu in range(n):
            for nu in range(n):
                lhs = cov_curl[mu, nu]
                rhs = _clean(dA[mu, nu] - T_correction[mu, nu])
                diff = sp.simplify(lhs - rhs)
                assert diff == 0, (
                    f"seed={seed}: curl identity fails at "
                    f"({mu},{nu}): cov_curl={lhs}, "
                    f"dA-T.A={rhs}, diff={diff}"
                )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_dA_equals_covariant_curl_plus_TA(self, seed):
        """dA = covariant curl + T.A (the identity as stated in the feature).
        VAL-GEOM-020 (SymPy cross-check, original form).

        2 partial_{[mu} A_{nu]} = 2 nabla_{[mu} A_{nu]} + T^lam_{mu nu} A_lam
        """
        geom, gamma, T = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _test_1form(x, seed)

        dA = exterior_derivative_of_1form(x, A)
        cov_curl = covariant_curl_of_1form(x, gamma, A)

        # T^lam_{mu nu} A_lam
        T_correction = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for lam in range(n):
                    val += T[lam, mu, nu] * A[lam]
                T_correction[mu, nu] = _clean(val)
        T_correction = sp.ImmutableDenseNDimArray(T_correction)

        # Verify: dA = cov_curl + T.A
        for mu in range(n):
            for nu in range(n):
                lhs = dA[mu, nu]
                rhs = _clean(cov_curl[mu, nu] + T_correction[mu, nu])
                diff = sp.simplify(lhs - rhs)
                assert diff == 0, (
                    f"seed={seed}: dA identity fails at "
                    f"({mu},{nu}): dA={lhs}, "
                    f"cov_curl+T.A={rhs}, diff={diff}"
                )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_torsion_term_nonzero_at_T_neq_0(self, seed):
        """The torsion term T^lam_{mu nu} A_lam is nonzero when T != 0.
        VAL-GEOM-020 (nonzero T-term at T!=0).

        This demonstrates that the covariant curl and the exterior derivative
        genuinely differ on torsionful backgrounds - the torsion correction
        is not just a formal expression that happens to vanish.
        """
        geom, gamma, T = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _test_1form(x, seed)

        # Check that T^lam_{mu nu} A_lam is nonzero for some components
        any_nonzero = False
        for mu in range(n):
            for nu in range(mu + 1, n):
                torsion_term = sum(T[lam, mu, nu] * A[lam] for lam in range(n))
                if sp.simplify(torsion_term) != 0:
                    any_nonzero = True
                    break
            if any_nonzero:
                break
        assert any_nonzero, (
            f"seed={seed}: torsion term T^lam_{{mu,nu}} A_lam is zero on a "
            "torsionful background (should be nonzero)"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_torsion_term_zero_at_T0(self, seed):
        """The torsion term T^lam_{mu nu} A_lam is zero when T=0
        (Levi-Civita / symmetric connection).  VAL-GEOM-020 (zero at T=0).

        When torsion vanishes, the covariant curl equals the exterior
        derivative, and LC code that equates them is valid.
        """
        geom, gamma, T = _sympy_background(seed, dim=3, symmetric=True)
        n, x = geom.dim, geom.coords
        A = _test_1form(x, seed)

        # T should be identically zero for a symmetric connection
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    assert T[lam, mu, nu] == 0, (
                        f"seed={seed}: T[{lam},{mu},{nu}] = {T[lam, mu, nu]} "
                        "on a symmetric connection (should be zero)"
                    )

        # The torsion correction should be identically zero
        for mu in range(n):
            for nu in range(mu + 1, n):
                torsion_term = sum(T[lam, mu, nu] * A[lam] for lam in range(n))
                assert torsion_term == 0, (
                    f"seed={seed}: T^lam_{{{mu},{nu}}} A_lam = {torsion_term} "
                    "at T=0 (should be zero)"
                )

        # The covariant curl should equal the exterior derivative
        dA = exterior_derivative_of_1form(x, A)
        cov_curl = covariant_curl_of_1form(x, gamma, A)
        for mu in range(n):
            for nu in range(mu + 1, n):
                diff = sp.simplify(cov_curl[mu, nu] - dA[mu, nu])
                assert diff == 0, (
                    f"seed={seed}: cov_curl[{mu},{nu}] != dA[{mu},{nu}] at T=0: "
                    f"cov_curl={cov_curl[mu, nu]}, dA={dA[mu, nu]}"
                )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_lc_equation_curl_equals_dA_fails_under_torsion(self, seed):
        """A deliberately LC-only equation (covariant curl = dA) disagrees
        with the actual covariant curl on a torsionful background.
        This is the torsion-trap demonstration for gauge fields: reusing
        the LC identity covariant_curl = dA under torsion gives a wrong
        answer.  VAL-GEOM-020 (trap guard for gauge field strength)."""
        geom, gamma, T = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        A = _test_1form(x, seed)

        cov_curl = covariant_curl_of_1form(x, gamma, A)
        dA = exterior_derivative_of_1form(x, A)

        # On a torsionful background, cov_curl != dA
        any_disagreement = False
        for mu in range(n):
            for nu in range(mu + 1, n):
                diff = sp.simplify(cov_curl[mu, nu] - dA[mu, nu])
                if diff != 0:
                    any_disagreement = True
                    break
            if any_disagreement:
                break
        assert any_disagreement, (
            f"seed={seed}: covariant curl equals dA on a torsionful "
            "background (torsion trap not caught for gauge fields!)"
        )
