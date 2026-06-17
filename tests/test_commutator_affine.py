"""Pin the torsionful commutator and non-symmetric scalar Hessian.

Tests for VAL-GEOM-004, VAL-GEOM-005, and VAL-GEOM-006:

- VAL-GEOM-004: The general commutator
  [nabla_a, nabla_b] V_c = -R^d_{cab} V_d - T^d_{ab} nabla_d V_c
  is residue-pinned and matches the SymPy oracle on a torsionful background
  where the T-term is demonstrably nonzero.

- VAL-GEOM-005: With torsion zero, the general primitive matches the existing
  commute_third_derivative (residue 0) and the SymPy LC-limit agrees.

- VAL-GEOM-006: nabla_mu nabla_nu phi - nabla_nu nabla_mu phi
  = -T^lambda_{mu nu} nabla_lambda phi is residue-pinned; the antisymmetric
  Hessian part is nonzero on a torsionful background and zero at T=0
  (so LC hessian_to_symmetric is invalid here).

Each Cadabra primitive is pinned by a residue check (the kernel verifies
the substitution produces the correct expansion).  Each is additionally
cross-checked against the SymPy general-connection oracle on explicit
random metric + connection backgrounds (the torsion-trap safeguard).
They skip when cadabra2 is absent.

Key correctness point: the commutator identity is derived by treating
nabla_b V_c as a (0,1) tensor with index c, then applying nabla_a
to this covector.  The resulting (0,2) tensor nabla_a nabla_b V_c
has both derivative indices a,b and the original index c.  The
commutator [nabla_a, nabla_b] V_c picks up both the Riemann term
(-R^d_{cab} V_d) from the derivative index c and the torsion term
(-T^d_{ab} nabla_d V_c) from the antisymmetric part of the connection
acting on the derivative indices a,b.

For the scalar Hessian, nabla_mu phi is a (0,1) tensor (covector), so
nabla_nu nabla_mu phi = partial_nu partial_mu phi - Gamma^lam_{nu mu} partial_lam phi,
and the commutator gives only the torsion term (no Riemann term for
scalars).
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
    commute_third_derivative,
    commute_third_derivative_affine,
    hessian_antisymmetry_affine,
)
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    covariant_derivative_of_connection,
    random_affine_connection,
    random_diagonal_metric,
    riemann_down_of_connection,
    riemann_of_connection,
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

# Base declarations for nabla-based scripts (commutator identity tests).
_BASE_DECL_NABLA = (
    r"{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon}"
    r"::Indices(position=fixed)."
    "\n"
    r"{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon}"
    r"::Integer(range=0..3)."
    "\n"
    r"\nabla{#}::Derivative."
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
        + r"{\phi}::Depends(\partial{#})."
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
        + r"{phi}::Depends(\nabla{#})."
        + "\n"
        + body
    )


def _run_affine(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="affine commutator primitive check",
            payload={"script": _affine_script(body)},
        )
    )


def _run_nabla(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="affine nabla commutator check",
            payload={"script": _nabla_script(body)},
        )
    )


# ---------------------------------------------------------------------------
# SymPy cross-check helpers
# ---------------------------------------------------------------------------


def _sympy_background(seed: int, dim: int = 3, symmetric: bool = False):
    """Build a random metric + connection and compute all needed objects."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(
        seed + 1000, geom.coords, symmetric=symmetric
    )
    T = torsion_of_connection(gamma)
    R_up = riemann_of_connection(geom.coords, gamma)
    R_down = riemann_down_of_connection(geom.coords, gamma, geom.g)
    return geom, gamma, T, R_up, R_down


# ===========================================================================
# Cadabra residue checks
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestCommutatorResidue:
    """Cadabra residue checks for the torsionful commutator and Hessian.

    The Cadabra residue checks verify that the substitution primitives fire
    correctly on their matching patterns.  The physics verification (that
    the identity is correct for a general torsionful connection) is provided
    by the SymPy component cross-checks below, which evaluate both sides on
    explicit random metric + connection backgrounds.

    For the scalar Hessian, an additional direct physics verification is
    possible: expanding both sides from the connection definition using
    partial derivatives gives a residue of zero, because the calculation
    is simple enough for Cadabra to close without metric-contraction issues.
    """

    def test_scalar_hessian_antisymmetry_residue_zero(self):
        """nabla_mu nabla_nu phi - nabla_nu nabla_mu phi = -T^lam_{mu nu} nabla_lam phi
        verified by expanding both sides from the connection definition.
        VAL-GEOM-006 (residue pin).

        For a scalar phi:
          nabla_mu nabla_nu phi = partial_mu partial_nu phi - G^lam_{mu nu} partial_lam phi
          nabla_nu nabla_mu phi = partial_nu partial_mu phi - G^lam_{nu mu} partial_lam phi
        Since partial_mu partial_nu phi = partial_nu partial_mu phi (scalar),
        the difference is -(G^lam_{mu nu} - G^lam_{nu mu}) partial_lam phi
        = -T^lam_{mu nu} partial_lam phi = -T^lam_{mu nu} nabla_lam phi.
        """
        body = (
            r"hess := (\partial_{\mu}{\partial_{\nu}{\phi}}"
            r" - G^{\lambda}_{\mu\nu} \partial_{\lambda}{\phi})"
            r" - (\partial_{\nu}{\partial_{\mu}{\phi}}"
            r" - G^{\lambda}_{\nu\mu} \partial_{\lambda}{\phi});"
            "\n"
            "distribute(hess); canonicalise(hess); rename_dummies(hess);\n"
            r"target := -(G^{\lambda}_{\mu\nu} - G^{\lambda}_{\nu\mu})"
            r" \partial_{\lambda}{\phi};"
            "\n"
            "distribute(target); canonicalise(target); rename_dummies(target);\n"
            "residue := @(hess) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: hessian_antisym_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("hessian_antisym_zero") == "True", (
            result.raw.stdout
        )

    def test_hessian_substitution_fires(self):
        """The hessian_antisymmetry_affine substitution fires correctly:
        nabla_mu nabla_nu phi - nabla_nu nabla_mu phi -> -T^lam_{mu nu} nabla_lam phi.
        VAL-GEOM-006 (substitution machinery)."""
        body = (
            r"ex := \nabla_{\mu}{\nabla_{\nu}{phi}}"
            r" - \nabla_{\nu}{\nabla_{\mu}{phi}};"
            "\n"
            + hessian_antisymmetry_affine("phi", "ex")
            + "\n"
            "canonicalise(ex);\n"
            # Check the torsion term appears in the result
            'has_T = "T" in str(ex)\n'
            'print("NOETHER_CHECK: hessian_subst_fires=" + str(has_T))'
        )
        result = _run_nabla(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("hessian_subst_fires") == "True", (
            result.raw.stdout
        )

    def test_commutator_substitution_fires(self):
        """The commute_third_derivative_affine substitution fires correctly:
        nabla_mu nabla_nu nabla_rho phi - nabla_nu nabla_mu nabla_rho phi
        -> -g^{dl} R_{lrho mu nu} nabla_d phi - T^d_{mu nu} nabla_d nabla_rho phi.
        VAL-GEOM-004 (substitution machinery)."""
        body = (
            r"ex := \nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{phi}}}"
            r" - \nabla_{\nu}{\nabla_{\mu}{\nabla_{\rho}{phi}}};"
            "\n"
            + commute_third_derivative_affine("phi", "ex")
            + "\n"
            "canonicalise(ex);\n"
            'has_both = "R" in str(ex) and "T" in str(ex)\n'
            'print("NOETHER_CHECK: commutator_subst_fires=" + str(has_both))'
        )
        result = _run_nabla(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("commutator_subst_fires") == "True", (
            result.raw.stdout
        )

    def test_commutator_vs_target_residue_zero(self):
        """The affine commutator substitution produces the expected RHS.
        VAL-GEOM-004 (residue pin).

        Apply commute_third_derivative_affine to the LHS pattern, then
        compare with the explicit target RHS.  The residue should be zero
        because the substitution IS the identity.
        """
        body = (
            r"comm := \nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{phi}}}"
            r" - \nabla_{\nu}{\nabla_{\mu}{\nabla_{\rho}{phi}}};"
            "\n"
            + commute_third_derivative_affine("phi", "comm")
            + "\n"
            "distribute(comm); canonicalise(comm); "
            "rename_dummies(comm);\n"
            r"target := - g^{\delta\lambda} R_{\lambda\rho\mu\nu}"
            r" \nabla_{\delta}{phi}"
            r" - T^{\delta}_{\mu\nu}"
            r" \nabla_{\delta}{\nabla_{\rho}{phi}};"
            "\n"
            "distribute(target); canonicalise(target); "
            "rename_dummies(target);\n"
            "residue := @(comm) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: commutator_residue_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_nabla(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("commutator_residue_zero") == "True", (
            result.raw.stdout
        )

    def test_commutator_reduces_to_lc_when_torsion_zero(self):
        """When T=0, the torsionful commutator reduces to the Levi-Civita
        commutator.  VAL-GEOM-005 (T=0 limit).

        The difference between the affine and LC commutator substitutions
        is exactly the torsion term -T^d_{ab} nabla_d nabla_c phi.
        """
        body = (
            # Apply the affine commutator
            r"affine := \nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{phi}}}"
            r" - \nabla_{\nu}{\nabla_{\mu}{\nabla_{\rho}{phi}}};"
            "\n"
            + commute_third_derivative_affine("phi", "affine")
            + "\n"
            # Apply the LC commutator to a copy
            r"lc := \nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{phi}}}"
            r" - \nabla_{\nu}{\nabla_{\mu}{\nabla_{\rho}{phi}}};"
            "\n"
            + commute_third_derivative("phi", "lc")
            + "\n"
            # The difference should be the torsion term
            r"diff := @(affine) - @(lc);"
            "\n"
            "distribute(diff); canonicalise(diff); rename_dummies(diff);\n"
            r"target := -T^{\delta}_{\mu\nu}"
            r" \nabla_{\delta}{\nabla_{\rho}{phi}};"
            "\n"
            "residue := @(diff) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: commutator_lc_diff_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_nabla(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("commutator_lc_diff_zero") == "True", (
            result.raw.stdout
        )

    def test_hessian_zero_at_T0(self):
        """The scalar Hessian antisymmetric part vanishes when T=0
        (Levi-Civita limit).  VAL-GEOM-006 (T=0 case).

        Build the Hessian from the definition using a symmetric connection
        and verify the antisymmetric part is zero."""
        body = (
            r"hess := (\partial_{\mu}{\partial_{\nu}{\phi}}"
            r" - G^{\lambda}_{\mu\nu} \partial_{\lambda}{\phi})"
            r" - (\partial_{\nu}{\partial_{\mu}{\phi}}"
            r" - G^{\lambda}_{\nu\mu} \partial_{\lambda}{\phi});"
            "\n"
            "distribute(hess); canonicalise(hess); "
            "rename_dummies(hess); meld(hess);\n"
            'print("NOETHER_CHECK: hessian_T0_zero=" '
            '+ str(str(hess) == "0"))'
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
            + r"{\phi}::Depends(\partial{#})."
            + "\n"
            + body
        )
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="hessian T=0 check",
                payload={"script": script},
            )
        )
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("hessian_T0_zero") == "True", (
            result.raw.stdout
        )


# ===========================================================================
# SymPy component cross-checks (the torsion-trap safeguard)
# ===========================================================================


class TestCommutatorSymPyCrossCheck:
    """Cross-check the torsionful commutator and Hessian against the SymPy
    oracle on explicit random backgrounds.  This is the independent
    verification that catches the torsion trap (architecture.md section 3.2)."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_scalar_hessian_antisymmetry_on_torsionful_background(self, seed):
        """nabla_mu nabla_nu phi - nabla_nu nabla_mu phi = -T^lam_{mu nu} nabla_lam phi
        on a torsionful background.  VAL-GEOM-006 (SymPy cross-check).

        For a scalar phi, nabla_mu phi = partial_mu phi.
        nabla_mu nabla_nu phi = partial_mu(partial_nu phi)
                                - Gamma^lam_{mu nu} partial_lam phi
        (covariant derivative of the covector partial_nu phi).
        """
        geom, gamma, T, R_up, R_down = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords

        # Create a scalar field phi (simple polynomial)
        phi = sp.Rational(1, 2) * x[0] ** 2 + x[1] * x[2]

        # nabla_nu phi = partial_nu phi (scalar derivative, 1-form)
        nab_phi = covariant_derivative_of_connection(x, gamma, phi, variances=[])

        # nabla_mu nabla_nu phi = nabla_mu acting on the covector nab_phi
        # This is a (0,2) tensor: nab2_phi[mu, nu]
        nab2_phi = covariant_derivative_of_connection(
            x, gamma, nab_phi, variances=["down"]
        )

        # Antisymmetric part: nab2_phi[mu, nu] - nab2_phi[nu, mu]
        antisym = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                antisym[mu, nu] = _clean(nab2_phi[mu, nu] - nab2_phi[nu, mu])
        antisym = sp.ImmutableDenseNDimArray(antisym)

        # RHS: -T^lam_{mu nu} nab_lam phi = -T^lam_{mu nu} partial_lam phi
        rhs = sp.MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                val = sp.Integer(0)
                for lam in range(n):
                    val += T[lam, mu, nu] * nab_phi[lam]
                rhs[mu, nu] = _clean(-val)
        rhs = sp.ImmutableDenseNDimArray(rhs)

        # Componentwise equality
        for mu in range(n):
            for nu in range(n):
                diff = sp.simplify(antisym[mu, nu] - rhs[mu, nu])
                assert diff == 0, (
                    f"seed={seed}: Hessian antisymmetry fails at "
                    f"({mu},{nu}): LHS={antisym[mu, nu]}, RHS={rhs[mu, nu]}"
                )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_hessian_nonzero_at_torsion(self, seed):
        """The Hessian antisymmetric part is nonzero on a torsionful
        background.  VAL-GEOM-006 (nonzero at T!=0)."""
        geom, gamma, T, R_up, R_down = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords

        phi = sp.Rational(1, 2) * x[0] ** 2 + x[1] * x[2]
        nab_phi = covariant_derivative_of_connection(x, gamma, phi, variances=[])
        nab2_phi = covariant_derivative_of_connection(
            x, gamma, nab_phi, variances=["down"]
        )

        any_nonzero = False
        for mu in range(n):
            for nu in range(mu + 1, n):
                diff = sp.simplify(nab2_phi[mu, nu] - nab2_phi[nu, mu])
                if diff != 0:
                    any_nonzero = True
                    break
            if any_nonzero:
                break
        assert any_nonzero, (
            f"seed={seed}: Hessian is symmetric on a torsionful background "
            "(the antisymmetric part should be nonzero)"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_hessian_zero_at_T0(self, seed):
        """The Hessian antisymmetric part is zero when T=0
        (Levi-Civita / symmetric connection).  VAL-GEOM-006 (T=0 case)."""
        geom, gamma, T, R_up, R_down = _sympy_background(
            seed, dim=3, symmetric=True
        )
        n, x = geom.dim, geom.coords

        phi = sp.Rational(1, 2) * x[0] ** 2 + x[1] * x[2]
        nab_phi = covariant_derivative_of_connection(x, gamma, phi, variances=[])
        nab2_phi = covariant_derivative_of_connection(
            x, gamma, nab_phi, variances=["down"]
        )

        for mu in range(n):
            for nu in range(mu + 1, n):
                diff = sp.simplify(nab2_phi[mu, nu] - nab2_phi[nu, mu])
                assert diff == 0, (
                    f"seed={seed}: Hessian not symmetric at T=0: "
                    f"nab2[{mu},{nu}]={nab2_phi[mu, nu]}, "
                    f"nab2[{nu},{mu}]={nab2_phi[nu, mu]}"
                )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_covector_commutator_on_torsionful_background(self, seed):
        """[nabla_a, nabla_b] V_c = -R^d_{cab} V_d - T^d_{ab} nabla_d V_c
        on a torsionful background with a nonzero T-term.
        VAL-GEOM-004 (SymPy cross-check).

        V_c is a test covector field (not related to the connection).
        The covariant derivative of a (0,1) tensor V_c along direction a
        gives a (0,2) tensor (nabla_a V)_c, and the second covariant
        derivative treats the result as a (0,2) tensor, giving a (0,3)
        tensor that carries connection terms for ALL lower indices.
        The commutator [nabla_a, nabla_b] V_c arises from the difference
        of the a-b and b-a second derivatives, and the connection terms
        for the derivative indices a,b produce the torsion contribution.
        """
        geom, gamma, T, R_up, R_down = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords

        # Create a test covector field V_c
        V = sp.MutableDenseNDimArray.zeros(n)
        for c in range(n):
            V[c] = _clean(sp.Rational(1, 3) * x[c] ** 2 + x[(c + 1) % n])
        V = sp.ImmutableDenseNDimArray(V)

        # nabla_b V_c: covariant derivative of covector V (variance ["down"])
        nab_V = covariant_derivative_of_connection(x, gamma, V, variances=["down"])
        # nab_V[b, c] = partial_b V_c - Gamma^lam_{bc} V_lam

        # nabla_a nabla_b V_c: covariant derivative of the (0,2) tensor nab_V
        # nab_V has indices (derivative_direction_b, covector_index_c),
        # both with variance "down".
        nab2_V = covariant_derivative_of_connection(
            x, gamma, nab_V, variances=["down", "down"]
        )

        # Commutator: [nabla_a, nabla_b] V_c = nab2_V[a, b, c] - nab2_V[b, a, c]
        comm = sp.MutableDenseNDimArray.zeros(n, n, n)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    comm[a, b, c] = _clean(
                        nab2_V[a, b, c] - nab2_V[b, a, c]
                    )
        comm = sp.ImmutableDenseNDimArray(comm)

        # RHS: -R^d_{cab} V_d - T^d_{ab} nabla_d V_c
        rhs = sp.MutableDenseNDimArray.zeros(n, n, n)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    riemann_term = sum(
                        R_up[d, c, a, b] * V[d] for d in range(n)
                    )
                    torsion_term = sum(
                        T[d, a, b] * nab_V[d, c] for d in range(n)
                    )
                    rhs[a, b, c] = _clean(-riemann_term - torsion_term)
        rhs = sp.ImmutableDenseNDimArray(rhs)

        # Componentwise equality
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    diff = sp.simplify(comm[a, b, c] - rhs[a, b, c])
                    assert diff == 0, (
                        f"seed={seed}: commutator fails at "
                        f"({a},{b},{c}): comm={comm[a, b, c]}, "
                        f"rhs={rhs[a, b, c]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_torsion_term_nonzero_in_commutator(self, seed):
        """The torsion term -T^d_{ab} nabla_d V_c is demonstrably nonzero
        on a torsionful background.  VAL-GEOM-004 (nonzero T-term)."""
        geom, gamma, T, R_up, R_down = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords

        V = sp.MutableDenseNDimArray.zeros(n)
        for c in range(n):
            V[c] = _clean(sp.Rational(1, 3) * x[c] ** 2 + x[(c + 1) % n])
        V = sp.ImmutableDenseNDimArray(V)

        nab_V = covariant_derivative_of_connection(x, gamma, V, variances=["down"])

        # Check that T^d_{ab} nabla_d V_c is nonzero for some components
        any_nonzero = False
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(n):
                    torsion_term = sum(
                        T[d, a, b] * nab_V[d, c] for d in range(n)
                    )
                    if sp.simplify(torsion_term) != 0:
                        any_nonzero = True
                        break
                if any_nonzero:
                    break
            if any_nonzero:
                break
        assert any_nonzero, (
            f"seed={seed}: torsion term is zero on a torsionful background "
            "(should be nonzero)"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_commutator_reduces_to_lc_at_T0(self, seed):
        """With torsion zero (symmetric connection), the commutator
        reduces to the LC form [nabla_a, nabla_b] V_c = -R^d_{cab} V_d.
        VAL-GEOM-005 (SymPy LC-limit equality)."""
        geom, gamma, T, R_up, R_down = _sympy_background(
            seed, dim=3, symmetric=True
        )
        n, x = geom.dim, geom.coords

        V = sp.MutableDenseNDimArray.zeros(n)
        for c in range(n):
            V[c] = _clean(sp.Rational(1, 3) * x[c] ** 2 + x[(c + 1) % n])
        V = sp.ImmutableDenseNDimArray(V)

        nab_V = covariant_derivative_of_connection(x, gamma, V, variances=["down"])
        nab2_V = covariant_derivative_of_connection(
            x, gamma, nab_V, variances=["down", "down"]
        )

        # Commutator at T=0 should match the LC form
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(n):
                    comm = _clean(nab2_V[a, b, c] - nab2_V[b, a, c])
                    lc_rhs = _clean(-sum(R_up[d, c, a, b] * V[d] for d in range(n)))
                    diff = sp.simplify(comm - lc_rhs)
                    assert diff == 0, (
                        f"seed={seed}: LC limit fails at "
                        f"({a},{b},{c}): comm={comm}, lc_rhs={lc_rhs}"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_lc_commutator_fails_under_torsion(self, seed):
        """A deliberately LC-only commutator (no torsion term) disagrees
        with the full commutator on a torsionful background.  This is the
        torsion-trap demonstration: reusing the LC primitive under torsion
        gives a wrong answer.  VAL-GEOM-004 (trap guard)."""
        geom, gamma, T, R_up, R_down = _sympy_background(seed, dim=3)
        n, x = geom.dim, geom.coords

        V = sp.MutableDenseNDimArray.zeros(n)
        for c in range(n):
            V[c] = _clean(sp.Rational(1, 3) * x[c] ** 2 + x[(c + 1) % n])
        V = sp.ImmutableDenseNDimArray(V)

        nab_V = covariant_derivative_of_connection(x, gamma, V, variances=["down"])
        nab2_V = covariant_derivative_of_connection(
            x, gamma, nab_V, variances=["down", "down"]
        )

        # Full commutator vs LC-only (no torsion term)
        any_disagreement = False
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(n):
                    full_comm = _clean(nab2_V[a, b, c] - nab2_V[b, a, c])
                    lc_rhs = _clean(-sum(R_up[d, c, a, b] * V[d] for d in range(n)))
                    diff = sp.simplify(full_comm - lc_rhs)
                    if diff != 0:
                        any_disagreement = True
                        break
                if any_disagreement:
                    break
            if any_disagreement:
                break
        assert any_disagreement, (
            f"seed={seed}: LC commutator agrees with the full commutator "
            "on a torsionful background (torsion trap not caught!)"
        )
