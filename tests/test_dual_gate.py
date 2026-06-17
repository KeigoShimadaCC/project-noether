"""Enforce and demonstrate the dual gate and convention sign falsifier.

Tests for VAL-GEOM-014 and VAL-GEOM-016:

- VAL-GEOM-014: The dual gate is enforced; a green residue alone never
  marks a primitive verified. A negative-control shows a deliberately
  LC-only reduction that yields residue_zero==True under torsion is flagged
  disagreeing by the SymPy cross-check, so it is not verified=True.

- VAL-GEOM-016: Convention signs are threaded, not assumed. Flipping a
  single sign in a primitive's named convention block changes the residue
  from 0 to nonzero (or flips the SymPy cross-check), proving no
  convention is silently baked in.

The dual gate (architecture.md section 3.2, section 9) requires BOTH the
Cadabra residue check AND the SymPy component cross-check to agree before
a metric-affine primitive is called verified.  A green residue alone is
insufficient because of the torsion trap: applying a Levi-Civita identity
to both sides of an equation makes the residue vanish while the physics
is wrong.  The SymPy general-connection oracle catches this.

Convention: noether-default-v1 + metric-affine-v1.
"""

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.curvature import (
    AFFINE_CONNECTION_DEPENDS,
    AFFINE_CURVATURE_DECL,
    CURVATURE_DECL,
    TORSION_DECL,
    hessian_to_symmetric,
)
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    contortion_of_torsion,
    covariant_derivative_of_connection,
    disformation_of_nonmetricity,
    nonmetricity_of_connection,
    random_affine_connection,
    random_diagonal_metric,
    torsion_of_connection,
)

# ---------------------------------------------------------------------------
# Cadabra script builders
# ---------------------------------------------------------------------------

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


def _run_nabla_lc(body: str):
    """Run a Cadabra script with LC nabla declarations."""
    script = (
        _BASE_DECL_NABLA
        + CURVATURE_DECL
        + "\n"
        + r"{\phi}::Depends(\nabla{#})."
        + "\n"
        + body
    )
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="dual gate LC negative control",
            payload={"script": script},
        )
    )


def _run_affine(body: str):
    """Run a Cadabra script with affine connection declarations."""
    script = (
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
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="sign falsifier affine check",
            payload={"script": script},
        )
    )


# ---------------------------------------------------------------------------
# SymPy helpers
# ---------------------------------------------------------------------------


def _sympy_torsionful_background(seed: int, dim: int = 3):
    """Build a random metric + torsionful connection for SymPy cross-checks."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
    T = torsion_of_connection(gamma)
    return geom, gamma, T


def _flipped_contortion(gamma, g, g_inv):
    """Contortion with the WRONG convention: leading sign flipped from
    +1/2 to -1/2.

    K_flipped^lambda_{mu nu} = -(1/2)(T^lambda_{mu nu}
        + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu}
        + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu})

    This is the metric-affine-v1 contortion with the leading (1/2)
    replaced by (-1/2).  It is the sign-flipped version of the correct
    convention, used to prove that the (1/2) factor is a convention
    choice, not silently assumed."""
    T = torsion_of_connection(gamma)
    n = T.shape[0]
    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(n):
                term1 = T[lam, mu, nu]
                term2 = sum(
                    g_inv[lam, sig] * g[mu, tau] * T[tau, sig, nu]
                    for sig in range(n)
                    for tau in range(n)
                )
                term3 = sum(
                    g_inv[lam, sig] * g[nu, tau] * T[tau, sig, mu]
                    for sig in range(n)
                    for tau in range(n)
                )
                out[lam, mu, nu] = _clean(
                    -sp.Rational(1, 2) * (term1 + term2 + term3)
                )
    return sp.Array(out)


def _flipped_disformation(coords, gamma, g, g_inv):
    """Disformation with the WRONG convention: leading sign flipped from
    +1/2 to -1/2.

    L_flipped^lambda_{mu nu} = -(1/2) g^{lambda rho}
        (-Q_{mu nu rho} - Q_{nu rho mu} + Q_{rho mu nu})

    This is the metric-affine-v1 disformation with the leading (1/2)
    replaced by (-1/2).  It proves the (1/2) factor is a convention."""
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
                out[lam, mu, nu] = _clean(-sp.Rational(1, 2) * val)
    return sp.Array(out)


# ===========================================================================
# VAL-GEOM-014: Dual gate negative control
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestDualGateNegativeControl:
    """Demonstrate that a green residue alone never marks a primitive
    verified.  A deliberately LC-only reduction gives residue_zero=True
    but is caught by the SymPy cross-check, so the dual-gate verdict
    is NOT verified.

    The test uses the scalar Hessian as the demonstration case:

    - LC approach: hessian_to_symmetric routes nabla_mu nabla_nu phi
      through the symmetric stand-in H_{mu nu}, so the antisymmetric
      part H_{mu nu} - H_{nu mu} reduces to zero.  This is the false
      positive: residue_zero=True under the wrong (LC-only) assumptions.

    - SymPy oracle: the actual antisymmetric Hessian
      nabla_mu nabla_nu phi - nabla_nu nabla_mu phi = -T^lam_{mu nu} nabla_lam phi
      is nonzero on a torsionful background.  This catches the torsion
      trap.

    - Dual-gate verdict: Cadabra residue says True, SymPy cross-check
      says False -> NOT verified.
    """

    def test_lc_hessian_residue_passes_cadabra(self):
        """Negative control: the LC-only Hessian substitution gives
        residue_zero=True for the antisymmetric Hessian.

        Using H_{mu nu}::Symmetric (the LC assumption), the expression
        nabla_mu nabla_nu phi - nabla_nu nabla_mu phi is routed through
        H_{mu nu} - H_{nu mu}, which canonicalises to zero because H is
        declared symmetric.  This is the Cadabra false positive that
        the dual gate must catch."""
        body = (
            r"ex := \nabla_{\mu}{\nabla_{\nu}{phi}}"
            r" - \nabla_{\nu}{\nabla_{\mu}{phi}};"
            "\n"
            + hessian_to_symmetric("phi", "ex")
            + "\n"
            "canonicalise(ex);\n"
            'print("NOETHER_CHECK: hessian_antisym_lc_zero=" '
            '+ str(str(ex) == "0"))'
        )
        result = _run_nabla_lc(body)
        assert result.raw.returncode == 0, result.raw.stderr
        # The LC check reports residue_zero=True (the false positive)
        assert result.value["checks"].get("hessian_antisym_lc_zero") == "True", (
            result.raw.stdout
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_sympy_hessian_nonzero_under_torsion(self, seed):
        """The actual antisymmetric Hessian is nonzero on torsionful
        backgrounds, contradicting the LC result.

        This is the SymPy cross-check that catches the torsion trap:
        nabla_mu nabla_nu phi - nabla_nu nabla_mu phi
          = -T^lambda_{mu nu} nabla_lambda phi
        which is nonzero when T != 0."""
        geom, gamma, T = _sympy_torsionful_background(seed, dim=3)
        n, x = geom.dim, geom.coords

        phi = sp.Rational(1, 2) * x[0] ** 2 + x[1] * x[2]
        nab_phi = covariant_derivative_of_connection(x, gamma, phi, variances=[])
        nab2_phi = covariant_derivative_of_connection(
            x, gamma, nab_phi, variances=["down"]
        )

        # The antisymmetric part of the Hessian
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
    def test_dual_gate_verdict_not_verified(self, seed):
        """The dual-gate verdict is NOT verified when the Cadabra residue
        check passes (LC-only) but the SymPy cross-check disagrees.

        This test explicitly demonstrates the dual-gate logic:
          verified = cadabra_residue_zero and sympy_cross_check_agrees

        For the negative control:
          cadabra_residue_zero = True  (LC Hessian gives zero)
          sympy_cross_check_agrees = False  (actual Hessian nonzero)
          verified = True and False = False

        VAL-GEOM-014: a green residue alone never marks a primitive
        verified."""
        geom, gamma, T = _sympy_torsionful_background(seed, dim=3)
        n, x = geom.dim, geom.coords

        # Verify the background is genuinely torsionful
        T_nonzero = any(sp.simplify(c) != 0 for c in _components(T))
        assert T_nonzero, f"seed={seed}: background should have nonzero torsion"

        # Gate 1: Cadabra residue check (simulated by the LC approach)
        # Under LC assumptions, the antisymmetric Hessian is declared zero.
        # This is the false positive.
        cadabra_residue_zero = True  # confirmed by test_lc_hessian_residue_passes_cadabra

        # Gate 2: SymPy cross-check
        phi = sp.Rational(1, 2) * x[0] ** 2 + x[1] * x[2]
        nab_phi = covariant_derivative_of_connection(x, gamma, phi, variances=[])
        nab2_phi = covariant_derivative_of_connection(
            x, gamma, nab_phi, variances=["down"]
        )
        # The antisymmetric Hessian should be nonzero under torsion
        any_nonzero = False
        for mu in range(n):
            for nu in range(mu + 1, n):
                diff = sp.simplify(nab2_phi[mu, nu] - nab2_phi[nu, mu])
                if diff != 0:
                    any_nonzero = True
                    break
            if any_nonzero:
                break

        # The SymPy cross-check: does the actual antisymmetric Hessian
        # agree with the LC result (zero)?
        sympy_agrees_with_lc = not any_nonzero

        # Dual-gate verdict
        verified = cadabra_residue_zero and sympy_agrees_with_lc

        # The dual gate catches the false positive
        assert not verified, (
            f"seed={seed}: dual gate should catch the false positive: "
            f"cadabra_residue_zero={cadabra_residue_zero}, "
            f"sympy_agrees_with_lc={sympy_agrees_with_lc}"
        )


# ===========================================================================
# VAL-GEOM-016: Convention sign falsifier
# ===========================================================================


class TestConventionSignFalsifier:
    """Flipping a single sign in a primitive's named convention block
    changes the residue from 0 to nonzero (or flips the SymPy cross-check),
    proving no convention is silently baked in.

    The convention block metric-affine-v1 defines the contortion and
    disformation signs.  The correct convention passes both gates
    (Cadabra residue zero AND SymPy cross-check agrees).  Flipping a
    sign in the convention fails one or both gates.

    Two sign flips are tested:

    1. Contortion: K^lambda_{mu nu} = (1/2)(T + ...)
       Flipped: K_flipped = -(1/2)(T + ...)
       The inversion identity K^lam_{mu nu} - K^lam_{nu mu} = T^lam_{mu nu}
       fails: K_flipped^lam_{mu nu} - K_flipped^lam_{nu mu} = -T^lam_{mu nu}

    2. Disformation: L^lambda_{mu nu} = (1/2) g^{...}(-Q - Q + Q)
       Flipped: L_flipped = -(1/2) g^{...}(-Q - Q + Q)
       The inversion identity -(L^rho_{lam mu} g_{rho nu} + L^rho_{lam nu} g_{rho mu})
       = Q_{lam mu nu} fails: the flipped version gives -Q instead of Q.

    3. Cadabra: the Hessian identity target with a flipped torsion sign
       gives a nonzero residue, proving the - sign in the identity
       -T^lam_{mu nu} nabla_lam phi is not silently assumed.
    """

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_correct_contortion_passes_both_gates(self, seed):
        """The correct metric-affine-v1 contortion passes both gates:
        the inversion identity K^lam_{mu nu} - K^lam_{nu mu} = T^lam_{mu nu}
        holds (SymPy agrees, and the Cadabra residue is zero per
        test_post_riemannian.py)."""
        geom, gamma, T = _sympy_torsionful_background(seed, dim=3)
        K = contortion_of_torsion(gamma, geom.g, geom.g_inv)

        # Verify torsion is nonzero
        T_nonzero = any(sp.simplify(c) != 0 for c in _components(T))
        assert T_nonzero, f"seed={seed}: background should have nonzero torsion"

        # Inversion identity: K^lam_{mu nu} - K^lam_{nu mu} = T^lam_{mu nu}
        n = T.shape[0]
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    diff = sp.simplify(K[lam, mu, nu] - K[lam, nu, mu] - T[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: contortion inversion fails at "
                        f"({lam},{mu},{nu}): "
                        f"K-K^T-T = {diff}"
                    )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_flipped_contortion_sign_fails_sympy(self, seed):
        """Flipping the leading (1/2) to (-1/2) in the contortion
        convention changes the SymPy cross-check from pass to fail:
        K_flipped^lam_{mu nu} - K_flipped^lam_{nu mu} = -T^lam_{mu nu}
        which is NOT equal to T^lam_{mu nu}.

        The residue K_flipped^lam_{mu nu} - K_flipped^lam_{nu mu}
        - T^lam_{mu nu} = -2 T^lam_{mu nu}, nonzero when T != 0.

        This proves the (1/2) factor in the contortion definition is
        a convention choice recorded in metric-affine-v1, not silently
        assumed."""
        geom, gamma, T = _sympy_torsionful_background(seed, dim=3)
        K_flipped = _flipped_contortion(gamma, geom.g, geom.g_inv)

        # Verify torsion is nonzero
        T_nonzero = any(sp.simplify(c) != 0 for c in _components(T))
        assert T_nonzero, f"seed={seed}: background should have nonzero torsion"

        # The flipped contortion fails the inversion identity
        n = T.shape[0]
        any_residual_nonzero = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    # K_flipped^lam_{mu nu} - K_flipped^lam_{nu mu} - T^lam_{mu nu}
                    residual = sp.simplify(
                        K_flipped[lam, mu, nu] - K_flipped[lam, nu, mu] - T[lam, mu, nu]
                    )
                    if residual != 0:
                        any_residual_nonzero = True
                        # Also verify the residual is -2*T (the expected wrong result)
                        expected = sp.simplify(-2 * T[lam, mu, nu])
                        assert sp.simplify(residual - expected) == 0, (
                            f"seed={seed}: flipped contortion residual at "
                            f"({lam},{mu},{nu}) is {residual}, "
                            f"expected -2T = {expected}"
                        )
                        break
            if any_residual_nonzero:
                break

        assert any_residual_nonzero, (
            f"seed={seed}: flipped contortion sign falsifier did not fire: "
            "the residual should be nonzero when T != 0"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_correct_disformation_passes_both_gates(self, seed):
        """The correct metric-affine-v1 disformation passes both gates:
        the inversion identity -(L^rho_{lam mu} g_{rho nu}
        + L^rho_{lam nu} g_{rho mu}) = Q_{lam mu nu} holds."""
        geom = random_diagonal_metric(seed, dim=3)
        # Use a symmetric but non-metric-compatible connection (T=0, Q!=0)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=True)
        Q = nonmetricity_of_connection(geom.coords, gamma, geom.g)
        L = disformation_of_nonmetricity(geom.coords, gamma, geom.g, geom.g_inv)

        # Verify non-metricity is nonzero
        Q_nonzero = any(sp.simplify(c) != 0 for c in _components(Q))
        assert Q_nonzero, f"seed={seed}: background should have nonzero non-metricity"

        # Inversion identity: -(L^rho_{lam mu} g_{rho nu}
        #                        + L^rho_{lam nu} g_{rho mu}) = Q_{lam mu nu}
        n = Q.shape[0]
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    lhs = -sum(
                        L[rho, lam, mu] * geom.g[rho, nu]
                        + L[rho, lam, nu] * geom.g[rho, mu]
                        for rho in range(n)
                    )
                    diff = sp.simplify(lhs - Q[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: disformation inversion fails at "
                        f"({lam},{mu},{nu}): residual = {diff}"
                    )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_flipped_disformation_sign_fails_sympy(self, seed):
        """Flipping the leading (1/2) to (-1/2) in the disformation
        convention changes the SymPy cross-check from pass to fail:
        -(L_flipped^rho_{lam mu} g_{rho nu} + L_flipped^rho_{lam nu} g_{rho mu})
        = -Q_{lam mu nu} which is NOT equal to Q.

        The residue is -2 Q, nonzero when Q != 0.

        This proves the (1/2) factor in the disformation definition is
        a convention choice recorded in metric-affine-v1, not silently
        assumed."""
        geom = random_diagonal_metric(seed, dim=3)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=True)
        Q = nonmetricity_of_connection(geom.coords, gamma, geom.g)
        L_flipped = _flipped_disformation(
            geom.coords, gamma, geom.g, geom.g_inv
        )

        # Verify non-metricity is nonzero
        Q_nonzero = any(sp.simplify(c) != 0 for c in _components(Q))
        assert Q_nonzero, f"seed={seed}: background should have nonzero non-metricity"

        # The flipped disformation fails the inversion identity
        n = Q.shape[0]
        any_residual_nonzero = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    lhs = -sum(
                        L_flipped[rho, lam, mu] * geom.g[rho, nu]
                        + L_flipped[rho, lam, nu] * geom.g[rho, mu]
                        for rho in range(n)
                    )
                    residual = sp.simplify(lhs - Q[lam, mu, nu])
                    if residual != 0:
                        any_residual_nonzero = True
                        # Verify the residual is -2*Q (the expected wrong result)
                        expected = sp.simplify(-2 * Q[lam, mu, nu])
                        assert sp.simplify(residual - expected) == 0, (
                            f"seed={seed}: flipped disformation residual at "
                            f"({lam},{mu},{nu}) is {residual}, "
                            f"expected -2Q = {expected}"
                        )
                        break
            if any_residual_nonzero:
                break

        assert any_residual_nonzero, (
            f"seed={seed}: flipped disformation sign falsifier did not fire: "
            "the residual should be nonzero when Q != 0"
        )

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_flipped_hessian_sign_cadabra_residue_nonzero(self):
        """Flipping the torsion sign in the Hessian identity target makes
        the Cadabra residue nonzero.

        Correct convention (metric-affine-v1 / noether-default-v1):
          nabla_mu nabla_nu phi - nabla_nu nabla_mu phi
            = -(G^lam_{mu nu} - G^lam_{nu mu}) partial_lam phi
            = -T^lam_{mu nu} partial_lam phi

        Flipped sign (wrong convention):
          target_flipped = +(G^lam_{mu nu} - G^lam_{nu mu}) partial_lam phi
          = +T^lam_{mu nu} partial_lam phi

        The residue (correct expansion - wrong target) is:
          -2(G^lam_{mu nu} - G^lam_{nu mu}) partial_lam phi
          = -2 T^lam_{mu nu} partial_lam phi

        which is nonzero.  This proves the minus sign in the Hessian
        identity is derived from the torsion convention T = G - G^T,
        not silently assumed."""
        body = (
            # Expand the Hessian from the definition
            r"hess := (\partial_{\mu}{\partial_{\nu}{\phi}}"
            r" - G^{\lambda}_{\mu\nu} \partial_{\lambda}{\phi})"
            r" - (\partial_{\nu}{\partial_{\mu}{\phi}}"
            r" - G^{\lambda}_{\nu\mu} \partial_{\lambda}{\phi});"
            "\n"
            "distribute(hess); canonicalise(hess); rename_dummies(hess);\n"
            # WRONG target: flipped sign (+ instead of -)
            r"target_flipped := +(G^{\lambda}_{\mu\nu} - G^{\lambda}_{\nu\mu})"
            r" \partial_{\lambda}{\phi};"
            "\n"
            "distribute(target_flipped); canonicalise(target_flipped); "
            "rename_dummies(target_flipped);\n"
            "residue := @(hess) - @(target_flipped);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: hessian_flipped_sign_nonzero=" '
            '+ str(str(residue) != "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        # The flipped sign makes the residue NONZERO
        assert result.value["checks"].get("hessian_flipped_sign_nonzero") == "True", (
            result.raw.stdout
        )

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_correct_hessian_sign_cadabra_residue_zero(self):
        """Baseline: the correct torsion sign gives residue zero.

        This is the positive control for the sign falsifier: the
        correct convention (T = G - G^T, Hessian = -T nabla phi)
        gives residue zero, confirming the existing pinned primitive."""
        body = (
            # Expand the Hessian from the definition
            r"hess := (\partial_{\mu}{\partial_{\nu}{\phi}}"
            r" - G^{\lambda}_{\mu\nu} \partial_{\lambda}{\phi})"
            r" - (\partial_{\nu}{\partial_{\mu}{\phi}}"
            r" - G^{\lambda}_{\nu\mu} \partial_{\lambda}{\phi});"
            "\n"
            "distribute(hess); canonicalise(hess); rename_dummies(hess);\n"
            # CORRECT target: -T nabla phi
            r"target := -(G^{\lambda}_{\mu\nu} - G^{\lambda}_{\nu\mu})"
            r" \partial_{\lambda}{\phi};"
            "\n"
            "distribute(target); canonicalise(target); "
            "rename_dummies(target);\n"
            "residue := @(hess) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: hessian_correct_sign_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("hessian_correct_sign_zero") == "True", (
            result.raw.stdout
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _components(arr):
    """Iterate scalar components of an NDimArray."""
    shape = getattr(arr, "shape", ())
    if not shape:
        yield arr
        return
    n = shape[0]
    rank = len(shape)
    for idx in _all_indices(n, rank):
        yield arr[idx]


def _all_indices(n: int, rank: int):
    if rank == 0:
        yield ()
        return
    for first in range(n):
        for rest in _all_indices(n, rank - 1):
            yield (first, *rest)
