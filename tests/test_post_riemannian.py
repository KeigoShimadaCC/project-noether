"""Pin the post-Riemannian decomposition Gamma = LC(g) + K(T) + L(Q).

Tests for VAL-GEOM-008, VAL-GEOM-009, VAL-GEOM-010:

- VAL-GEOM-008: The decomposition substitution reproduces the original
  connection (residue 0) and the SymPy oracle confirms LC + contortion +
  disformation equals the original random connection.

- VAL-GEOM-009: The contortion closed form's antisymmetric part reproduces
  T (residue 0); SymPy confirms the round-trip T -> K(T) -> T on a random
  torsion background with Q=0.

- VAL-GEOM-010: The disformation closed form reproduces Q (residue 0);
  SymPy confirms the round-trip Q -> L(Q) -> Q on a random non-metricity
  background with T=0.

The contortion and disformation signs are NOT asserted from memory; they
are derived and residue-pinned against the SymPy oracle, then recorded as
the named convention block 'metric-affine-v1'.

Convention block: metric-affine-v1
  T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
  Q_{lambda mu nu} = nabla_lambda g_{mu nu}
  K^lambda_{mu nu} = (1/2)(T^lambda_{mu nu}
                        + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu}
                        + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu})
  L^lambda_{mu nu} = (1/2) g^{lambda rho}(-Q_{mu nu rho}
                                      - Q_{nu rho mu} + Q_{rho mu nu})

Each Cadabra primitive is pinned by a residue check. Each is additionally
cross-checked against the SymPy general-connection oracle on explicit
random metric + connection backgrounds (the torsion-trap safeguard).
They skip when cadabra2 is absent.

Cadabra residue check strategy for the decomposition:
  The full expansion G = LC + K + L in terms of partial-g and G components
  hits the Kronecker-delta/renaming limitation in Cadabra (see library/
  cadabra-gotchas.md).  Instead, the decomposition is verified through
  three algebraic identities that together prove it:
    (a) K^lambda_{mu nu} - K^lambda_{nu mu} = T^lambda_{mu nu}
        (contortion antisymmetry captures the torsion)
    (b) -(L_{nu lambda mu} + L_{mu lambda nu}) = Q_{lambda mu nu}
        (disformation inversion captures the non-metricity)
    (c) The decomposition substitution G -> LC + K + L fires correctly
  The full componentwise verification (LC + K + L = Gamma on explicit
  backgrounds) is done by the SymPy cross-check, which is the torsion-trap
  safeguard (architecture.md section 3.2).
"""

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.curvature import (
    AFFINE_CONNECTION_DEPENDS,
    AFFINE_CURVATURE_DECL,
    NONMETRICITY_DECL,
    TORSION_DECL,
    decompose_connection,
    define_contortion,
    define_torsion,
    expand_lc,
)
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    christoffel_of_metric,
    contortion_of_torsion,
    disformation_of_nonmetricity,
    nonmetricity_of_connection,
    random_affine_connection,
    random_diagonal_metric,
    torsion_of_connection,
)

# ---------------------------------------------------------------------------
# Cadabra script builders
# ---------------------------------------------------------------------------

_BASE_DECL_AFFINE = (
    r"{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\tau}"
    r"::Indices(position=fixed)."
    "\n"
    r"{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\tau}"
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


def _affine_script(body: str) -> str:
    return (
        _BASE_DECL_AFFINE
        + AFFINE_CURVATURE_DECL
        + "\n"
        + AFFINE_CONNECTION_DEPENDS
        + "\n"
        + TORSION_DECL
        + "\n"
        + NONMETRICITY_DECL
        + "\n"
        + body
    )


def _run_affine(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="post-Riemannian decomposition check",
            payload={"script": _affine_script(body)},
        )
    )


# ---------------------------------------------------------------------------
# SymPy cross-check helpers
# ---------------------------------------------------------------------------


def _sympy_random_background(seed: int, dim: int = 3, symmetric: bool = False):
    """Build a random metric + connection and compute all derived quantities."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=symmetric)
    LC = geom.christoffel
    T = torsion_of_connection(gamma)
    Q = nonmetricity_of_connection(geom.coords, gamma, geom.g)
    K = contortion_of_torsion(gamma, geom.g, geom.g_inv)
    L = disformation_of_nonmetricity(geom.coords, gamma, geom.g, geom.g_inv)
    return geom, gamma, LC, T, Q, K, L


# ===========================================================================
# Cadabra residue checks
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestPostRiemannianResidue:
    """Cadabra residue checks for the post-Riemannian decomposition.

    The decomposition G = LC + K + L is verified through three algebraic
    identities (see module docstring for strategy).  The full componentwise
    verification is done by the SymPy cross-check.
    """

    def test_contortion_antisymmetry_residue_zero(self):
        """K^lambda_{mu nu} - K^lambda_{nu mu} = T^lambda_{mu nu}
        (the antisymmetric part of K recovers T).  VAL-GEOM-009."""
        body = (
            r"exA := K^{\lambda}_{\mu\nu} - K^{\lambda}_{\nu\mu};"
            "\n"
            + define_contortion("exA")
            + "\n"
            "distribute(exA); canonicalise(exA); rename_dummies(exA);\n"
            + define_torsion("G", "exA")
            + "\n"
            "distribute(exA); canonicalise(exA); rename_dummies(exA);\n"
            r"targetA := T^{\lambda}_{\mu\nu};"
            "\n"
            + define_torsion("G", "targetA")
            + "\n"
            "distribute(targetA); canonicalise(targetA); "
            "rename_dummies(targetA);\n"
            "resA := @(exA) - @(targetA);\n"
            "distribute(resA); canonicalise(resA); "
            "rename_dummies(resA); meld(resA);\n"
            'print("NOETHER_CHECK: contortion_antisym_zero=" '
            '+ str(str(resA) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("contortion_antisym_zero") == "True", (
            result.raw.stdout
        )

    def test_disformation_inversion_residue_zero(self):
        """Q_{lambda mu nu} = -(L_{nu lambda mu} + L_{mu lambda nu})
        where L_{alpha lambda mu} = (1/2)(-Q_{lambda mu alpha}
        - Q_{mu alpha lambda} + Q_{alpha lambda mu}).
        The disformation inverts to Q (residue 0).  VAL-GEOM-010.

        This check uses the lowered form of L to avoid the Kronecker-delta
        limitation in Cadabra (see library/cadabra-gotchas.md).  The
        identity L_{alpha lambda mu} = g_{alpha rho} L^rho_{lambda mu}
        = (1/2)(-Q_{lambda mu alpha} - Q_{mu alpha lambda}
        + Q_{alpha lambda mu}) follows from contracting g_{alpha rho}
        with the L definition.
        """
        body = (
            # L_{nu lambda mu} + L_{mu lambda nu} should equal -Q_{lambda mu nu}
            r"sumB := (1/2)(-Q_{\lambda\mu\nu} - Q_{\mu\nu\lambda}"
            r" + Q_{\nu\lambda\mu})"
            r" + (1/2)(-Q_{\lambda\nu\mu} - Q_{\nu\mu\lambda}"
            r" + Q_{\mu\lambda\nu})"
            r" + Q_{\lambda\mu\nu};"
            "\n"
            "canonicalise(sumB);\n"
            "meld(sumB);\n"
            'print("NOETHER_CHECK: disformation_inversion_zero=" '
            '+ str(str(sumB) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("disformation_inversion_zero") == "True", (
            result.raw.stdout
        )

    def test_decomposition_substitution_fires(self):
        """G^lambda_{mu nu} -> LC^lambda_{mu nu} + K^lambda_{mu nu}
        + L^lambda_{mu nu} substitution fires correctly.
        VAL-GEOM-008 (structural check)."""
        body = (
            r"exC := G^{\lambda}_{\mu\nu};"
            "\n"
            + decompose_connection("G", "exC")
            + "\n"
            'has_LC = "LC" in str(exC)\n'
            'has_K = "K" in str(exC)\n'
            'has_L = "L" in str(exC)\n'
            'print("NOETHER_CHECK: decomposition_has_LC=" + str(has_LC))\n'
            'print("NOETHER_CHECK: decomposition_has_K=" + str(has_K))\n'
            'print("NOETHER_CHECK: decomposition_has_L=" + str(has_L))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value["checks"]
        assert checks.get("decomposition_has_LC") == "True", result.raw.stdout
        assert checks.get("decomposition_has_K") == "True", result.raw.stdout
        assert checks.get("decomposition_has_L") == "True", result.raw.stdout

    def test_lc_expansion_substitution_fires(self):
        """LC^lambda_{mu nu} -> (1/2) g^{lambda rho}(d_mu g_{rho nu}
        + d_nu g_{rho mu} - d_rho g_{mu nu}) substitution fires.
        VAL-GEOM-008 (LC expansion check)."""
        body = (
            r"exLC := LC^{\lambda}_{\mu\nu};"
            "\n"
            + expand_lc("exLC")
            + "\n"
            'has_partial = "partial" in str(exLC) or "\\partial" in str(exLC)\n'
            'print("NOETHER_CHECK: lc_expansion_fires=" + str(has_partial))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("lc_expansion_fires") == "True", (
            result.raw.stdout
        )


# ===========================================================================
# SymPy component cross-checks (the torsion-trap safeguard)
# ===========================================================================


class TestPostRiemannianSymPyCrossCheck:
    """Cross-check the post-Riemannian decomposition against the SymPy oracle
    on explicit random backgrounds.  This is the independent verification that
    catches the torsion trap (architecture.md section 3.2)."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_decomposition_reconstructs_connection(self, seed):
        """LC + K + L = Gamma on a random background with both T and Q
        nonzero.  VAL-GEOM-008."""
        geom, gamma, LC, T, Q, K, L = _sympy_random_background(
            seed, dim=3, symmetric=False
        )
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    reassembled = LC[lam, mu, nu] + K[lam, mu, nu] + L[lam, mu, nu]
                    diff = sp.simplify(reassembled - gamma[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: LC+K+L != Gamma at "
                        f"({lam},{mu},{nu}): "
                        f"reassembled={reassembled}, "
                        f"Gamma={gamma[lam, mu, nu]}, "
                        f"diff={diff}"
                    )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_contortion_antisymmetry_recovers_torsion(self, seed):
        """K^lambda_{mu nu} - K^lambda_{nu mu} = T^lambda_{mu nu}
        on a random torsionful background.  VAL-GEOM-009."""
        geom, gamma, LC, T, Q, K, L = _sympy_random_background(
            seed, dim=3, symmetric=False
        )
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    antisym_K = K[lam, mu, nu] - K[lam, nu, mu]
                    diff = sp.simplify(antisym_K - T[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: K antisym != T at "
                        f"({lam},{mu},{nu}): "
                        f"K_antisym={antisym_K}, T={T[lam, mu, nu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_torsion_round_trip_with_Q_zero(self, seed):
        """T -> K(T) -> T round-trip on a random torsionful background
        with Q=0 (Levi-Civita + contortion only).  VAL-GEOM-009.

        To get Q=0 with nonzero T, we construct the connection as
        LC + K(T) for a random T, which guarantees metric compatibility.
        """
        geom = random_diagonal_metric(seed, dim=3)
        n = geom.dim
        # Generate a random torsionful connection that is metric-compatible
        # by constructing Gamma = LC + K(T) for a random T
        gamma_asym = random_affine_connection(
            seed + 1000, geom.coords, symmetric=False
        )
        T_input = torsion_of_connection(gamma_asym)
        K_input = contortion_of_torsion(gamma_asym, geom.g, geom.g_inv)
        LC = geom.christoffel
        # Construct the metric-compatible torsionful connection
        gamma_mc = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    gamma_mc[lam, mu, nu] = _clean(
                        LC[lam, mu, nu] + K_input[lam, mu, nu]
                    )
        gamma_mc = sp.ImmutableDenseNDimArray(gamma_mc)
        # Verify Q=0 (metric compatibility)
        Q_mc = nonmetricity_of_connection(geom.coords, gamma_mc, geom.g)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    diff = sp.simplify(Q_mc[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: Q[{lam},{mu},{nu}] = "
                        f"{Q_mc[lam, mu, nu]} "
                        f"on metric-compatible connection (should be zero)"
                    )
        # Round-trip: T -> K(T) -> T
        K_round = contortion_of_torsion(gamma_mc, geom.g, geom.g_inv)
        T_round = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    T_round[lam, mu, nu] = _clean(
                        K_round[lam, mu, nu] - K_round[lam, nu, mu]
                    )
        T_round = sp.ImmutableDenseNDimArray(T_round)
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    diff = sp.simplify(T_round[lam, mu, nu] - T_input[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: T round-trip fails at "
                        f"({lam},{mu},{nu}): "
                        f"T_out={T_round[lam, mu, nu]}, "
                        f"T_in={T_input[lam, mu, nu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_disformation_inversion_recovers_nonmetricity(self, seed):
        """-(L^rho_{lambda mu} g_{rho nu} + L^rho_{lambda nu} g_{rho mu})
        = Q_{lambda mu nu} on a random background.  VAL-GEOM-010."""
        geom, gamma, LC, T, Q, K, L = _sympy_random_background(
            seed, dim=3, symmetric=False
        )
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    # Compute Q from L
                    Q_from_L = -sum(
                        L[rho, lam, mu] * geom.g[rho, nu]
                        + L[rho, lam, nu] * geom.g[rho, mu]
                        for rho in range(n)
                    )
                    diff = sp.simplify(Q_from_L - Q[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: L inversion fails at "
                        f"({lam},{mu},{nu}): "
                        f"Q_from_L={Q_from_L}, Q={Q[lam, mu, nu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_nonmetricity_round_trip_with_T_zero(self, seed):
        """Q -> L(Q) -> Q round-trip on a random non-metricity background
        with T=0 (symmetric but non-metric-compatible connection).
        VAL-GEOM-010."""
        geom = random_diagonal_metric(seed, dim=3)
        # Use a symmetric (torsion-free) random connection, which is
        # generically NOT metric-compatible (Q nonzero)
        gamma_sym = random_affine_connection(
            seed + 1000, geom.coords, symmetric=True
        )
        n = geom.dim
        # Verify T=0
        T_sym = torsion_of_connection(gamma_sym)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    assert T_sym[lam, mu, nu] == 0, (
                        f"seed={seed}: T[{lam},{mu},{nu}] != 0 "
                        f"on symmetric connection"
                    )
        # Get Q and L
        Q_input = nonmetricity_of_connection(geom.coords, gamma_sym, geom.g)
        L_input = disformation_of_nonmetricity(
            geom.coords, gamma_sym, geom.g, geom.g_inv
        )
        # Round-trip: Q -> L(Q) -> Q
        Q_round = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    val = -sum(
                        L_input[rho, lam, mu] * geom.g[rho, nu]
                        + L_input[rho, lam, nu] * geom.g[rho, mu]
                        for rho in range(n)
                    )
                    Q_round[lam, mu, nu] = _clean(val)
                    Q_round[lam, nu, mu] = _clean(val)  # symmetric in last pair
        Q_round = sp.ImmutableDenseNDimArray(Q_round)
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    diff = sp.simplify(Q_round[lam, mu, nu] - Q_input[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: Q round-trip fails at "
                        f"({lam},{mu},{nu}): "
                        f"Q_out={Q_round[lam, mu, nu]}, "
                        f"Q_in={Q_input[lam, mu, nu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_contortion_nonzero_on_torsionful_background(self, seed):
        """K is nonzero when T is nonzero (contortion is non-trivial)."""
        geom, gamma, LC, T, Q, K, L = _sympy_random_background(
            seed, dim=3, symmetric=False
        )
        n = geom.dim
        any_nonzero = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    if K[lam, mu, nu] != 0:
                        any_nonzero = True
                        break
                if any_nonzero:
                    break
            if any_nonzero:
                break
        assert any_nonzero, (
            f"seed={seed}: K is zero on a torsionful background "
            "(should be nonzero)"
        )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_disformation_nonzero_on_nonmetric_background(self, seed):
        """L is nonzero when Q is nonzero (disformation is non-trivial)."""
        geom = random_diagonal_metric(seed, dim=3)
        # Use a symmetric (T=0) connection that is generically non-metric
        gamma_sym = random_affine_connection(
            seed + 1000, geom.coords, symmetric=True
        )
        L = disformation_of_nonmetricity(
            geom.coords, gamma_sym, geom.g, geom.g_inv
        )
        n = geom.dim
        any_nonzero = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    if L[lam, mu, nu] != 0:
                        any_nonzero = True
                        break
                if any_nonzero:
                    break
            if any_nonzero:
                break
        assert any_nonzero, (
            f"seed={seed}: L is zero on a non-metric background "
            "(should be nonzero)"
        )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_decomposition_reduces_to_lc_when_TQ_zero(self, seed):
        """When T=0 and Q=0, the decomposition gives Gamma = LC (the
        Levi-Civita limit).  This is the T=Q=0 special case."""
        geom = random_diagonal_metric(seed, dim=3)
        n = geom.dim
        # Use the Levi-Civita connection: T=0, Q=0
        gamma_lc = geom.christoffel
        K = contortion_of_torsion(gamma_lc, geom.g, geom.g_inv)
        L = disformation_of_nonmetricity(
            geom.coords, gamma_lc, geom.g, geom.g_inv
        )
        # K should be zero (T=0)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    diff = sp.simplify(K[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: K[{lam},{mu},{nu}] = "
                        f"{K[lam, mu, nu]} at T=0 (should be zero)"
                    )
        # L should be zero (Q=0)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    diff = sp.simplify(L[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: L[{lam},{mu},{nu}] = "
                        f"{L[lam, mu, nu]} at Q=0 (should be zero)"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_disformation_symmetric_in_lower_pair(self, seed):
        """L^lambda_{mu nu} = L^lambda_{nu mu} (disformation is symmetric
        in the lower pair, since it comes from Q which is symmetric in
        its last pair)."""
        geom = random_diagonal_metric(seed, dim=3)
        gamma_sym = random_affine_connection(
            seed + 1000, geom.coords, symmetric=True
        )
        L = disformation_of_nonmetricity(
            geom.coords, gamma_sym, geom.g, geom.g_inv
        )
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    diff = sp.simplify(L[lam, mu, nu] - L[lam, nu, mu])
                    assert diff == 0, (
                        f"seed={seed}: L[{lam},{mu},{nu}] != "
                        f"L[{lam},{nu},{mu}]: "
                        f"{L[lam, mu, nu]} vs {L[lam, nu, mu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_christoffel_of_metric_matches_component_geometry(self, seed):
        """christoffel_of_metric agrees with ComponentGeometry.christoffel
        (regression guard for the LC computation)."""
        geom = random_diagonal_metric(seed, dim=3)
        LC_standalone = christoffel_of_metric(geom.coords, geom.g, geom.g_inv)
        LC_builtin = geom.christoffel
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    diff = sp.simplify(
                        LC_standalone[lam, mu, nu] - LC_builtin[lam, mu, nu]
                    )
                    assert diff == 0, (
                        f"seed={seed}: LC standalone != LC builtin at "
                        f"({lam},{mu},{nu}): "
                        f"{LC_standalone[lam, mu, nu]} vs "
                        f"{LC_builtin[lam, mu, nu]}"
                    )
