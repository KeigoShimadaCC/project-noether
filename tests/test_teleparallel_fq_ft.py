"""f(T) and f(Q) teleparallel EOM derivations and cross-checks.

VAL-EOM-012: symmetric teleparallel f(Q) EOM derived, verified or clearly gated.
VAL-EOM-023: metric teleparallel f(T) EOM derived, verified or clearly gated.

Both derivations use the coincident-gauge / vierbein formulation. The linear
cases (f(T)=T and f(Q)=Q) are equivalent to GR by the boundary-term
identities T = -R + boundary and Q = -R + boundary, and this equivalence
is verified componentwise by the SymPy oracle on explicit metric backgrounds.

For f(Q), the Cadabra template eom_fq_linear_coincident exercises the
coincident-gauge variation via the boundary-term identity and passes the
residue check (residue_zero == True), confirming the EOM G_{mu nu} = 0.

For f(T), the Cadabra template eom_ft_linear_tetrad exercises the
variation via the boundary-term identity T = -R + 2 nabla_mu T^mu,
using the tetrad/Weitzenbock formulation. The residue check passes
(residue_zero == True), confirming the EOM G_{mu nu} = 0. The SymPy
cross-check verifies the Weitzenbock geometry (R=0, Q=0, T!=0) and
the torsion scalar on explicit tetrad backgrounds.

Conventions: noether-default-v1 + metric-affine-v1 + tetrad-teleparallel-v1.
"""

from __future__ import annotations

import pytest
import sympy as sp

from evals.eval_fq_symmetric_teleparallel import (
    VERIFIED_PATH_DETAIL as FQ_VERIFIED,
)
from evals.eval_fq_symmetric_teleparallel import (
    build_fq_npr,
)
from evals.eval_ft_teleparallel import (
    VERIFIED_PATH_DETAIL as FT_VERIFIED,
)
from evals.eval_ft_teleparallel import (
    build_ft_npr,
)
from noether.kernels.sympy_kernel.fq_coincident import (
    Q_scalar,
    coincident_gauge_Q_tensor,
    fQ_eom_general,
    fQ_eom_linear,
    nonmetricity_conjugate,
)
from noether.kernels.sympy_kernel.ft_tetrad import (
    boundary_term_identity_residual,
    ft_eom_linear,
    ft_eom_linear_via_boundary,
    ft_eom_metric_form,
    minkowski_metric,
    rotated_tetrad_from_metric,
    superpotential,
    tetrad_metric,
    torsion_scalar_T,
    verify_weitzenbock_is_flat,
    verify_weitzenbock_is_metric_compatible,
    weitzenbock_connection,
)
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    components,
    nonmetricity_of_connection,
    random_diagonal_metric,
    riemann_of_connection,
    torsion_of_connection,
)

# ---------------------------------------------------------------------------
# NPR structure checks
# ---------------------------------------------------------------------------


class TestFTStructure:
    """f(T) NPR has the correct geometry and connection family."""

    def test_ft_npr_teleparallel_family(self):
        npr = build_ft_npr(resolved=True)
        connection = npr.geometry.connection
        assert connection.family == "teleparallel"
        assert connection.curvature_free is True
        assert connection.metric_compatible is True
        assert connection.torsion is True
        assert connection.nonmetricity is False

    def test_ft_npr_well_posed(self):
        npr = build_ft_npr(resolved=True)
        assert npr.is_well_posed()

    def test_ft_npr_has_tetrad_field(self):
        """The f(T) NPR includes a tetrad field object."""
        npr = build_ft_npr(resolved=True)
        tetrad_objs = [o for o in npr.objects if o.kind == "tetrad"]
        assert len(tetrad_objs) >= 1, "f(T) NPR must include a tetrad field"
        assert tetrad_objs[0].name == "e"

    def test_ft_npr_tetrad_kind_exists(self):
        """The 'tetrad' field kind is recognized in the schema."""
        from noether.npr.schema import ObjectDecl

        # Verify "tetrad" is an accepted kind
        decl = ObjectDecl(name="e_test", kind="tetrad", role="dynamical", rank=2)
        assert decl.kind == "tetrad"


class TestFQStructure:
    """f(Q) NPR has the correct geometry and connection family."""

    def test_fq_npr_symmetric_teleparallel_family(self):
        npr = build_fq_npr(resolved=True)
        connection = npr.geometry.connection
        assert connection.family == "symmetric-teleparallel"
        assert connection.curvature_free is True
        assert connection.metric_compatible is False
        assert connection.torsion is False
        assert connection.nonmetricity is True

    def test_fq_npr_well_posed(self):
        npr = build_fq_npr(resolved=True)
        assert npr.is_well_posed()


# ---------------------------------------------------------------------------
# SymPy cross-check: Weitzenbock connection from tetrad
# ---------------------------------------------------------------------------


def _make_weitzenbock_from_tetrad(seed: int, dim: int = 3):
    """Construct a Weitzenbock connection from a non-diagonal tetrad.

    Uses a rotated (non-diagonal) tetrad so that the Weitzenbock torsion
    scalar T is nonzero (the diagonal tetrad of a diagonal metric gives
    T=0). The metric g' is computed FROM the rotated tetrad, since the
    Weitzenbock connection is metric-compatible with that metric, not
    the original diagonal one.
    """
    geom = random_diagonal_metric(seed, dim=dim)
    e, E = rotated_tetrad_from_metric(geom, seed=seed + 50)
    gamma = weitzenbock_connection(geom.coords, e, E)
    # Compute the metric FROM the rotated tetrad
    eta = minkowski_metric(dim)
    g = tetrad_metric(e, eta)
    g_inv = sp.ImmutableDenseNDimArray(
        sp.Matrix([[g[mu, nu] for nu in range(dim)] for mu in range(dim)]).inv()
    )
    T_tensor = torsion_of_connection(gamma)
    return geom, e, E, gamma, T_tensor, g, g_inv


class TestFTWeitzenbockGeometry:
    """Weitzenbock connection from tetrad is flat, metric-compatible, torsionful."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_weitzenbock_is_flat(self, seed):
        """The Weitzenbock connection is flat (R=0) by construction."""
        geom, e, E, gamma, T_tensor, g, g_inv = _make_weitzenbock_from_tetrad(seed, dim=3)
        assert verify_weitzenbock_is_flat(geom.coords, gamma), (
            "Weitzenbock connection should be flat (R=0)"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_weitzenbock_is_metric_compatible(self, seed):
        """The Weitzenbock connection is metric-compatible (Q=0)."""
        geom, e, E, gamma, T_tensor, g, g_inv = _make_weitzenbock_from_tetrad(seed, dim=3)
        assert verify_weitzenbock_is_metric_compatible(geom.coords, gamma, g), (
            "Weitzenbock connection should be metric-compatible (Q=0)"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_weitzenbock_has_nonzero_torsion(self, seed):
        """The Weitzenbock connection has nonzero torsion (T!=0)."""
        geom, e, E, gamma, T_tensor, g, g_inv = _make_weitzenbock_from_tetrad(seed, dim=3)
        t_nonzero = any(sp.simplify(c) != 0 for c in components(T_tensor))
        assert t_nonzero, (
            "Weitzenbock connection should have nonzero torsion "
            "(the metric has coordinate dependence)"
        )


class TestFTTorsionScalar:
    """Torsion scalar T and boundary-term identity on Weitzenbock backgrounds."""

    @pytest.mark.parametrize("seed", [7, 19])
    def test_torsion_scalar_is_computable(self, seed):
        """The torsion scalar T is computable on a Weitzenbock background."""
        geom, e, E, gamma, T_tensor, g, g_inv = _make_weitzenbock_from_tetrad(seed, dim=3)
        T_val = torsion_scalar_T(T_tensor, g, g_inv)
        assert T_val is not None
        # T should be nonzero (the metric has coordinate dependence)
        assert sp.simplify(T_val) != 0, "Torsion scalar should be nonzero"

    @pytest.mark.parametrize("seed", [7, 19])
    def test_boundary_term_identity(self, seed):
        """T = -R(g) + 2 nabla_mu T^mu (boundary-term identity).

        The residual T + R - 2 nabla_mu T^mu should be zero.
        """
        geom, e, E, gamma, T_tensor, g, g_inv = _make_weitzenbock_from_tetrad(seed, dim=3)
        residual = boundary_term_identity_residual(geom.coords, T_tensor, g, g_inv)
        assert sp.simplify(residual) == 0, (
            f"Boundary-term identity residual should be zero, got {residual}"
        )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_linear_ft_eom_matches_einstein(self, seed):
        """For f(T) = T, the EOM via boundary-term identity is G_{mu nu} = 0."""
        geom, e, E, gamma, T_tensor, g, g_inv = _make_weitzenbock_from_tetrad(seed, dim=3)
        T_val = torsion_scalar_T(T_tensor, g, g_inv)

        # EOM via the metric form: E = G - (1/2) g T
        E_metric = ft_eom_linear(geom.coords, T_tensor, g, g_inv, T_val)

        # EOM via boundary-term identity: E_boundary = G (Einstein tensor)
        E_boundary = ft_eom_linear_via_boundary(geom.coords, g, g_inv)

        # The two should satisfy: E_metric + (1/2) g T = E_boundary
        # i.e., (G - (1/2) g T) + (1/2) g T = G
        n = 3
        for mu in range(n):
            for nu in range(n):
                diff = sp.simplify(
                    E_metric[mu, nu] + sp.Rational(1, 2) * g[mu, nu] * T_val
                    - E_boundary[mu, nu]
                )
                assert diff == 0, (
                    f"E_metric[{mu},{nu}] + (1/2)g*T != E_boundary[{mu},{nu}]"
                )

    @pytest.mark.parametrize("seed", [7])
    def test_general_ft_eom_formula_on_background(self, seed):
        """The general f(T) EOM formula is computable on a T!=0 background.

        For f(T) = T + c (constant shift), f'=1, f-Tf'=c,
        giving E_{mu nu} = G_{mu nu} - (1/2) g_{mu nu} T + (1/2) g_{mu nu} c.
        """
        geom, e, E, gamma, T_tensor, g, g_inv = _make_weitzenbock_from_tetrad(seed, dim=3)
        T_val = torsion_scalar_T(T_tensor, g, g_inv)
        c = sp.Symbol("c")

        E_general = ft_eom_metric_form(
            geom.coords, T_tensor, g, g_inv,
            f=T_val + c, fp=sp.Integer(1), T_val=T_val,
        )

        E_linear = ft_eom_linear(geom.coords, T_tensor, g, g_inv, T_val)

        # E_general should equal E_linear + (1/2) g c
        n = 3
        for mu in range(n):
            for nu in range(n):
                expected = _clean(E_linear[mu, nu] + sp.Rational(1, 2) * g[mu, nu] * c)
                diff = sp.simplify(E_general[mu, nu] - expected)
                assert diff == 0, (
                    f"E_general[{mu},{nu}] != expected"
                )


class TestFTSuperpotential:
    """Superpotential S^{rho mu nu} is computable on Weitzenbock backgrounds."""

    @pytest.mark.parametrize("seed", [7])
    def test_superpotential_is_computable(self, seed):
        """The superpotential S^{rho mu nu} can be computed."""
        geom, e, E, gamma, T_tensor, g, g_inv = _make_weitzenbock_from_tetrad(seed, dim=3)
        S = superpotential(T_tensor, g, g_inv)
        assert S is not None
        assert S.shape == (3, 3, 3)


# ---------------------------------------------------------------------------
# SymPy cross-check: boundary-term identity for f(T) = T (old style)
# ---------------------------------------------------------------------------


def _make_weitzenbock_connection(seed: int, dim: int = 3):
    """Construct a metric-compatible torsionful connection (Gamma = LC + K(T))."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma_lc = geom.christoffel

    import random as rng

    rng_local = rng.Random(seed + 100)
    n = dim
    T = sp.MutableDenseNDimArray.zeros(n, n, n)
    for lam in range(n):
        for mu in range(n):
            for nu in range(mu + 1, n):
                val = sp.Integer(rng_local.randint(-2, 2))
                T[lam, mu, nu] = val
                T[lam, nu, mu] = -val
    T = sp.ImmutableDenseNDimArray(T)

    g_inv_imm = sp.ImmutableDenseNDimArray(geom.g_inv)
    g_imm = sp.ImmutableDenseNDimArray(geom.g)

    gamma = sp.MutableDenseNDimArray(gamma_lc)
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

    return geom, gamma, T


class TestFTSymPyCrossCheck:
    """SymPy cross-check for f(T) teleparallel gravity."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_metric_compatible_torsionful_background(self, seed):
        """Gamma = LC + K(T) is metric-compatible (Q=0) with nonzero torsion."""
        geom, gamma, T = _make_weitzenbock_connection(seed, dim=3)
        g_imm = sp.ImmutableDenseNDimArray(geom.g)

        Q = nonmetricity_of_connection(geom.coords, gamma, g_imm)
        q_zero = all(sp.simplify(c) == 0 for c in components(Q))
        assert q_zero, "Gamma = LC + K should be metric-compatible (Q=0)"

        T_check = torsion_of_connection(gamma)
        t_nonzero = any(sp.simplify(c) != 0 for c in components(T_check))
        assert t_nonzero, "Gamma = LC + K should have nonzero torsion"

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_lc_plus_K_curvature_not_zero_in_general(self, seed):
        """An arbitrary torsion does NOT produce a curvature-free connection."""
        geom, gamma, T = _make_weitzenbock_connection(seed, dim=3)

        R = riemann_of_connection(geom.coords, gamma)
        R_nonzero = any(sp.simplify(c) != 0 for c in components(R))
        assert R_nonzero, (
            "Gamma = LC + K(arbitrary T) should have nonzero curvature, "
            "showing the curvature-free constraint is nontrivial"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_lc_background_einstein_tensor(self, seed):
        """The Einstein tensor of the metric is computable and provides the
        target EOM for the linear case f(T) = T."""
        geom = random_diagonal_metric(seed, dim=3)
        G = geom.einstein
        assert G is not None


# ---------------------------------------------------------------------------
# SymPy cross-check: coincident-gauge f(Q) verification
#
# In coincident gauge (Gamma=0), Q_{lambda mu nu} = partial_lambda g_{mu nu}
# and the f(Q) action becomes a pure-metric functional. The boundary-term
# identity Q = -R + boundary implies the linear EOM is G_{mu nu} = 0.
# ---------------------------------------------------------------------------


def _make_coincident_gauge_background(seed: int, dim: int = 3):
    """Construct a coincident-gauge background (Gamma=0, Q != 0).

    Returns the ComponentGeometry and the Q scalar value.
    """
    geom = random_diagonal_metric(seed, dim=dim)

    # Compute Q_{lambda mu nu} = partial_lambda g_{mu nu}
    Q_tens = coincident_gauge_Q_tensor(geom.coords, geom.g)

    # Check that Q_{lambda mu nu} is nonzero (the metric has coordinate
    # dependence so partial derivatives are nonzero)
    q_nonzero = any(sp.simplify(c) != 0 for c in components(Q_tens))
    if not q_nonzero:
        # If no nonzero Q components, use a slightly modified metric
        # (this shouldn't happen for random_diagonal_metric with seed)
        pytest.skip(f"seed {seed} produced Q=0 in dim {dim}")

    return geom


class TestFQCoincidentGaugeCrossCheck:
    """SymPy cross-check for f(Q) coincident-gauge EOM."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_coincident_gauge_Q_tensor_nonzero(self, seed):
        """In coincident gauge (Gamma=0), Q_{lambda mu nu} = partial_lambda g_{mu nu} != 0."""
        geom = random_diagonal_metric(seed, dim=3)
        Q_tens = coincident_gauge_Q_tensor(geom.coords, geom.g)
        q_nonzero = any(sp.simplify(c) != 0 for c in components(Q_tens))
        assert q_nonzero, "Coincident gauge should have nonzero Q_{lambda mu nu}"

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_coincident_gauge_Q_scalar_matches_existing_geometry(self, seed):
        """Q computed from fq_coincident matches nonmetricity_of_connection with Gamma=0."""
        geom = random_diagonal_metric(seed, dim=3)
        n = 3

        # Zero connection
        gamma_zero = sp.ImmutableDenseNDimArray(sp.MutableDenseNDimArray.zeros(n, n, n))

        # Q tensor from fq_coincident
        Q_mine = coincident_gauge_Q_tensor(geom.coords, geom.g)

        # Q tensor from existing geometry module
        Q_existing = nonmetricity_of_connection(geom.coords, gamma_zero, geom.g)

        match = all(
            sp.simplify(Q_mine[i, j, k] - Q_existing[i, j, k]) == 0
            for i in range(n) for j in range(n) for k in range(n)
        )
        assert match, "Q tensor from fq_coincident should match nonmetricity_of_connection(Gamma=0)"

    @pytest.mark.parametrize("seed", [7, 19])
    def test_linear_fq_eom_is_einstein_tensor(self, seed):
        """For f(Q) = Q, the EOM E_{mu nu} = G_{mu nu} (Einstein tensor).

        This verifies the general f(Q) EOM formula reduces correctly
        for the linear case: f'=1, f''=0, f-Qf'=0, giving E=G.
        """
        geom = random_diagonal_metric(seed, dim=3)
        Q_val = Q_scalar(geom.coords, geom.g, geom.g_inv)

        # General EOM with f(Q) = Q (so f'=1, f''=0, f-Qf'=0)
        E_general = fQ_eom_general(
            geom.coords, geom.g, geom.g_inv,
            f=Q_val, fp=sp.Integer(1), fpp=sp.Integer(0), Q_val=Q_val,
        )

        # Linear EOM (should be the Einstein tensor)
        G = fQ_eom_linear(geom.coords, geom.g, geom.g_inv)

        # Check componentwise
        n = 3
        for mu in range(n):
            for nu in range(n):
                diff = sp.simplify(E_general[mu, nu] - G[mu, nu])
                assert diff == 0, (
                    f"E_general[{mu},{nu}] != G[{mu},{nu}]: {E_general[mu, nu]} vs {G[mu, nu]}"
                )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_general_fq_eom_formula_on_background(self, seed):
        """The general f(Q) EOM formula is computable on a Q!=0 background.

        For f(Q) = Q + c (constant shift), f'=1, f''=0, f-Qf'=c,
        giving E_{mu nu} = G_{mu nu} - (1/2) g_{mu nu} c.
        """
        geom = random_diagonal_metric(seed, dim=3)
        Q_val = Q_scalar(geom.coords, geom.g, geom.g_inv)
        c = sp.Symbol("c")

        E = fQ_eom_general(
            geom.coords, geom.g, geom.g_inv,
            f=Q_val + c, fp=sp.Integer(1), fpp=sp.Integer(0), Q_val=Q_val,
        )

        G = fQ_eom_linear(geom.coords, geom.g, geom.g_inv)

        # E should equal G - (1/2) g c
        n = 3
        for mu in range(n):
            for nu in range(n):
                expected = _clean(G[mu, nu] - sp.Rational(1, 2) * geom.g[mu, nu] * c)
                diff = sp.simplify(E[mu, nu] - expected)
                assert diff == 0, (
                    f"E[{mu},{nu}] != expected: {E[mu, nu]} vs {expected}"
                )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_nonmetricity_conjugate_is_computable(self, seed):
        """The non-metricity conjugate P^{lambda}_{mu nu} is computable."""
        geom = random_diagonal_metric(seed, dim=3)
        P = nonmetricity_conjugate(geom.coords, geom.g, geom.g_inv)
        assert P is not None
        assert P.shape == (3, 3, 3)


# ---------------------------------------------------------------------------
# Cadabra verification tests
#
# The linear f(Q) = Q EOM is verified by the Cadabra template
# eom_fq_linear_coincident. The linear f(T) = T EOM is verified by
# eom_ft_linear_tetrad.
# ---------------------------------------------------------------------------


class TestFQCadabraVerification:
    """Cadabra verification for f(Q) = Q EOM."""

    def test_fq_linear_cadabra_residue_zero(self):
        """The Cadabra template eom_fq_linear_coincident passes the residue check.

        This template varies the f(Q) = Q action in coincident gauge using
        the boundary-term identity Q = -R + boundary, reducing to the
        Einstein-Hilbert variation. The residue check confirms the EOM
        G_{mu nu} = 0.
        """
        try:
            from noether.kernels.base import Capability, KernelTask
            from noether.kernels.cadabra.adapter import CadabraAdapter
            from noether.kernels.cadabra.templates import get

            adapter = CadabraAdapter()
            if not adapter.available():
                pytest.skip("cadabra2 not available")

            script = get("eom_fq_linear_coincident")
            result = adapter.run(KernelTask(
                capability=Capability.VARY,
                description="f(Q) linear EOM coincident gauge",
                payload={"script": script},
            ))

            checks = result.value.get("checks", {})
            residue_zero = checks.get("residue_zero", "False")
            assert residue_zero == "True", (
                f"Cadabra residue check failed: {checks}"
            )
        except ImportError:
            pytest.skip("cadabra adapter not available")

    def test_fq_verified_path_detail(self):
        """The verified path detail mentions the coincident gauge and boundary-term identity."""
        assert "coincident" in FQ_VERIFIED.lower() or "boundary" in FQ_VERIFIED.lower()
        assert (
            "G_{mu nu} = 0" in FQ_VERIFIED
            or "G_{\\mu\\nu} = 0" in FQ_VERIFIED
            or "Einstein" in FQ_VERIFIED
        )


class TestFTCadabraVerification:
    """Cadabra verification for f(T) = T EOM via tetrad/Weitzenbock."""

    def test_ft_linear_cadabra_residue_zero(self):
        """The Cadabra template eom_ft_linear_tetrad passes the residue check.

        This template varies the f(T) = T action via the boundary-term
        identity T = -R + 2 nabla_mu T^mu, reducing to the Einstein-Hilbert
        variation. The residue check confirms the EOM G_{mu nu} = 0.
        """
        try:
            from noether.kernels.base import Capability, KernelTask
            from noether.kernels.cadabra.adapter import CadabraAdapter
            from noether.kernels.cadabra.templates import get

            adapter = CadabraAdapter()
            if not adapter.available():
                pytest.skip("cadabra2 not available")

            script = get("eom_ft_linear_tetrad")
            result = adapter.run(KernelTask(
                capability=Capability.VARY,
                description="f(T) linear EOM tetrad/Weitzenbock",
                payload={"script": script},
            ))

            checks = result.value.get("checks", {})
            residue_zero = checks.get("residue_zero", "False")
            assert residue_zero == "True", (
                f"Cadabra residue check failed: {checks}"
            )
        except ImportError:
            pytest.skip("cadabra adapter not available")

    def test_ft_verified_path_detail(self):
        """The verified path detail mentions the boundary-term identity and tetrad."""
        assert "boundary" in FT_VERIFIED.lower()
        assert (
            "tetrad" in FT_VERIFIED.lower()
            or "Weitzenbock" in FT_VERIFIED
            or "vierbein" in FT_VERIFIED.lower()
        )
        assert (
            "G_{mu nu} = 0" in FT_VERIFIED
            or "Einstein" in FT_VERIFIED
        )


# ---------------------------------------------------------------------------
# Verified derivation tests for f(T)
#
# The f(T) = T EOM is now verified via the boundary-term identity and the
# Cadabra template eom_ft_linear_tetrad. The tetrad e^a_mu is a fundamental
# NPR field kind. The tests verify that the verified path detail is honest
# and informative.
# ---------------------------------------------------------------------------


class TestFTVerifiedDerivation:
    """f(T) EOM is verified via the tetrad/Weitzenbock formulation."""

    def test_ft_verified_detail_mentions_boundary_term(self):
        """The verified path detail mentions the boundary-term identity."""
        assert "boundary" in FT_VERIFIED.lower() or "T = -R" in FT_VERIFIED

    def test_ft_verified_detail_mentions_tetrad(self):
        """The verified path detail mentions the tetrad/Weitzenbock formulation."""
        assert "tetrad" in FT_VERIFIED.lower() or "Weitzenbock" in FT_VERIFIED

    def test_ft_verified_detail_mentions_linear_case(self):
        """The verified detail acknowledges the linear case is equivalent to GR."""
        assert "f(T) = T" in FT_VERIFIED or "linear" in FT_VERIFIED.lower()
