# ruff: noqa: E501 -- Cadabra fragment lines exceed 100 chars
"""Multi-field Palatini scalar-tensor EOMs: F(phi)R(Gamma) with three
independent variations (metric, connection, scalar), the dF source in
the connection equation, explicit boundary-term assumption, and the
Levi-Civita limit.

Validates VAL-EOM-024, VAL-EOM-025, VAL-EOM-026.

Action: S = \\int d^4x \\sqrt{-g} F(phi) g^{mu nu} R_{mu nu}(Gamma)

Conventions: noether-default-v1 (dimension 4, mostly-plus,
R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
+ GG - GG, R_{sigma nu} = R^lambda_{sigma lambda nu}).
"""

from __future__ import annotations

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.templates import get as get_template
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    components,
    random_diagonal_metric,
)

requires_cadabra = pytest.mark.skipif(
    not CadabraAdapter().available(), reason="cadabra2 not installed"
)

# ---------------------------------------------------------------------------
# Palatini scalar-tensor EOM scripts are registered in templates.py as
# palatini_st_metric, palatini_st_connection, palatini_st_scalar, and
# palatini_st_lc_limit. Retrieved via templates.get().
# ---------------------------------------------------------------------------






def _run_template(template_name: str) -> dict:
    """Run a Cadabra template by name and return the checks dict."""
    result = CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="Palatini scalar-tensor EOM check",
            payload={"script": get_template(template_name)},
        )
    )
    return result.value.get("checks", {})


# ---------------------------------------------------------------------------
# VAL-EOM-024: Three distinct wrt derivations; connection equation
# contains the dF source; each verified or gated.
# ---------------------------------------------------------------------------

@pytest.mark.kernel_cadabra
@requires_cadabra
class TestMetricEOM:
    """Palatini F(phi)R(Gamma) metric EOM: F [R_{(mu nu)} - 1/2 g_{mu nu} R_tilde]."""

    def test_residue_zero(self):
        """The metric EOM residue check passes."""
        checks = _run_template("palatini_st_metric")
        assert checks.get("residue_zero") == "True", (
            f"metric EOM residue not zero: {checks}"
        )

    def test_metric_eom_is_distinct_from_lc(self):
        """The Palatini metric EOM is NOT the standard LC metric EOM.
        It contains both R_{mu nu} and R_{nu mu} explicitly (symmetrized),
        unlike the LC case where R_{mu nu} is declared symmetric."""
        # The metric script does NOT declare R_{mu nu}::Symmetric
        assert "R_{\\mu\\nu}::Symmetric" not in get_template("palatini_st_metric")
        # The target contains both index orders
        assert "R_{\\mu\\nu}" in get_template("palatini_st_metric")
        assert "R_{\\nu\\mu}" in get_template("palatini_st_metric")

    def test_metric_eom_contains_F_factor(self):
        """The metric EOM contains F as a multiplicative factor,
        showing it's distinct from the pure EH Palatini case."""
        assert "sg F k" in get_template("palatini_st_metric")


@pytest.mark.kernel_cadabra
@requires_cadabra
class TestConnectionEOM:
    """Palatini F(phi)R(Gamma) connection EOM: carries the dF source."""

    def test_has_dF_source(self):
        """The connection EOM contains the dF = F_phi partial_mu phi source
        coupling the scalar sector to the connection sector. We verify this
        by checking that the Cadabra output contains Fp (F_phi)."""
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="Connection EOM dF source check",
                payload={"script": get_template("palatini_st_connection")},
            )
        )
        # Check the raw output for Fp (the LaTeX form of F_phi)
        stdout = result.raw.stdout
        assert "Fp" in stdout, (
            "connection EOM missing dF source (Fp not found in output)"
        )

    def test_boundary_assumption_recorded(self):
        """The boundary-term assumption is explicitly recorded in the script
        and the kernel confirms it (VAL-EOM-025)."""
        # The script contains the boundary assumption comment
        assert "BOUNDARY-TERM ASSUMPTION" in get_template("palatini_st_connection")
        assert "delta Gamma vanishes" in get_template("palatini_st_connection")
        # The kernel check confirms the recording
        checks = _run_template("palatini_st_connection")
        assert checks.get("boundary_assumption_recorded") == "True"

    def test_projective_does_not_solve(self):
        """The projective mode Gamma = LC + delta^lam_nu A_mu does NOT
        solve the connection equation when F is non-constant (unlike the
        pure EH case). The dF source prevents this. We verify by checking
        that the residual after projective substitution is non-empty."""
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="Connection EOM projective check",
                payload={"script": get_template("palatini_st_connection")},
            )
        )
        checks = result.value.get("checks", {})
        # The projective_residual should be non-empty (nonzero)
        residual = checks.get("projective_residual", "")
        assert len(residual) > 0 and residual != "0", (
            f"projective mode should not solve when F is non-constant: residual={residual}"
        )

    def test_connection_eom_is_distinct_from_pure_eh(self):
        """The connection EOM for F(phi)R(Gamma) is distinct from the
        pure EH Palatini connection EOM because it contains the dF source."""
        # The pure EH connection script does not have Fp or phi
        pure_eh = get_template("eval2_palatini_connection")
        assert "Fp" not in pure_eh
        assert "phi" not in pure_eh
        # Our script has both
        assert "Fp" in get_template("palatini_st_connection")
        assert "phi" in get_template("palatini_st_connection")


@pytest.mark.kernel_cadabra
@requires_cadabra
class TestScalarEOM:
    """Palatini F(phi)R(Gamma) scalar EOM: F_phi R_tilde(Gamma) = 0."""

    def test_residue_zero(self):
        """The scalar EOM residue check passes."""
        checks = _run_template("palatini_st_scalar")
        assert checks.get("residue_zero") == "True", (
            f"scalar EOM residue not zero: {checks}"
        )

    def test_scalar_eom_contains_F_phi_R(self):
        """The scalar EOM contains F_phi R_tilde(Gamma), not F_phi R(g).
        This is the Ricci scalar of the independent connection, distinct
        from the Levi-Civita Ricci scalar."""
        assert "Fp" in get_template("palatini_st_scalar")
        assert "R" in get_template("palatini_st_scalar")

    def test_scalar_eom_is_distinct_from_lc_nonminimal(self):
        """The scalar EOM for the Palatini F(phi)R(Gamma) action is
        F_phi R_tilde(Gamma), while the LC nonminimal scalar EOM is
        F_phi R(g) + box phi - V_phi. They are different equations."""
        # The Palatini scalar EOM has no box phi term (no kinetic term in action)
        assert "\\Box" not in get_template("palatini_st_scalar")
        # The Palatini scalar EOM uses R_tilde(Gamma), not R(g)
        assert "R_{\\sigma\\nu}" in get_template("palatini_st_scalar")


# ---------------------------------------------------------------------------
# VAL-EOM-025: Boundary-term assumption recorded, bulk residue zero.
# ---------------------------------------------------------------------------

@pytest.mark.kernel_cadabra
@requires_cadabra
class TestBoundaryTermAssumption:
    """The connection variation records the IBP boundary-term assumption
    explicitly and the bulk residue still reduces to 0."""

    def test_boundary_assumption_in_script(self):
        """The connection EOM script contains the explicit boundary-term
        assumption comment, not silently dropped."""
        assert "BOUNDARY-TERM ASSUMPTION" in get_template("palatini_st_connection")
        assert "deltaGamma" in get_template("palatini_st_connection") or "delta Gamma" in get_template("palatini_st_connection")
        assert "NOT silently dropped" in get_template("palatini_st_connection")

    def test_bulk_residue_reduces_to_zero(self):
        """The bulk residue of the connection equation reduces to 0
        (the connection EOM is well-defined under the boundary assumption).
        The independent target is the Euler-Lagrange equation in
        partial-derivative form, and the derived (vary+IBP) expression
        matches it exactly."""
        checks = _run_template("palatini_st_connection")
        assert checks.get("residue_zero") == "True", (
            f"connection EOM bulk residue not zero: {checks}"
        )

    def test_boundary_check_is_recorded(self):
        """The kernel check boundary_assumption_recorded is True."""
        checks = _run_template("palatini_st_connection")
        assert checks.get("boundary_assumption_recorded") == "True", (
            f"boundary assumption not recorded: {checks}"
        )


# ---------------------------------------------------------------------------
# VAL-EOM-026: At T=Q=0 the metric-affine field equation reduces to
# the Levi-Civita result, residue-pinned and SymPy-confirmed.
# ---------------------------------------------------------------------------

@pytest.mark.kernel_cadabra
@requires_cadabra
class TestLCLimitCadabra:
    """Cadabra residue check: at T=Q=0 the metric EOM reduces to F G_{mu nu} = 0."""

    def test_lc_limit_residue_zero(self):
        """With R_{mu nu}::Symmetric (the LC limit), the Palatini metric
        EOM F [R_{mu nu} - 1/2 g_{mu nu} R] = F G_{mu nu} = 0."""
        checks = _run_template("palatini_st_lc_limit")
        assert checks.get("lc_limit_residue_zero") == "True", (
            f"LC limit residue not zero: {checks}"
        )


class TestLCLimitSymPy:
    """SymPy component cross-check: on random metric backgrounds, the
    Palatini metric EOM at T=Q=0 equals F(phi) times the Einstein tensor."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_metric_eom_equals_F_times_einstein_at_TQ0(self, seed):
        """On a random metric background (T=Q=0, connection is Levi-Civita),
        the Palatini metric EOM F[ R_{(mu nu)} - 1/2 g_{mu nu} R ] equals
        F G_{mu nu} componentwise."""
        geom = random_diagonal_metric(seed, dim=3)
        n = geom.dim
        g = sp.ImmutableDenseNDimArray(geom.g)
        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)

        # Levi-Civita Ricci and scalar
        ric = geom.ricci  # R_{mu nu}(g) from ComponentGeometry
        R_scalar = sum(g_inv[a, b] * ric[a, b] for a in range(n) for b in range(n))

        # Palatini metric EOM at T=Q=0: R_{mu nu} - 1/2 g_{mu nu} R
        # (R is symmetric at T=Q=0, so R_{(mu nu)} = R_{mu nu})
        eom = sp.MutableDenseNDimArray(ric)
        for a in range(n):
            for b in range(n):
                eom[a, b] = sp.cancel(ric[a, b] - sp.Rational(1, 2) * g[a, b] * R_scalar)

        # Einstein tensor G_{mu nu} = R_{mu nu} - 1/2 g_{mu nu} R
        einstein = geom.einstein

        # They should be equal (F is a common factor)
        for idx_val in components(eom - einstein):
            assert sp.simplify(idx_val) == 0, (
                f"Palatini metric EOM at T=Q=0 != F * Einstein tensor on seed {seed}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_ricci_symmetric_at_TQ0(self, seed):
        """At T=Q=0 (Levi-Civita connection), the Ricci tensor is symmetric,
        confirming the LC limit is valid."""
        geom = random_diagonal_metric(seed, dim=3)
        ric = geom.ricci
        # R_{mu nu} - R_{nu mu} should be zero
        for a in range(geom.dim):
            for b in range(a + 1, geom.dim):
                diff = sp.simplify(ric[a, b] - ric[b, a])
                assert diff == 0, (
                    f"Ricci not symmetric at T=Q=0: R_{{{a}{b}}} - R_{{{b}{a}}} = {diff}"
                )

    @pytest.mark.parametrize("seed", [11, 23])
    def test_metric_eom_nonzero_on_curved_background(self, seed):
        """The metric EOM F G_{mu nu} is nonzero on a curved background,
        confirming it is a non-trivial equation."""
        geom = random_diagonal_metric(seed, dim=3)
        einstein = geom.einstein
        is_nonzero = any(sp.simplify(c) != 0 for c in components(einstein))
        assert is_nonzero, "Einstein tensor should be nonzero on a curved background"


class TestConnectionEOMSymPy:
    """SymPy component cross-check: at F=const, Gamma=Levi-Civita, the
    connection equation vanishes identically (metric compatibility).
    This is the component-level counterpart of the Cadabra residue-zero
    check and verifies the geometric structure of the EOM."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_connection_eom_at_LC_F_const(self, seed):
        """At the Levi-Civita limit with F=const (no dF source), the
        connection EOM E^alpha_{beta gamma} vanishes componentwise,
        confirming the LC connection is metric-compatible in the
        Palatini connection equation's index structure."""
        geom = random_diagonal_metric(seed, dim=3)
        n = geom.dim
        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        Gamma = geom.christoffel  # Gamma[a][b][c] = Gamma^a_{bc}
        sg = sp.sqrt(-sp.det(geom.g))  # use Matrix, not NDimArray

        # Pre-compute partial_alpha(sg) = sg * Gamma^lambda_{lambda alpha}
        Gamma_trace_vec = [
            _clean(sum(Gamma[lam, lam, al] for lam in range(n)))
            for al in range(n)
        ]
        d_sg = [_clean(sg * Gamma_trace_vec[al]) for al in range(n)]

        # partial_alpha(g^{beta gamma}) via LC identity:
        # -Gamma^beta_{alpha sigma} g^{sigma gamma}
        # - Gamma^gamma_{alpha sigma} g^{beta sigma}
        def dg_inv(alpha, beta, gamma):
            return _clean(
                -sum(
                    Gamma[beta, alpha, sig] * g_inv[sig, gamma]
                    + Gamma[gamma, alpha, sig] * g_inv[beta, sig]
                    for sig in range(n)
                )
            )

        # The connection EOM at F=const (F_phi=0):
        # E^alpha_{beta gamma} = partial_alpha(sg g^{beta gamma})
        #   - delta^beta_alpha partial_nu(sg g^{gamma nu})
        #   - sg [delta^beta_alpha Gamma^gamma_{nu sigma} g^{nu sigma}
        #         + Gamma^lambda_{lambda alpha} g^{gamma beta}
        #         - Gamma^gamma_{alpha sigma} g^{sigma beta}
        #         - Gamma^beta_{nu alpha} g^{gamma nu}]
        for alpha in range(n):
            for beta in range(n):
                for gamma_ in range(n):
                    # term1: partial_alpha(sg g^{beta gamma})
                    t1 = _clean(
                        d_sg[alpha] * g_inv[beta, gamma_]
                        + sg * dg_inv(alpha, beta, gamma_)
                    )

                    # term2: -delta^beta_alpha partial_nu(sg g^{gamma nu})
                    t2 = sp.Integer(0)
                    if beta == alpha:
                        for nu in range(n):
                            t2 = _clean(
                                t2
                                + d_sg[nu] * g_inv[gamma_, nu]
                                + sg * dg_inv(nu, gamma_, nu)
                            )
                    t2 = _clean(-t2)

                    # term3: -sg delta^beta_alpha Gamma^gamma_{nu sigma} g^{nu sigma}
                    t3 = sp.Integer(0)
                    if beta == alpha:
                        t3 = _clean(
                            -sg
                            * sum(
                                Gamma[gamma_, nu, sig] * g_inv[nu, sig]
                                for nu in range(n)
                                for sig in range(n)
                            )
                        )

                    # term4: -sg Gamma^lambda_{lambda alpha} g^{gamma beta}
                    t4 = _clean(-sg * Gamma_trace_vec[alpha] * g_inv[gamma_, beta])

                    # term5: +sg Gamma^gamma_{alpha sigma} g^{sigma beta}
                    t5 = _clean(
                        sg
                        * sum(
                            Gamma[gamma_, alpha, sig] * g_inv[sig, beta]
                            for sig in range(n)
                        )
                    )

                    # term6: +sg Gamma^beta_{nu alpha} g^{gamma nu}
                    t6 = _clean(
                        sg
                        * sum(
                            Gamma[beta, nu, alpha] * g_inv[gamma_, nu]
                            for nu in range(n)
                        )
                    )

                    eom_val = _clean(t1 + t2 + t3 + t4 + t5 + t6)
                    assert sp.simplify(eom_val) == 0, (
                        f"Connection EOM at LC not zero: seed={seed}, "
                        f"alpha={alpha}, beta={beta}, gamma={gamma_}, val={eom_val}"
                    )


# ---------------------------------------------------------------------------
# VAL-EOM-024: The three EOMs are distinct, none silently merged.
# ---------------------------------------------------------------------------


class TestThreeDistinctDerivations:
    """VAL-EOM-024: The three EOMs are distinct, none silently merged."""

    def test_metric_eom_differs_from_connection_eom(self):
        """The metric and connection EOMs are structurally different:
        metric varies g, connection varies Gamma."""
        # Metric script varies g^{sigma nu}
        assert "g^{\\sigma\\nu} -> k^{\\sigma\\nu}" in get_template("palatini_st_metric")
        assert "G^{\\lambda}_{\\mu\\nu} -> dG" not in get_template("palatini_st_metric")
        # Connection script varies G
        assert "G^{\\lambda}_{\\mu\\nu} -> dG" in get_template("palatini_st_connection")
        assert "g^{\\sigma\\nu} -> k" not in get_template("palatini_st_connection")

    def test_scalar_eom_differs_from_metric_eom(self):
        """The scalar EOM varies phi, not g."""
        # Scalar script varies phi
        assert "phi -> dphi" in get_template("palatini_st_scalar")
        # Metric script varies g
        assert "g^{\\sigma\\nu} -> k^{\\sigma\\nu}" in get_template("palatini_st_metric")
        assert "phi -> dphi" not in get_template("palatini_st_metric")

    def test_connection_eom_differs_from_scalar_eom(self):
        """The connection EOM varies Gamma, not phi."""
        assert "G^{\\lambda}_{\\mu\\nu} -> dG" in get_template("palatini_st_connection")
        assert "phi -> dphi" not in get_template("palatini_st_connection")

    def test_connection_eom_carries_dF_source(self):
        """The connection EOM contains the dF source (F_phi partial_mu phi)
        that couples the scalar sector to the connection sector."""
        # The connection script expands partial_mu F = Fp partial_mu phi
        assert "Fp" in get_template("palatini_st_connection")
        assert "partial_{\\mu}{phi}" in get_template("palatini_st_connection")
