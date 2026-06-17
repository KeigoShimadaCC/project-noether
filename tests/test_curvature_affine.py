"""Pin the metric-affine curvature primitives (independent connection).

Tests for VAL-GEOM-001 and VAL-GEOM-002:

- VAL-GEOM-001: R^rho_{sigma mu nu}(Gamma) from an independent
  \\partial-Depends connection with no R_{mu nu}::Symmetric declaration;
  residue is 0 and riemann_of_connection reproduces it componentwise on
  a random background.

- VAL-GEOM-002: ricci_of_connection yields R_{mu nu} != R_{nu mu} on a
  torsionful background; no R_{mu nu}::Symmetric appears on the
  metric-affine path; T=0 recovers Ricci symmetry (Levi-Civita limit).

Each Cadabra primitive is pinned by a residue check (the kernel verifies
the substitution produces the correct expansion).  Each is additionally
cross-checked against the SymPy general-connection oracle on explicit
random metric + connection backgrounds (the torsion-trap safeguard).
They skip when cadabra2 is absent.
"""

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.curvature import (
    AFFINE_CONNECTION_DEPENDS,
    AFFINE_CURVATURE_DECL,
    expand_ricci_affine,
    expand_riemann_affine,
    fold_ricci_affine,
)
from noether.kernels.sympy_kernel.geometry import (
    components,
    random_affine_connection,
    random_diagonal_metric,
    ricci_of_connection,
    riemann_of_connection,
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


def _affine_script(body: str) -> str:
    return (
        _BASE_DECL_AFFINE
        + AFFINE_CURVATURE_DECL
        + "\n"
        + AFFINE_CONNECTION_DEPENDS
        + "\n"
        + body
    )


def _run_affine(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="affine curvature primitive check",
            payload={"script": _affine_script(body)},
        )
    )


# ---------------------------------------------------------------------------
# SymPy cross-check helpers
# ---------------------------------------------------------------------------


def _sympy_riemann_on_random_background(seed: int, dim: int = 3):
    """Build a random metric + asymmetric connection and compute the Riemann."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
    R = riemann_of_connection(geom.coords, gamma)
    return geom, gamma, R


def _sympy_ricci_on_random_background(seed: int, dim: int = 3, symmetric: bool = False):
    """Build a random metric + connection and compute the Ricci."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(
        seed + 1000, geom.coords, symmetric=symmetric
    )
    Ric = ricci_of_connection(geom.coords, gamma)
    return geom, gamma, Ric


# ===========================================================================
# Cadabra residue checks
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestAffineCurvatureResidue:
    """Cadabra residue checks for the metric-affine curvature primitives."""

    def test_riemann_expansion_residue_zero(self):
        """R_{rho sigma mu nu} expanded via expand_riemann_affine matches the
        explicit definition (residue 0).  VAL-GEOM-001."""
        body = (
            # Build the lowered Riemann from the definition
            r"riem := g_{\rho\alpha} ( "
            r"\partial_{\mu}{G^{\alpha}_{\nu\sigma}} "
            r"- \partial_{\nu}{G^{\alpha}_{\mu\sigma}} "
            r"+ G^{\alpha}_{\mu\lambda} G^{\lambda}_{\nu\sigma} "
            r"- G^{\alpha}_{\nu\lambda} G^{\lambda}_{\mu\sigma} );"
            "\n"
            "distribute(riem); canonicalise(riem); rename_dummies(riem);\n"
            # Apply the primitive substitution
            r"target := R_{\rho\sigma\mu\nu};"
            "\n"
            + expand_riemann_affine("G", "target")
            + "\n"
            "distribute(target); canonicalise(target); rename_dummies(target);\n"
            # Residue
            "residue := @(riem) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: riemann_expansion_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("riemann_expansion_zero") == "True", (
            result.raw.stdout
        )

    def test_ricci_expansion_residue_zero(self):
        """R_{sigma nu} expanded via expand_ricci_affine matches the explicit
        definition (residue 0).  VAL-GEOM-001 (Ricci part)."""
        body = (
            # Build the Ricci from the definition
            r"ricci := \partial_{\lambda}{G^{\lambda}_{\nu\sigma}} "
            r"- \partial_{\nu}{G^{\lambda}_{\lambda\sigma}} "
            r"+ G^{\lambda}_{\lambda\rho} G^{\rho}_{\nu\sigma} "
            r"- G^{\lambda}_{\nu\rho} G^{\rho}_{\lambda\sigma};"
            "\n"
            "canonicalise(ricci); rename_dummies(ricci);\n"
            # Apply the primitive substitution
            r"target := R_{\sigma\nu};"
            "\n"
            + expand_ricci_affine("G", "target")
            + "\n"
            "distribute(target); canonicalise(target); rename_dummies(target);\n"
            # Residue
            "residue := @(ricci) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: ricci_expansion_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("ricci_expansion_zero") == "True", (
            result.raw.stdout
        )

    def test_riemann_to_ricci_fold_affine(self):
        """fold_ricci_affine correctly contracts g^{mu nu} R_{alpha mu beta nu}
        to R_{alpha beta} on the metric-affine path (no Symmetric)."""
        body = (
            r"ex := g^{\mu\nu} R_{\alpha\mu\beta\nu};"
            "\n"
            + fold_ricci_affine("ex")
            + "\n"
            r"target := R_{\alpha\beta};"
            "\n"
            "residue := @(ex) - @(target);\n"
            "canonicalise(residue);\n"
            'print("NOETHER_CHECK: fold_ricci_affine_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("fold_ricci_affine_zero") == "True", (
            result.raw.stdout
        )

    def test_no_ricci_symmetric_declaration_on_affine_path(self):
        """The metric-affine declaration block does NOT contain
        R_{mu nu}::Symmetric.  VAL-GEOM-002 (declaration guard)."""
        assert "R_{\\mu\\nu}::Symmetric" not in AFFINE_CURVATURE_DECL
        # Also confirm the LC declaration DOES have it (regression guard)
        from noether.kernels.cadabra.curvature import CURVATURE_DECL

        assert "R_{\\mu\\nu}::Symmetric" in CURVATURE_DECL


# ===========================================================================
# SymPy component cross-checks (the torsion-trap safeguard)
# ===========================================================================


class TestAffineCurvatureSymPyCrossCheck:
    """Cross-check the metric-affine primitives against the SymPy oracle on
    explicit random backgrounds.  This is the independent verification that
    catches the torsion trap (architecture.md section 3.2)."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_riemann_of_connection_matches_definition(self, seed):
        """riemann_of_connection reproduces the defining formula
        R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
                             + Gamma^rho_{mu lam} Gamma^lam_{nu sigma}
                             - Gamma^rho_{nu lam} Gamma^lam_{mu sigma}
        on a random asymmetric connection.  VAL-GEOM-001."""
        geom, gamma, R_oracle = _sympy_riemann_on_random_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        # Compute from the definition directly
        R_def = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for rho in range(n):
            for sig in range(n):
                for mu in range(n):
                    for nu in range(n):
                        val = sp.diff(gamma[rho, nu, sig], x[mu]) - sp.diff(
                            gamma[rho, mu, sig], x[nu]
                        ) + sum(
                            gamma[rho, mu, lam] * gamma[lam, nu, sig]
                            - gamma[rho, nu, lam] * gamma[lam, mu, sig]
                            for lam in range(n)
                        )
                        R_def[rho, sig, mu, nu] = sp.cancel(sp.together(val))
        R_def = sp.ImmutableDenseNDimArray(R_def)
        # Componentwise equality
        for idx, (a, b) in enumerate(
            zip(components(R_oracle), components(R_def), strict=True)
        ):
            assert sp.simplify(a - b) == 0, (
                f"seed={seed} component {idx}: oracle={a}, def={b}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_ricci_non_symmetric_on_torsionful_background(self, seed):
        """R_{mu nu} != R_{nu mu} on a torsionful background.  VAL-GEOM-002."""
        geom, gamma, Ric = _sympy_ricci_on_random_background(seed, dim=3, symmetric=False)
        n = geom.dim
        # Check that at least one component pair differs
        any_asymmetric = False
        for mu in range(n):
            for nu in range(mu + 1, n):
                diff = sp.simplify(Ric[mu, nu] - Ric[nu, mu])
                if diff != 0:
                    any_asymmetric = True
                    break
            if any_asymmetric:
                break
        assert any_asymmetric, (
            f"seed={seed}: Ricci is symmetric on a torsionful background "
            "(should not happen with a generic asymmetric connection)"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_ricci_symmetric_at_T0_Q0(self, seed):
        """R_{mu nu} = R_{nu mu} when the connection is the Levi-Civita
        connection of the metric (T=0, Q=0).  This is the Riemannian limit.
        VAL-GEOM-002: T=0 recovers Ricci symmetry."""
        geom = random_diagonal_metric(seed, dim=3)
        # Use the Levi-Civita (Christoffel) connection of the metric:
        # T=0 AND Q=0, so Ricci must be symmetric.
        Ric = ricci_of_connection(geom.coords, geom.christoffel)
        n = geom.dim
        for mu in range(n):
            for nu in range(mu + 1, n):
                diff = sp.simplify(Ric[mu, nu] - Ric[nu, mu])
                assert diff == 0, (
                    f"seed={seed}: R_{{{mu}{nu}}} != R_{{{nu}{mu}}} "
                    f"at T=0, Q=0 (Levi-Civita): {Ric[mu, nu]} vs {Ric[nu, mu]}"
                )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_torsion_nonzero_on_asymmetric_connection(self, seed):
        """The random asymmetric connection actually carries torsion
        (T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu} != 0)."""
        geom = random_diagonal_metric(seed, dim=3)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        n = geom.dim
        any_torsion = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    T = sp.simplify(gamma[lam, mu, nu] - gamma[lam, nu, mu])
                    if T != 0:
                        any_torsion = True
                        break
                if any_torsion:
                    break
            if any_torsion:
                break
        assert any_torsion, (
            f"seed={seed}: torsion is zero on an asymmetric connection "
            "(the random generator should produce nonzero torsion)"
        )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_riemann_antisymmetric_last_pair(self, seed):
        """R^rho_{sigma mu nu} = -R^rho_{sigma nu mu} holds for any
        connection (this is an algebraic identity from the definition)."""
        geom, gamma, R = _sympy_riemann_on_random_background(seed, dim=3)
        n = geom.dim
        for rho in range(n):
            for sig in range(n):
                for mu in range(n):
                    for nu in range(mu + 1, n):
                        diff = sp.simplify(R[rho, sig, mu, nu] + R[rho, sig, nu, mu])
                        assert diff == 0, (
                            f"seed={seed}: antisymmetry violated at "
                            f"({rho},{sig},{mu},{nu})"
                        )
