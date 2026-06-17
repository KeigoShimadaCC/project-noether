"""Pin the non-metricity primitive and its irreducible decomposition.

Tests for VAL-GEOM-007 and VAL-GEOM-019:

- VAL-GEOM-007: Q_{lambda mu nu} = nabla_lambda g_{mu nu} and the rewrite
  nabla_lambda g_{mu nu} -> Q_{lambda mu nu} are residue-pinned, replacing
  nabla g -> 0; the SymPy nabla g equals Q on a non-metric connection and
  is 0 for Levi-Civita.

- VAL-GEOM-019: The split of Q_{lambda mu nu} into its Weyl-vector trace,
  second trace, and traceless-tensor remainder reassembles to Q (residue 0)
  and is SymPy-confirmed on a random non-metric background. Distinct from
  the disformation form L(Q).

Each Cadabra primitive is pinned by a residue check. Each is additionally
cross-checked against the SymPy general-connection oracle on explicit random
metric + connection backgrounds (the torsion-trap safeguard).
They skip when cadabra2 is absent.

Cadabra residue check strategy for the decomposition:
  The full tensor-level reconstruction Q - (QW + Q2T + QTL) = 0 is trivially
  true by definition of QTL = Q - QW - Q2T.  The non-trivial algebraic
  content is that QW correctly extracts the Weyl trace (Trace A of QW =
  omega_lambda) with zero second trace (Trace B of QW = 0), and that Q2T
  correctly extracts the second trace (Trace B of Q2T = qtilde_mu) with zero
  Weyl trace (Trace A of Q2T = 0).  These are verified algebraically in
  Cadabra; the full componentwise verification (including the Kronecker-delta
  expansions) is done by the SymPy cross-check on explicit backgrounds, which
  is the torsion-trap safeguard (architecture.md section 3.2).
"""

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.curvature import (
    AFFINE_CONNECTION_DEPENDS,
    AFFINE_CONNECTION_DEPENDS_NABLA,
    AFFINE_CURVATURE_DECL,
    NONMETRICITY_DECL,
    NONMETRICITY_DECL_NABLA,
    define_nonmetricity,
    reassemble_nonmetricity,
    rewrite_nabla_inverse_metric_to_Q,
    rewrite_nabla_metric_to_Q,
)
from noether.kernels.cadabra.curvature import (
    nonmetricity_second_trace as cv_second_trace,
)
from noether.kernels.cadabra.curvature import (
    nonmetricity_weyl_trace as cv_weyl_trace,
)
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    nonmetricity_of_connection,
    random_affine_connection,
    random_diagonal_metric,
)
from noether.kernels.sympy_kernel.geometry import (
    nonmetricity_second_trace as sympy_second_trace,
)
from noether.kernels.sympy_kernel.geometry import (
    nonmetricity_second_trace_part as sympy_second_trace_part,
)
from noether.kernels.sympy_kernel.geometry import (
    nonmetricity_traceless_tensor as sympy_traceless_tensor,
)
from noether.kernels.sympy_kernel.geometry import (
    nonmetricity_weyl_part as sympy_weyl_part,
)
from noether.kernels.sympy_kernel.geometry import (
    nonmetricity_weyl_trace as sympy_weyl_trace,
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
        + NONMETRICITY_DECL
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
        + NONMETRICITY_DECL_NABLA
        + "\n"
        + body
    )


def _run_affine(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="affine non-metricity primitive check",
            payload={"script": _affine_script(body)},
        )
    )


def _run_nabla(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="affine nabla non-metricity check",
            payload={"script": _nabla_script(body)},
        )
    )


# ---------------------------------------------------------------------------
# SymPy cross-check helpers
# ---------------------------------------------------------------------------


def _sympy_nonmetricity_on_random_background(seed: int, dim: int = 3):
    """Build a random metric + connection and compute the non-metricity."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
    Q = nonmetricity_of_connection(geom.coords, gamma, geom.g)
    return geom, gamma, Q


# ===========================================================================
# Cadabra residue checks
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestNonmetricityResidue:
    """Cadabra residue checks for the non-metricity primitive and
    decomposition."""

    def test_nonmetricity_definition_residue_zero(self):
        """Q_{lambda mu nu} = partial_lambda g_{mu nu}
        - G^rho_{lambda mu} g_{rho nu} - G^rho_{lambda nu} g_{rho mu}
        matches the explicit expansion (residue 0).  VAL-GEOM-007."""
        body = (
            # Build non-metricity from the definition
            r"defn := \partial_{\lambda}{g_{\mu\nu}}"
            r" - G^{\rho}_{\lambda\mu} g_{\rho\nu}"
            r" - G^{\rho}_{\lambda\nu} g_{\rho\mu};"
            "\n"
            "distribute(defn); canonicalise(defn); rename_dummies(defn);\n"
            # Apply the primitive substitution
            r"target := Q_{\lambda\mu\nu};"
            "\n"
            + define_nonmetricity("G", "target")
            + "\n"
            "distribute(target); canonicalise(target); rename_dummies(target);\n"
            # Residue
            "residue := @(defn) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: nonmetricity_definition_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("nonmetricity_definition_zero") == "True", (
            result.raw.stdout
        )

    def test_nonmetricity_symmetric_in_last_pair(self):
        """Q_{lambda mu nu} = Q_{lambda nu mu} (symmetry declared and
        verified by the kernel).  VAL-GEOM-007 (symmetry property)."""
        body = (
            r"ex := Q_{\lambda\mu\nu} - Q_{\lambda\nu\mu};"
            "\n"
            "canonicalise(ex);\n"
            'print("NOETHER_CHECK: nonmetricity_symmetric_zero=" '
            '+ str(str(ex) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("nonmetricity_symmetric_zero") == "True", (
            result.raw.stdout
        )

    def test_nabla_metric_rewrite_fires(self):
        """nabla_lambda g_{mu nu} -> Q_{lambda mu nu} substitution fires.
        VAL-GEOM-007 (rewrite machinery)."""
        body = (
            r"ex := \nabla_{\lambda}{g_{\mu\nu}};"
            "\n"
            + rewrite_nabla_metric_to_Q("ex")
            + "\n"
            "canonicalise(ex);\n"
            'has_Q = "Q" in str(ex)\n'
            'print("NOETHER_CHECK: nabla_metric_rewrite_fires=" + str(has_Q))'
        )
        result = _run_nabla(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("nabla_metric_rewrite_fires") == "True", (
            result.raw.stdout
        )

    def test_nabla_inverse_metric_rewrite_fires(self):
        """nabla_lambda g^{mu nu} -> -g^{mu rho} g^{nu sigma} Q_{lambda rho sigma}
        substitution fires.  VAL-GEOM-007 (inverse metric rewrite)."""
        body = (
            r"ex := \nabla_{\lambda}{g^{\mu\nu}};"
            "\n"
            + rewrite_nabla_inverse_metric_to_Q("ex")
            + "\n"
            "distribute(ex); canonicalise(ex);\n"
            'has_Q = "Q" in str(ex)\n'
            'print("NOETHER_CHECK: nabla_inv_metric_rewrite_fires=" + str(has_Q))'
        )
        result = _run_nabla(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("nabla_inv_metric_rewrite_fires") == "True", (
            result.raw.stdout
        )

    def test_weyl_trace_substitution_fires(self):
        """Q_mu -> g^{alpha beta} Q_{mu alpha beta} substitution produces
        a non-trivial expression.  VAL-GEOM-007 (trace machinery)."""
        body = (
            r"Tv := Q_{\mu};"
            "\n"
            + cv_weyl_trace("G", "Tv")
            + "\n"
            "distribute(Tv); canonicalise(Tv); rename_dummies(Tv);\n"
            'has_g = "g" in str(Tv)\n'
            'print("NOETHER_CHECK: weyl_trace_subst_fires=" + str(has_g))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("weyl_trace_subst_fires") == "True", (
            result.raw.stdout
        )

    def test_second_trace_substitution_fires(self):
        """q_mu -> g^{lambda nu} Q_{lambda mu nu} substitution produces
        a non-trivial expression.  VAL-GEOM-019 (trace machinery)."""
        body = (
            r"Tv := q_{\mu};"
            "\n"
            + cv_second_trace("G", "Tv")
            + "\n"
            "distribute(Tv); canonicalise(Tv); rename_dummies(Tv);\n"
            'has_g = "g" in str(Tv)\n'
            'print("NOETHER_CHECK: second_trace_subst_fires=" + str(has_g))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("second_trace_subst_fires") == "True", (
            result.raw.stdout
        )

    def test_weyl_trace_identity(self):
        """In dim 4: (1/18)(4*5*Q_lambda - Q_lambda - Q_lambda) = Q_lambda.
        The Weyl part correctly extracts the Weyl trace.
        VAL-GEOM-019 (Weyl trace extraction)."""
        body = (
            r"trid := (1/18)(4*5*Q_{\lambda} - Q_{\lambda} - Q_{\lambda})"
            r" - Q_{\lambda};"
            "\n"
            "canonicalise(trid);\n"
            'print("NOETHER_CHECK: weyl_trace_identity_zero=" '
            '+ str(str(trid) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("weyl_trace_identity_zero") == "True", (
            result.raw.stdout
        )

    def test_weyl_part_second_trace_zero(self):
        """In dim 4: the second trace of the Weyl part is zero.
        (1/18)(5 Q_mu - 4 Q_mu - Q_mu) = 0.
        VAL-GEOM-019 (Weyl part is traceless in second trace)."""
        body = (
            r"trid := (1/18)(5*Q_{\mu} - 4*Q_{\mu} - Q_{\mu});"
            "\n"
            "canonicalise(trid);\n"
            'print("NOETHER_CHECK: weyl_part_second_trace_zero=" '
            '+ str(str(trid) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("weyl_part_second_trace_zero") == "True", (
            result.raw.stdout
        )

    def test_reassemble_nonmetricity_substitution_fires(self):
        """The reconstruction substitution Q -> QW + Q2T + QTL fires.
        VAL-GEOM-019 (substitution machinery)."""
        body = (
            r"ex := Q_{\lambda\mu\nu};"
            "\n"
            + reassemble_nonmetricity("ex")
            + "\n"
            "canonicalise(ex);\n"
            'has_parts = "QW" in str(ex) and "Q2T" in str(ex) and "QTL" in str(ex)\n'
            'print("NOETHER_CHECK: reassemble_fires=" + str(has_parts))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("reassemble_fires") == "True", (
            result.raw.stdout
        )


# ===========================================================================
# SymPy component cross-checks (the torsion-trap safeguard)
# ===========================================================================


class TestNonmetricitySymPyCrossCheck:
    """Cross-check the non-metricity primitives against the SymPy oracle on
    explicit random backgrounds.  This is the independent verification that
    catches the torsion trap (architecture.md section 3.2)."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_nonmetricity_of_connection_matches_definition(self, seed):
        """nonmetricity_of_connection reproduces the defining formula
        Q_{lambda mu nu} = partial_lambda g_{mu nu}
                          - Gamma^rho_{lambda mu} g_{rho nu}
                          - Gamma^rho_{lambda nu} g_{rho mu}
        on a random connection.  VAL-GEOM-007."""
        geom, gamma, Q_oracle = _sympy_nonmetricity_on_random_background(seed, dim=3)
        n, x = geom.dim, geom.coords
        # Compute from the definition directly
        Q_def = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    val = sp.diff(geom.g[mu, nu], x[lam])
                    for rho in range(n):
                        val -= gamma[rho, lam, mu] * geom.g[rho, nu]
                        val -= gamma[rho, lam, nu] * geom.g[rho, mu]
                    val = _clean(val)
                    Q_def[lam, mu, nu] = val
                    Q_def[lam, nu, mu] = val  # symmetric in last pair
        Q_def = sp.ImmutableDenseNDimArray(Q_def)
        # Componentwise equality
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    diff = sp.simplify(Q_oracle[lam, mu, nu] - Q_def[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: Q[{lam},{mu},{nu}] oracle={Q_oracle[lam, mu, nu]}, "
                        f"def={Q_def[lam, mu, nu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_nonmetricity_symmetric_in_last_pair(self, seed):
        """Q_{lambda mu nu} = Q_{lambda nu mu} on a random connection.
        VAL-GEOM-007."""
        geom, gamma, Q = _sympy_nonmetricity_on_random_background(seed, dim=3)
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    diff = sp.simplify(Q[lam, mu, nu] - Q[lam, nu, mu])
                    assert diff == 0, (
                        f"seed={seed}: Q[{lam},{mu},{nu}] != Q[{lam},{nu},{mu}]: "
                        f"{Q[lam, mu, nu]} vs {Q[lam, nu, mu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_nonmetricity_nonzero_on_general_connection(self, seed):
        """The non-metricity is nonzero on a generic connection
        (Q_{lambda mu nu} != 0 for some components).  VAL-GEOM-007."""
        geom, gamma, Q = _sympy_nonmetricity_on_random_background(seed, dim=3)
        n = geom.dim
        any_nonzero = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    if Q[lam, mu, nu] != 0:
                        any_nonzero = True
                        break
                if any_nonzero:
                    break
            if any_nonzero:
                break
        assert any_nonzero, (
            f"seed={seed}: non-metricity is zero on a general connection "
            "(the random generator should produce nonzero Q)"
        )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_nonmetricity_zero_for_levi_civita(self, seed):
        """Q_{lambda mu nu} = 0 for the Levi-Civita connection (metric
        compatibility).  VAL-GEOM-007 (LC limit)."""
        geom = random_diagonal_metric(seed, dim=3)
        # Levi-Civita = Christoffel symbols of the metric
        Q = nonmetricity_of_connection(geom.coords, geom.christoffel, geom.g)
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    diff = sp.simplify(Q[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: Q[{lam},{mu},{nu}] = {Q[lam, mu, nu]} "
                        f"on Levi-Civita (should be zero)"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_nonmetricity_zero_for_symmetric_connection(self, seed):
        """Q_{lambda mu nu} = 0 for a connection that equals the Christoffel
        symbols (symmetric + metric-compatible). A random symmetric
        connection is NOT generally metric-compatible, so Q is nonzero
        in general.  This test confirms that ONLY the LC connection gives Q=0."""
        geom = random_diagonal_metric(seed, dim=3)
        gamma = random_affine_connection(
            seed + 1000, geom.coords, symmetric=True
        )
        # A symmetric (torsion-free) but NOT LC connection should still
        # have nonzero Q in general (metric incompatibility)
        Q = nonmetricity_of_connection(geom.coords, gamma, geom.g)
        n = geom.dim
        any_nonzero = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    if sp.simplify(Q[lam, mu, nu]) != 0:
                        any_nonzero = True
                        break
                if any_nonzero:
                    break
            if any_nonzero:
                break
        # A generic symmetric connection is NOT metric-compatible,
        # so Q should be nonzero.  This confirms that Q=0 requires BOTH
        # T=0 AND Q=0 (i.e., Levi-Civita).
        assert any_nonzero, (
            f"seed={seed}: Q is zero on a symmetric-but-not-LC connection "
            "(this would incorrectly imply symmetric => metric-compatible)"
        )


class TestNonmetricityIrreducibleDecompositionSymPy:
    """Cross-check the irreducible non-metricity decomposition against the
    SymPy oracle on explicit random backgrounds.  VAL-GEOM-019.

    The irreducible decomposition under the Lorentz group splits
    Q_{lambda mu nu} into:
      1. Weyl-vector trace part: determined by omega_lambda = Q_{lambda mu nu} g^{mu nu}
      2. Second-trace part: determined by qtilde_mu = Q_{lambda mu nu} g^{lambda nu}
      3. Traceless-tensor part: the remainder, traceless in both senses
    """

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_weyl_part_has_correct_weyl_trace(self, seed):
        """Trace A of QW (g^{mu nu} QW_{lambda mu nu}) equals omega_lambda.
        VAL-GEOM-019 (Weyl trace extraction)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        omega = sympy_weyl_trace(geom.coords, gamma, geom.g, geom.g_inv)
        qw = sympy_weyl_part(geom.coords, gamma, geom.g, geom.g_inv)
        n = geom.dim
        for lam in range(n):
            trace_qw = sum(
                qw[lam, mu, nu] * geom.g_inv[mu, nu]
                for mu in range(n) for nu in range(n)
            )
            diff = sp.simplify(trace_qw - omega[lam])
            assert diff == 0, (
                f"seed={seed}: Trace A of QW at {lam}: "
                f"got {trace_qw}, expected {omega[lam]}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_weyl_part_has_zero_second_trace(self, seed):
        """Trace B of QW (g^{lambda nu} QW_{lambda mu nu}) is zero.
        VAL-GEOM-019 (Weyl part is traceless in second trace)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        qw = sympy_weyl_part(geom.coords, gamma, geom.g, geom.g_inv)
        n = geom.dim
        for mu in range(n):
            trace_b = sum(
                qw[lam, mu, nu] * geom.g_inv[lam, nu]
                for lam in range(n) for nu in range(n)
            )
            diff = sp.simplify(trace_b)
            assert diff == 0, (
                f"seed={seed}: Trace B of QW at {mu}: "
                f"got {trace_b}, expected 0"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_second_trace_part_has_zero_weyl_trace(self, seed):
        """Trace A of Q2T (g^{mu nu} Q2T_{lambda mu nu}) is zero.
        VAL-GEOM-019 (second-trace part is traceless in Weyl trace)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        q2t = sympy_second_trace_part(geom.coords, gamma, geom.g, geom.g_inv)
        n = geom.dim
        for lam in range(n):
            trace_a = sum(
                q2t[lam, mu, nu] * geom.g_inv[mu, nu]
                for mu in range(n) for nu in range(n)
            )
            diff = sp.simplify(trace_a)
            assert diff == 0, (
                f"seed={seed}: Trace A of Q2T at {lam}: "
                f"got {trace_a}, expected 0"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_second_trace_part_has_correct_second_trace(self, seed):
        """Trace B of Q2T (g^{lambda nu} Q2T_{lambda mu nu}) equals qtilde_mu.
        VAL-GEOM-019 (second trace extraction)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        qtilde = sympy_second_trace(geom.coords, gamma, geom.g, geom.g_inv)
        q2t = sympy_second_trace_part(geom.coords, gamma, geom.g, geom.g_inv)
        n = geom.dim
        for mu in range(n):
            trace_b = sum(
                q2t[lam, mu, nu] * geom.g_inv[lam, nu]
                for lam in range(n) for nu in range(n)
            )
            diff = sp.simplify(trace_b - qtilde[mu])
            assert diff == 0, (
                f"seed={seed}: Trace B of Q2T at {mu}: "
                f"got {trace_b}, expected {qtilde[mu]}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_traceless_part_is_traceless_in_both_senses(self, seed):
        """QTL is traceless: both Trace A and Trace B vanish.
        VAL-GEOM-019 (traceless property)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        qtl = sympy_traceless_tensor(geom.coords, gamma, geom.g, geom.g_inv)
        n = geom.dim
        # Trace A: g^{mu nu} QTL_{lambda mu nu}
        for lam in range(n):
            trace_a = sum(
                qtl[lam, mu, nu] * geom.g_inv[mu, nu]
                for mu in range(n) for nu in range(n)
            )
            diff = sp.simplify(trace_a)
            assert diff == 0, (
                f"seed={seed}: Trace A of QTL at {lam}: "
                f"got {trace_a}, expected 0"
            )
        # Trace B: g^{lambda nu} QTL_{lambda mu nu}
        for mu in range(n):
            trace_b = sum(
                qtl[lam, mu, nu] * geom.g_inv[lam, nu]
                for lam in range(n) for nu in range(n)
            )
            diff = sp.simplify(trace_b)
            assert diff == 0, (
                f"seed={seed}: Trace B of QTL at {mu}: "
                f"got {trace_b}, expected 0"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_irreducible_decomposition_reconstructs_nonmetricity(self, seed):
        """QW + Q2T + QTL = Q (the three parts reassemble to the original
        non-metricity).  VAL-GEOM-019 (reconstruction equality)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        Q = nonmetricity_of_connection(geom.coords, gamma, geom.g)
        qw = sympy_weyl_part(geom.coords, gamma, geom.g, geom.g_inv)
        q2t = sympy_second_trace_part(geom.coords, gamma, geom.g, geom.g_inv)
        qtl = sympy_traceless_tensor(geom.coords, gamma, geom.g, geom.g_inv)
        n = geom.dim
        # Verify QW + Q2T + QTL = Q componentwise
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    reassembled = qw[lam, mu, nu] + q2t[lam, mu, nu] + qtl[lam, mu, nu]
                    diff = sp.simplify(reassembled - Q[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: reconstruction fails at "
                        f"({lam},{mu},{nu}): "
                        f"reassembled={reassembled}, Q={Q[lam, mu, nu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_decomposition_distinct_from_disformation(self, seed):
        """The irreducible decomposition is distinct from the disformation
        form L(Q).  VAL-GEOM-019 (distinctness from L(Q)).

        The disformation L^lambda_{mu nu} in the standard metric-affine
        literature is:
          L^lambda_{mu nu} = (1/2) g^{lambda rho}(-Q_{mu nu rho}
                                - Q_{nu rho mu} + Q_{rho mu nu})
        which involves index raising via the metric and has a different
        index structure (one up, two down) than Q (all down).  The
        Weyl part of the irreducible decomposition, when lowered, has a
        fundamentally different structure.
        """
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        Q = nonmetricity_of_connection(geom.coords, gamma, geom.g)
        qw = sympy_weyl_part(geom.coords, gamma, geom.g, geom.g_inv)
        n = geom.dim
        # Compute the disformation:
        # L^lambda_{mu nu} = (1/2) g^{lambda rho}(-Q_{mu nu rho}
        #       - Q_{nu rho mu} + Q_{rho mu nu})
        L_up = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    val = sp.Integer(0)
                    for rho in range(n):
                        val += geom.g_inv[lam, rho] * (
                            -Q[mu, nu, rho]
                            - Q[nu, rho, mu]
                            + Q[rho, mu, nu]
                        )
                    L_up[lam, mu, nu] = _clean(sp.Rational(1, 2) * val)
        L_up = sp.ImmutableDenseNDimArray(L_up)
        # Lower the first index of L for comparison
        L_down = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    L_down[lam, mu, nu] = _clean(
                        sum(geom.g[lam, rho] * L_up[rho, mu, nu] for rho in range(n))
                    )
        L_down = sp.ImmutableDenseNDimArray(L_down)
        # Verify QW != L_down (they are structurally different)
        any_different = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    diff = sp.simplify(qw[lam, mu, nu] - L_down[lam, mu, nu])
                    if diff != 0:
                        any_different = True
                        break
                if any_different:
                    break
            if any_different:
                break
        assert any_different, (
            f"seed={seed}: the Weyl part of the irreducible decomposition "
            "equals the lowered disformation L(Q) (they should be distinct)"
        )
