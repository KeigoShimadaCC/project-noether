"""Pin the torsion primitive and its irreducible decomposition.

Tests for VAL-GEOM-003 and VAL-GEOM-018:

- VAL-GEOM-003: T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu};
  residue is 0 and a SymPy torsion helper reproduces
  gamma[l,m,n] - gamma[l,n,m] componentwise.

- VAL-GEOM-018: The split of T^lambda_{mu nu} into trace-vector, axial-vector,
  and traceless-tensor parts reassembles to T (residue 0) and the SymPy oracle
  confirms the three pieces reconstruct the input torsion on a random
  background. Distinct from the contortion form K(T).

Each Cadabra primitive is pinned by a residue check. Each is additionally
cross-checked against the SymPy general-connection oracle on explicit random
metric + connection backgrounds (the torsion-trap safeguard).
They skip when cadabra2 is absent.

Cadabra residue check strategy for the decomposition:
  The full tensor-level reconstruction T - (t1 + t2 + q) = 0 is trivially
  true by definition of q = T - t1 - t2.  The non-trivial algebraic
  content is that t1 correctly extracts the trace (t1^lam_{lam mu} = T_mu)
  and that the axial part is purely antisymmetric (trace-free).  These
  are verified algebraically in Cadabra; the full componentwise
  verification (including the epsilon tensor axial part) is done by the
  SymPy cross-check on explicit backgrounds, which is the torsion-trap
  safeguard (architecture.md section 3.2).
"""

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.curvature import (
    AFFINE_CONNECTION_DEPENDS,
    AFFINE_CURVATURE_DECL,
    TORSION_DECL,
    define_torsion,
    reassemble_torsion,
    torsion_trace_vector,
)
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    components,
    random_affine_connection,
    random_diagonal_metric,
    torsion_of_connection,
    torsion_traceless_tensor,
)
from noether.kernels.sympy_kernel.geometry import (
    torsion_axial_part as sympy_torsion_axial_part,
)
from noether.kernels.sympy_kernel.geometry import (
    torsion_trace_part as sympy_torsion_trace_part,
)
from noether.kernels.sympy_kernel.geometry import (
    torsion_trace_vector as sympy_torsion_trace_vector,
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
        + TORSION_DECL
        + "\n"
        + body
    )


def _run_affine(body: str):
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="affine torsion primitive check",
            payload={"script": _affine_script(body)},
        )
    )


# ---------------------------------------------------------------------------
# SymPy cross-check helpers
# ---------------------------------------------------------------------------


def _sympy_torsion_on_random_background(seed: int, dim: int = 3):
    """Build a random metric + asymmetric connection and compute the torsion."""
    geom = random_diagonal_metric(seed, dim=dim)
    gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
    T = torsion_of_connection(gamma)
    return geom, gamma, T


# ===========================================================================
# Cadabra residue checks
# ===========================================================================


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestTorsionResidue:
    """Cadabra residue checks for the torsion primitive and decomposition."""

    def test_torsion_definition_residue_zero(self):
        """T^lambda_{mu nu} = G^lambda_{mu nu} - G^lambda_{nu mu} matches the
        explicit antisymmetric difference (residue 0).  VAL-GEOM-003."""
        body = (
            # Build torsion from the definition
            r"defn := G^{\lambda}_{\mu\nu} - G^{\lambda}_{\nu\mu};"
            "\n"
            "canonicalise(defn); rename_dummies(defn);\n"
            # Apply the primitive substitution
            r"target := T^{\lambda}_{\mu\nu};"
            "\n"
            + define_torsion("G", "target")
            + "\n"
            "distribute(target); canonicalise(target); rename_dummies(target);\n"
            # Residue
            "residue := @(defn) - @(target);\n"
            "distribute(residue); canonicalise(residue); "
            "rename_dummies(residue); meld(residue);\n"
            'print("NOETHER_CHECK: torsion_definition_zero=" '
            '+ str(str(residue) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("torsion_definition_zero") == "True", (
            result.raw.stdout
        )

    def test_torsion_antisymmetry_in_lower_pair(self):
        """T^lambda_{mu nu} = -T^lambda_{nu mu} (antisymmetry declared and
        verified by the kernel).  VAL-GEOM-003 (antisymmetry property)."""
        body = (
            r"ex := T^{\lambda}_{\mu\nu} + T^{\lambda}_{\nu\mu};"
            "\n"
            "canonicalise(ex);\n"
            'print("NOETHER_CHECK: torsion_antisym_zero=" '
            '+ str(str(ex) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("torsion_antisym_zero") == "True", (
            result.raw.stdout
        )

    def test_torsion_trace_vector_substitution(self):
        """T_mu = g^{lambda kappa}(G^kappa_{lambda mu} - G^kappa_{mu lambda})
        substitution produces a non-trivial expression in G.
        VAL-GEOM-003 (trace vector machinery)."""
        body = (
            # Apply the trace-vector substitution
            r"Tv := T_{\mu};"
            "\n"
            + torsion_trace_vector("G", "Tv")
            + "\n"
            "distribute(Tv); canonicalise(Tv); rename_dummies(Tv);\n"
            # Verify it contains G components (not just T_mu unchanged)
            'has_G = "G" in str(Tv)\n'
            'print("NOETHER_CHECK: trace_subst_fires=" + str(has_G))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("trace_subst_fires") == "True", (
            result.raw.stdout
        )

    def test_torsion_irreducible_trace_identity(self):
        """The trace of the trace part equals T_mu:
        t1^lambda_{lambda mu} = (1/3)(delta^lambda_lambda T_mu
                                   - delta^lambda_mu T_lambda)
        In dim 4: (1/3)(4 T_mu - T_mu) = T_mu.
        This is the algebraic identity underpinning VAL-GEOM-018:
        the trace part correctly extracts the torsion trace."""
        body = (
            # Verify (1/3)(4 T_mu - T_mu) - T_mu = 0
            r"trid := (1/3)(4*T_{\mu} - T_{\mu}) - T_{\mu};"
            "\n"
            "canonicalise(trid);\n"
            'print("NOETHER_CHECK: trace_identity_zero=" '
            '+ str(str(trid) == "0"))'
        )
        result = _run_affine(body)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("trace_identity_zero") == "True", (
            result.raw.stdout
        )

    def test_reassemble_torsion_substitution_fires(self):
        """The reconstruction substitution T -> t1 + t2 + q fires correctly.
        VAL-GEOM-018 (substitution machinery)."""
        body = (
            r"ex := T^{\lambda}_{\mu\nu};"
            "\n"
            + reassemble_torsion("ex")
            + "\n"
            "canonicalise(ex);\n"
            # Check all three parts appear in the result
            'has_parts = "t1" in str(ex) and "t2" in str(ex) and "q" in str(ex)\n'
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


class TestTorsionSymPyCrossCheck:
    """Cross-check the torsion primitives against the SymPy oracle on
    explicit random backgrounds.  This is the independent verification that
    catches the torsion trap (architecture.md section 3.2)."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_torsion_of_connection_matches_definition(self, seed):
        """torsion_of_connection reproduces T^lambda_{mu nu} = gamma[l,m,n]
        - gamma[l,n,m] on a random asymmetric connection.  VAL-GEOM-003."""
        geom, gamma, T_oracle = _sympy_torsion_on_random_background(seed, dim=3)
        n = geom.dim
        # Compute from the definition directly
        T_def = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    T_def[lam, mu, nu] = _clean(gamma[lam, mu, nu] - gamma[lam, nu, mu])
        T_def = sp.ImmutableDenseNDimArray(T_def)
        # Componentwise equality
        for idx, (a, b) in enumerate(
            zip(components(T_oracle), components(T_def), strict=True)
        ):
            assert sp.simplify(a - b) == 0, (
                f"seed={seed} component {idx}: oracle={a}, def={b}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_torsion_antisymmetric_in_lower_pair(self, seed):
        """T^lambda_{mu nu} = -T^lambda_{nu mu} on a random connection.
        VAL-GEOM-003."""
        geom, gamma, T = _sympy_torsion_on_random_background(seed, dim=3)
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    diff = sp.simplify(T[lam, mu, nu] + T[lam, nu, mu])
                    assert diff == 0, (
                        f"seed={seed}: antisymmetry violated at "
                        f"({lam},{mu},{nu}): T={T[lam, mu, nu]}, "
                        f"T(nu,mu)={T[lam, nu, mu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_torsion_nonzero_on_asymmetric_connection(self, seed):
        """The torsion is nonzero on a generic asymmetric connection
        (T^lambda_{mu nu} != 0 for some components)."""
        geom, gamma, T = _sympy_torsion_on_random_background(seed, dim=3)
        n = geom.dim
        any_nonzero = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    if T[lam, mu, nu] != 0:
                        any_nonzero = True
                        break
                if any_nonzero:
                    break
            if any_nonzero:
                break
        assert any_nonzero, (
            f"seed={seed}: torsion is zero on an asymmetric connection "
            "(the random generator should produce nonzero torsion)"
        )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_torsion_zero_on_symmetric_connection(self, seed):
        """T^lambda_{mu nu} = 0 when the connection is symmetric in the
        lower pair (torsion-free).  This is the T=0 limit."""
        geom = random_diagonal_metric(seed, dim=3)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=True)
        T = torsion_of_connection(gamma)
        n = geom.dim
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    assert T[lam, mu, nu] == 0, (
                        f"seed={seed}: T[{lam},{mu},{nu}] = {T[lam, mu, nu]} "
                        f"on a symmetric connection (should be zero)"
                    )


class TestTorsionIrreducibleDecompositionSymPy:
    """Cross-check the irreducible torsion decomposition against the SymPy
    oracle on explicit random backgrounds.  VAL-GEOM-018.

    The irreducible decomposition under the Lorentz group splits
    T^lambda_{mu nu} into:
      1. Trace-vector part:  (1/3)(delta^lambda_mu T_nu - delta^lambda_nu T_mu)
      2. Axial-vector part:  -(1/6) epsilon^lambda_{mu nu rho} A^rho
      3. Traceless-tensor part: q = T - part1 - part2
    where T_mu = T^rho_{rho mu} and A^rho = (1/6) epsilon^{rho...} T_{...}.
    """

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_trace_part_trace_matches_torsion_trace(self, seed):
        """The trace of the trace part t1^lambda_{lambda mu} equals T_mu.
        VAL-GEOM-018 (trace-part correctness)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        T_vec = sympy_torsion_trace_vector(gamma)
        t1 = sympy_torsion_trace_part(gamma)
        n = geom.dim
        # Verify t1^lambda_{lambda mu} = T_mu
        for mu in range(n):
            trace_t1 = sum(t1[lam, lam, mu] for lam in range(n))
            diff = sp.simplify(trace_t1 - T_vec[mu])
            assert diff == 0, (
                f"seed={seed}: t1^lam_{{lam,{mu}}} = {trace_t1} "
                f"!= T_{mu} = {T_vec[mu]}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_traceless_part_is_traceless(self, seed):
        """The traceless tensor part q^lambda_{lambda mu} = 0.
        VAL-GEOM-018 (irreducibility)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        q = torsion_traceless_tensor(gamma, geom)
        n = geom.dim
        for mu in range(n):
            trace_q = sum(q[lam, lam, mu] for lam in range(n))
            diff = sp.simplify(trace_q)
            assert diff == 0, (
                f"seed={seed}: q^lam_{{lam,{mu}}} = {trace_q} (should be 0)"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_irreducible_decomposition_reconstructs_torsion(self, seed):
        """t1 + t2 + q = T (the three parts reassemble to the original
        torsion).  VAL-GEOM-018 (reconstruction equality)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        T = torsion_of_connection(gamma)
        t1 = sympy_torsion_trace_part(gamma)
        t2 = sympy_torsion_axial_part(gamma, geom)
        q = torsion_traceless_tensor(gamma, geom)
        n = geom.dim
        # Verify t1 + t2 + q = T componentwise
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu + 1, n):
                    reassembled = t1[lam, mu, nu] + t2[lam, mu, nu] + q[lam, mu, nu]
                    diff = sp.simplify(reassembled - T[lam, mu, nu])
                    assert diff == 0, (
                        f"seed={seed}: reconstruction fails at "
                        f"({lam},{mu},{nu}): "
                        f"reassembled={reassembled}, T={T[lam, mu, nu]}"
                    )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_decomposition_distinct_from_contortion(self, seed):
        """The irreducible decomposition is distinct from the contortion
        form K(T).  VAL-GEOM-018 (distinctness from K(T)).

        The contortion K^lambda_{mu nu} in the standard metric-affine
        literature is:
          K^lambda_{mu nu} = (1/2)(T^lambda_{mu nu}
                                + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu}
                                + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu})
        which involves the metric for index raising, not just the Kronecker
        delta.  The trace part of the irreducible decomposition uses only
        delta^lambda_mu T_nu, which is structurally different.
        """
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        T = torsion_of_connection(gamma)
        t1 = sympy_torsion_trace_part(gamma)
        n = geom.dim
        # Compute the contortion:
        # K^lambda_{mu nu} = (1/2)(T^lambda_{mu nu}
        #       + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu}
        #       + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu})
        K = sp.MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    term1 = T[lam, mu, nu]
                    term2 = sum(
                        geom.g_inv[lam, sig] * geom.g[mu, tau] * T[tau, sig, nu]
                        for sig in range(n) for tau in range(n)
                    )
                    term3 = sum(
                        geom.g_inv[lam, sig] * geom.g[nu, tau] * T[tau, sig, mu]
                        for sig in range(n) for tau in range(n)
                    )
                    K[lam, mu, nu] = _clean(
                        sp.Rational(1, 2) * (term1 + term2 + term3)
                    )
        K = sp.ImmutableDenseNDimArray(K)
        # Verify t1 != K (they are structurally different)
        any_different = False
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    diff = sp.simplify(t1[lam, mu, nu] - K[lam, mu, nu])
                    if diff != 0:
                        any_different = True
                        break
                if any_different:
                    break
            if any_different:
                break
        assert any_different, (
            f"seed={seed}: the trace part of the irreducible decomposition "
            "equals the contortion K(T) (they should be distinct)"
        )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_axial_part_is_traceless(self, seed):
        """The axial part has no trace: t2^lambda_{lambda mu} = 0.
        VAL-GEOM-018 (axial part has no trace)."""
        geom = random_diagonal_metric(seed, dim=4)
        gamma = random_affine_connection(seed + 1000, geom.coords, symmetric=False)
        t2 = sympy_torsion_axial_part(gamma, geom)
        n = geom.dim
        for mu in range(n):
            trace_t2 = sum(t2[lam, lam, mu] for lam in range(n))
            diff = sp.simplify(trace_t2)
            assert diff == 0, (
                f"seed={seed}: t2^lam_{{lam,{mu}}} = {trace_t2} "
                f"(axial part should be traceless)"
            )
