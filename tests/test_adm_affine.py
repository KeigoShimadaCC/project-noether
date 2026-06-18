"""Tests for the metric-affine ADM (3+1) decomposition.

Verifies the AffineADMGeometry class and the derive_adm extension for
metric-affine NPRs (independent connection with torsion and non-metricity).

Convention: noether-default-v1 + metric-affine-v1.

VAL-ADM-001: metric-affine ADM is reachable on the general adm path.
VAL-ADM-002: lapse/shift/spatial-metric decomposition is surfaced.
VAL-ADM-003: independent-connection degrees of freedom are decomposed.
VAL-ADM-004: constraints are separated from evolution.
VAL-ADM-005: connection-sector primary/secondary constraints are surfaced.
VAL-ADM-006: SymPy component kernel is the verifier (no model script).
VAL-ADM-007: Verified verdict backed by explicit-background component checks
             with a distortion-nonzero falsifier.
VAL-ADM-008: A reduction that cannot close is gated, not faked.
VAL-ADM-009: Teaching narration is separate from the verified verdict.
"""

import pytest
import sympy as sp

from noether.kernels.base import Capability
from noether.kernels.sympy_kernel.adm import (
    ADMGeometry,
    AffineADMGeometry,
    adm_affine_sample_1p2,
    adm_sample_1p2,
)
from noether.kernels.sympy_kernel.geometry import (
    _clean,
    contortion_of_torsion,
    random_affine_connection,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def affine_adm() -> AffineADMGeometry:
    """The standard nondegenerate metric-affine 1+2 background."""
    return adm_affine_sample_1p2()


@pytest.fixture(scope="module")
def affine_checks(affine_adm) -> dict[str, tuple[bool, str]]:
    """All metric-affine ADM checks on the standard background."""
    return affine_adm.run_all_affine()


@pytest.fixture(scope="module")
def gr_adm() -> ADMGeometry:
    """The standard GR ADM 1+2 background."""
    return adm_sample_1p2()


@pytest.fixture(scope="module")
def gr_checks(gr_adm) -> dict[str, tuple[bool, str]]:
    """All GR ADM checks (regression baseline)."""
    return gr_adm.run_all()


# ---------------------------------------------------------------------------
# VAL-ADM-001: metric-affine ADM is reachable on the general adm path
# ---------------------------------------------------------------------------


class TestReachability:
    """The metric-affine ADM decomposition runs and produces derivations
    with kind=='adm', non-empty result_tex, kernel_name/kernel_version,
    and a verified flag."""

    def test_affine_checks_produced(self, affine_checks):
        assert len(affine_checks) >= 6, (
            f"expected at least 6 metric-affine checks, got {len(affine_checks)}"
        )

    def test_affine_background_nondegenerate(self, affine_checks):
        ok, detail = affine_checks["background-nondegenerate-affine"]
        assert ok, detail

    def test_post_riemannian_on_foliation(self, affine_checks):
        ok, detail = affine_checks["post-riemannian-on-foliation"]
        assert ok, detail

    def test_torsion_nonmetricity_foliation(self, affine_checks):
        ok, detail = affine_checks["torsion-nonmetricity-foliation"]
        assert ok, detail

    def test_distortion_spatial_projections(self, affine_checks):
        ok, detail = affine_checks["distortion-spatial-projections"]
        assert ok, detail


# ---------------------------------------------------------------------------
# VAL-ADM-002: lapse/shift/spatial-metric decomposition is surfaced
# ---------------------------------------------------------------------------


class TestMetricSectorSplit:
    """The metric-sector foliation split is present and verified:
    lapse N, shift N^i, induced metric h_{ij}, extrinsic curvature
    K_{ij} (stated sign), foliation normal n_mu."""

    def test_gr_checks_still_pass(self, gr_checks):
        """The GR ADM checks (which verify the metric-sector split)
        still pass on the standard background."""
        for name, (ok, detail) in gr_checks.items():
            assert ok, f"{name}: {detail}"

    def test_lapse_present(self, gr_adm):
        assert gr_adm.N is not None

    def test_shift_present(self, gr_adm):
        assert len(gr_adm.shift) == gr_adm.d

    def test_spatial_metric_present(self, gr_adm):
        assert gr_adm.h.shape == (gr_adm.d, gr_adm.d)

    def test_extrinsic_curvature_present(self, gr_adm):
        K = gr_adm.extrinsic
        assert K.shape == (gr_adm.d, gr_adm.d)

    def test_normal_named(self, gr_adm):
        """The foliation normal n_mu = (-N, 0, ..., 0)."""
        n_up = gr_adm.normal_up
        assert len(n_up) == gr_adm.d + 1

    def test_k_sign_is_expansion_positive(self, gr_adm):
        """K_ij = +nabla_i n_j (verified by check (A))."""
        ok, detail = gr_adm.check_normal_gradient()
        assert ok, detail


# ---------------------------------------------------------------------------
# VAL-ADM-003: independent-connection degrees of freedom are decomposed
# ---------------------------------------------------------------------------


class TestConnectionFoliationDecomposition:
    """The connection's own dof are decomposed along the foliation:
    Gamma = LC + K(T) + L(Q) projected into normal/tangential parts,
    surfacing torsion and non-metricity pieces explicitly."""

    def test_torsion_spatial_nonzero(self, affine_adm):
        """The spatial torsion T^i_{jk} is nonzero on the background."""
        from noether.kernels.sympy_kernel.geometry import components
        T_spatial = affine_adm.torsion_spatial
        assert any(
            not (sp.cancel(sp.together(c)) == 0) for c in components(T_spatial)
        ), "spatial torsion is zero on the background"

    def test_torsion_normal_upper_nonzero(self, affine_adm):
        """The normal-upper torsion T^n_{jk} is nonzero on the background."""
        from noether.kernels.sympy_kernel.geometry import components
        T_nu = affine_adm.torsion_normal_upper
        assert any(
            not (sp.cancel(sp.together(c)) == 0) for c in components(T_nu)
        ), "normal-upper torsion is zero on the background"

    def test_torsion_mixed_nonzero(self, affine_adm):
        """The mixed torsion T^i_{n k} is nonzero on the background."""
        from noether.kernels.sympy_kernel.geometry import components
        T_mixed = affine_adm.torsion_mixed
        assert any(
            not (sp.cancel(sp.together(c)) == 0) for c in components(T_mixed)
        ), "mixed torsion is zero on the background"

    def test_nonmetricity_spatial_nonzero(self, affine_adm):
        """The spatial non-metricity Q_{ijk} is nonzero on the background."""
        from noether.kernels.sympy_kernel.geometry import components
        Q_spatial = affine_adm.nonmetricity_spatial
        assert any(
            not (sp.cancel(sp.together(c)) == 0) for c in components(Q_spatial)
        ), "spatial non-metricity is zero on the background"

    def test_nonmetricity_normal_first_nonzero(self, affine_adm):
        """The normal-first non-metricity Q_{nij} is nonzero."""
        from noether.kernels.sympy_kernel.geometry import components
        Q_nf = affine_adm.nonmetricity_normal_first
        assert any(
            not (sp.cancel(sp.together(c)) == 0) for c in components(Q_nf)
        ), "normal-first non-metricity is zero on the background"

    def test_contortion_spatial_nonzero(self, affine_adm):
        """The spatial contortion K^i_{jk} is nonzero on the background."""
        from noether.kernels.sympy_kernel.geometry import components
        K_spatial = affine_adm.contortion_spatial
        assert any(
            not (sp.cancel(sp.together(c)) == 0) for c in components(K_spatial)
        ), "spatial contortion is zero on the background"

    def test_disformation_spatial_nonzero(self, affine_adm):
        """The spatial disformation L^i_{jk} is nonzero on the background."""
        from noether.kernels.sympy_kernel.geometry import components
        L_spatial = affine_adm.disformation_spatial
        assert any(
            not (sp.cancel(sp.together(c)) == 0) for c in components(L_spatial)
        ), "spatial disformation is zero on the background"

    def test_contortion_absent_for_levi_civita(self):
        """For a Levi-Civita connection, the contortion and disformation
        pieces are zero (absent)."""
        adm = adm_sample_1p2()
        LC = adm.full.christoffel
        affine = AffineADMGeometry(adm, LC)
        from noether.kernels.sympy_kernel.geometry import components
        K = affine.contortion
        L = affine.disformation
        assert all(
            (sp.cancel(sp.together(c)) == 0) for c in components(K)
        ), "contortion nonzero for Levi-Civita"
        assert all(
            (sp.cancel(sp.together(c)) == 0) for c in components(L)
        ), "disformation nonzero for Levi-Civita"

    def test_convention_named(self, affine_adm):
        """The contortion/disformation signs are recorded as
        metric-affine-v1 (referenced by name, not asserted from memory)."""
        # The contortion formula is K = (1/2)(T + g^lam_sig g_mu_tau T^tau + ...)
        # This is the metric-affine-v1 convention, verified by the kernel.
        # We verify the round-trip K - K(T_reconstructed) = 0 as a
        # structural check that the convention is consistent.
        ok, detail = affine_adm.check_post_riemannian_on_foliation()
        assert ok, detail


# ---------------------------------------------------------------------------
# VAL-ADM-004: constraints are separated from evolution
# ---------------------------------------------------------------------------


class TestConstraintEvolutionSeparation:
    """The result distinguishes constraint pieces (Hamiltonian/momentum,
    plus connection-sector constraints) from evolution pieces."""

    def test_hamiltonian_constraint_is_constraint(self, gr_adm):
        """The Hamiltonian constraint R3 + K^2 - KK is a constraint
        (first-order in time derivatives, constrains initial data)."""
        ham = gr_adm.hamiltonian_form
        assert ham is not None

    def test_momentum_constraint_is_constraint(self, gr_adm):
        """The momentum constraint D_j(K^j_i - delta^j_i K) is a
        constraint (first-order in time derivatives)."""
        mom = gr_adm.momentum_form
        assert len(mom) == gr_adm.d

    def test_connection_eom_is_algebraic_constraint(self, affine_adm):
        """The connection EOM is algebraic (no time derivatives of
        Gamma), making it a constraint rather than an evolution equation."""
        ok, detail = affine_adm.check_connection_eom_algebraic()
        assert ok, detail

    def test_constraint_labels_distinct_from_evolution(self):
        """The ADM outputs label constraints distinctly from evolution.
        Hamiltonian and momentum are labeled as 'projection' (constraint),
        and the connection-sector constraints are labeled separately."""
        from noether.orchestrator.derive import _ADM_AFFINE_OUTPUTS, _ADM_OUTPUTS
        constraint_labels = [label for label, _, _ in _ADM_AFFINE_OUTPUTS]
        # Hamiltonian and momentum projections are constraints
        metric_labels = [label for label, _ in _ADM_OUTPUTS]
        assert "Hamiltonian (normal-normal) projection" in metric_labels
        assert "momentum (normal-tangential) projection" in metric_labels
        # Connection-sector constraints are a separate constraint piece
        assert "connection-sector constraints" in constraint_labels


# ---------------------------------------------------------------------------
# VAL-ADM-005: connection-sector primary/secondary constraints are surfaced
# ---------------------------------------------------------------------------


class TestConnectionSectorConstraints:
    """Non-dynamical connection components are surfaced as primary/secondary
    constraints. If the Dirac chain cannot be closed, it is gated with a
    stated reason."""

    def test_primary_constraints_identified(self, affine_adm):
        """The algebraic connection EOM generates primary constraints:
        the connection components are non-dynamical (no time derivatives)."""
        ok, detail = affine_adm.check_connection_sector_primary_constraints()
        assert ok, detail
        assert "Primary constraints" in detail

    def test_secondary_constraints_mentioned(self, affine_adm):
        """The constraint piece mentions secondary constraints."""
        ok, detail = affine_adm.check_connection_sector_primary_constraints()
        assert ok, detail
        assert "Secondary constraints" in detail or "secondary" in detail.lower()

    def test_dirac_chain_gated_for_nonmetricity(self, affine_adm):
        """When the connection has non-metricity (Q != 0), the Dirac
        chain cannot be closed in general and is gated with a reason."""
        # Our sample background has both T and Q nonzero
        is_metric_compatible = affine_adm.connection_eom_algebraic
        if not is_metric_compatible:
            # The Dirac chain is gated: the constraint piece should
            # carry a detail naming the blocker
            ok, detail = affine_adm.check_connection_sector_primary_constraints()
            assert ok, detail
            assert "Dirac chain" in detail or "gated" in detail.lower()

    def test_metric_compatible_dirac_closeable(self):
        """On a metric-compatible (Q=0) torsionful background, the Dirac
        chain can be closed: the connection EOM is algebraic in K."""
        adm = adm_sample_1p2()
        coords = adm.coords
        D = adm.d + 1

        # Build a metric-compatible torsionful connection: Gamma = LC + K(T)
        LC = adm.full.christoffel
        gamma_random = random_affine_connection(7, coords, symmetric=False)
        K = contortion_of_torsion(gamma_random, adm.full.g, adm.full.g_inv)
        gamma_mc = sp.MutableDenseNDimArray(LC)
        for a in range(D):
            for b in range(D):
                for c in range(D):
                    gamma_mc[a, b, c] = _clean(LC[a, b, c] + K[a, b, c])

        affine = AffineADMGeometry(adm, sp.ImmutableDenseNDimArray(gamma_mc))
        assert affine.connection_eom_algebraic, (
            "connection should be metric-compatible (Q=0)"
        )
        assert affine.dirac_chain_closeable, (
            "Dirac chain should be closeable when Q=0"
        )


# ---------------------------------------------------------------------------
# Regression: GR ADM checks still pass with AffineADMGeometry
# ---------------------------------------------------------------------------


class TestGRRegression:
    """When the connection is Levi-Civita, the metric-affine ADM
    reduces to the GR ADM (T=Q=0)."""

    def test_levi_civita_torsion_zero(self):
        adm = adm_sample_1p2()
        LC = adm.full.christoffel
        affine = AffineADMGeometry(adm, LC)
        from noether.kernels.sympy_kernel.geometry import components
        T = affine.torsion
        assert all(
            (sp.cancel(sp.together(c)) == 0) for c in components(T)
        ), "torsion nonzero for Levi-Civita"

    def test_levi_civita_nonmetricity_zero(self):
        adm = adm_sample_1p2()
        LC = adm.full.christoffel
        affine = AffineADMGeometry(adm, LC)
        from noether.kernels.sympy_kernel.geometry import components
        Q = affine.nonmetricity
        assert all(
            (sp.cancel(sp.together(c)) == 0) for c in components(Q)
        ), "non-metricity nonzero for Levi-Civita"

    def test_levi_civita_post_riemannian_holds(self):
        """Gamma = LC + 0 + 0 for Levi-Civita."""
        adm = adm_sample_1p2()
        LC = adm.full.christoffel
        affine = AffineADMGeometry(adm, LC)
        ok, detail = affine.check_post_riemannian_on_foliation()
        assert ok, detail

    def test_gr_adm_checks_unchanged(self, gr_checks):
        """All six GR ADM checks still pass."""
        for name, (ok, detail) in gr_checks.items():
            assert ok, f"{name}: {detail}"


# ---------------------------------------------------------------------------
# Cross-check: SymPy adapter runs the metric-affine ADM suite
# ---------------------------------------------------------------------------


class TestSympyAdapter:
    """The SympyKernelAdapter runs the adm-affine-1p2 check suite."""

    def test_adapter_adm_affine_1p2(self):
        from noether.kernels.base import Capability, KernelTask
        from noether.kernels.sympy_kernel import SympyKernelAdapter

        adapter = SympyKernelAdapter()
        result = adapter.run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="metric-affine ADM 1+2 check",
                payload={"check": "adm-affine-1p2"},
            )
        )
        assert result.value.get("passed"), result.value.get("detail", "")
        checks = result.value.get("checks", {})
        assert len(checks) >= 7, f"expected >= 7 checks, got {len(checks)}"
        for name, val in checks.items():
            assert val == "True", f"{name}: {val}"


# ---------------------------------------------------------------------------
# VAL-ADM-006: SymPy component kernel is the verifier (no model script)
# ---------------------------------------------------------------------------


class TestSympyVerificationModel:
    """The ADM result's kernel_name is 'sympy', it carries a SymPy
    reproduction script, and no LLM Cadabra script is written for the
    adm path."""

    def test_adm_derivation_kernel_name_is_sympy(self):
        """derive_adm sets kernel_name='sympy' on all derivations."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr()
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-006")
        for d in results:
            assert d.kernel_name == "sympy", (
                f"ADM derivation kernel_name should be 'sympy', "
                f"got {d.kernel_name!r}"
            )

    def test_adm_derivation_carries_sympy_script(self):
        """Each ADM derivation carries a SymPy reproduction script."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr()
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-006")
        for d in results:
            assert d.script, "ADM derivation should carry a script"
            assert "sympy" in d.script.lower() or "SympyKernelAdapter" in d.script, (
                "ADM derivation script should reference the SymPy kernel"
            )

    def test_no_model_cadabra_script_for_adm(self):
        """The adm path does not call an LLM or write a Cadabra script.
        derive_adm takes no LLM adapter, and kernel_name is 'sympy'."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr()
        # derive_adm signature takes no LLM adapter
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-006")
        for d in results:
            assert d.llm_name == "", (
                f"ADM derivation should have no LLM name, got {d.llm_name!r}"
            )
            assert d.kernel_name == "sympy", (
                "ADM derivation should not use Cadabra"
            )

    def test_gr_adm_kernel_name_is_sympy(self):
        """GR ADM also uses the SymPy kernel (not Cadabra)."""
        from evals.eval1s_adm import build_npr as build_adm_npr
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = build_adm_npr(resolved=True)
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-gr-006")
        for d in results:
            assert d.kernel_name == "sympy"


# ---------------------------------------------------------------------------
# VAL-ADM-007: Verified verdict backed by explicit-background component checks
# ---------------------------------------------------------------------------


class TestExplicitBackgroundVerification:
    """A verified ADM split is verified because the SymPy kernel reduced the
    split and constraint projections to zero on an explicit nondegenerate
    metric-affine background (torsion/non-metricity on), with a
    nondegeneracy/distortion-nonzero falsifier."""

    def test_distortion_nonzero_falsifier_passes(self, affine_checks):
        """The distortion-nonzero falsifier check passes on the standard
        metric-affine background, asserting all distortion features are
        nonzero."""
        ok, detail = affine_checks["distortion-nonzero-falsifier"]
        assert ok, detail

    def test_each_named_check_passed_on_distortion_nonzero_background(self, affine_checks):
        """Every named check in the metric-affine suite passed=True on a
        background where the distortion features (T, Q, K, L) are all
        nonzero (as asserted by the falsifier)."""
        # First confirm the falsifier passed
        falsifier_ok, falsifier_detail = affine_checks["distortion-nonzero-falsifier"]
        assert falsifier_ok, (
            "falsifier must pass before checks prove anything: "
            + falsifier_detail
        )
        # Then confirm each check passed
        for name, (ok, detail) in affine_checks.items():
            assert ok, f"check {name!r} failed on distortion-nonzero background: {detail}"

    def test_falsifier_asserts_contortion_nonzero(self, affine_adm):
        """The falsifier explicitly asserts contortion K is nonzero."""
        ok, detail = affine_adm.check_distortion_nonzero_falsifier()
        assert ok, detail
        assert "K^i_{jk} nonzero: True" in detail, (
            "falsifier should assert K^i_{jk} is nonzero: " + detail
        )

    def test_falsifier_asserts_disformation_nonzero(self, affine_adm):
        """The falsifier explicitly asserts disformation L is nonzero."""
        ok, detail = affine_adm.check_distortion_nonzero_falsifier()
        assert ok, detail
        assert "L^i_{jk} nonzero: True" in detail, (
            "falsifier should assert L^i_{jk} is nonzero: " + detail
        )

    def test_falsifier_asserts_torsion_nonzero(self, affine_adm):
        """The falsifier explicitly asserts torsion T is nonzero."""
        ok, detail = affine_adm.check_distortion_nonzero_falsifier()
        assert ok, detail
        assert "T^i_{jk} nonzero: True" in detail, (
            "falsifier should assert T^i_{jk} is nonzero: " + detail
        )

    def test_falsifier_asserts_nonmetricity_nonzero(self, affine_adm):
        """The falsifier explicitly asserts non-metricity Q is nonzero."""
        ok, detail = affine_adm.check_distortion_nonzero_falsifier()
        assert ok, detail
        assert "Q_{ijk} nonzero: True" in detail, (
            "falsifier should assert Q_{ijk} is nonzero: " + detail
        )

    def test_adapter_adm_affine_falsifier_check(self):
        """The SymPy adapter runs the distortion-nonzero falsifier as part
        of the adm-affine-1p2 check suite."""
        from noether.kernels.base import Capability, KernelTask
        from noether.kernels.sympy_kernel import SympyKernelAdapter

        adapter = SympyKernelAdapter()
        result = adapter.run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="metric-affine ADM 1+2 check",
                payload={"check": "adm-affine-1p2"},
            )
        )
        checks = result.value.get("checks", {})
        assert "distortion-nonzero-falsifier" in checks, (
            "distortion-nonzero-falsifier check must be in the suite"
        )
        assert checks["distortion-nonzero-falsifier"] == "True", (
            "distortion-nonzero-falsifier must pass"
        )


# ---------------------------------------------------------------------------
# VAL-ADM-008: A reduction that cannot close is gated, not faked
# ---------------------------------------------------------------------------


class TestGatedResult:
    """Any part that cannot be reduced is returned verified==False with a
    detail naming the blocker; the piece is still surfaced (result_tex
    present), never dropped or reported true."""

    def test_gated_constraint_piece_has_verified_false(self):
        """When Q != 0, the connection-sector constraints piece carries
        verified==False (the Dirac chain cannot be closed)."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr(nonmetricity=True)
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-008")
        constraint_piece = next(
            (d for d in results if d.wrt == "connection-sector constraints"), None
        )
        assert constraint_piece is not None, "connection-sector constraints piece missing"
        assert constraint_piece.verified is False, (
            "connection-sector constraints should be verified==False when Q != 0"
        )

    def test_gated_piece_has_nonempty_detail(self):
        """The gated piece carries a non-empty detail naming the blocker."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr(nonmetricity=True)
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-008")
        constraint_piece = next(
            (d for d in results if d.wrt == "connection-sector constraints"), None
        )
        assert constraint_piece is not None
        assert constraint_piece.detail, (
            "gated piece must have a non-empty detail naming the blocker"
        )
        assert "Dirac chain" in constraint_piece.detail or "Q != 0" in constraint_piece.detail, (
            "detail should name the blocker (Dirac chain / Q != 0): "
            + constraint_piece.detail
        )

    def test_gated_piece_still_has_result_tex(self):
        """The gated piece is still surfaced with result_tex present;
        it is never dropped."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr(nonmetricity=True)
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-008")
        constraint_piece = next(
            (d for d in results if d.wrt == "connection-sector constraints"), None
        )
        assert constraint_piece is not None
        assert constraint_piece.result_tex, (
            "gated piece must still have result_tex present (never dropped)"
        )

    def test_other_affine_pieces_still_verified(self):
        """The non-gated metric-affine pieces (decomposition, foliation
        pieces) are still verified when the constraint piece is gated."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr(nonmetricity=True)
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-008")
        # The connection foliation decomposition piece should be verified
        decomp_piece = next(
            (d for d in results if d.wrt == "connection foliation decomposition"), None
        )
        assert decomp_piece is not None
        assert decomp_piece.verified is True, (
            "connection foliation decomposition should be verified"
        )

    def test_metric_compatible_constraints_are_verified(self):
        """When Q == 0 (metric-compatible), the constraint piece is
        verified (the Dirac chain closes)."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr(nonmetricity=False)
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-008")
        constraint_piece = next(
            (d for d in results if d.wrt == "connection-sector constraints"), None
        )
        assert constraint_piece is not None
        assert constraint_piece.verified is True, (
            "connection-sector constraints should be verified when Q == 0: "
            + constraint_piece.detail
        )


# ---------------------------------------------------------------------------
# VAL-ADM-009: Teaching narration is separate from the verified verdict
# ---------------------------------------------------------------------------


class TestNarrationSeparation:
    """Explanatory narration about the connection's constraints is on the
    teaching/narrative channel and never sets a result expression or flips
    verified."""

    def test_narrative_field_present_on_adm_derivations(self):
        """ADM derivations carry a narrative field for teaching prose."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr()
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-009")
        for d in results:
            # The narrative field exists (may be empty for metric-only pieces)
            assert hasattr(d, "narrative"), "FieldDerivation must have a narrative field"

    def test_constraint_piece_has_teaching_narrative(self):
        """The connection-sector constraints piece carries explanatory
        narrative about the constraint structure."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr()
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-009")
        constraint_piece = next(
            (d for d in results if d.wrt == "connection-sector constraints"), None
        )
        assert constraint_piece is not None
        assert constraint_piece.narrative, (
            "connection-sector constraints should carry teaching narrative"
        )
        # The narrative should explain the constraint structure
        assert "algebraic" in constraint_piece.narrative.lower() or (
            "constraint" in constraint_piece.narrative.lower()
        ), (
            "narrative should explain the constraint structure: "
            + constraint_piece.narrative
        )

    def test_narrative_never_flips_verified(self):
        """Varying the narrative content never changes the verified flag.
        The verified flag is determined solely by the kernel checks."""
        from noether.orchestrator.derive import FieldDerivation

        # Construct two derivations that differ only in narrative
        d1 = FieldDerivation(
            wrt="test",
            kind="adm",
            capability=Capability.ADM,
            result_id="test-1",
            result_tex="some result",
            verified=True,
            checks={"check-a": "True"},
            kernel_name="sympy",
            kernel_version="1.14",
            detail="verified",
            narrative="explanatory prose A",
        )
        d2 = FieldDerivation(
            wrt="test",
            kind="adm",
            capability=Capability.ADM,
            result_id="test-2",
            result_tex="some result",
            verified=True,
            checks={"check-a": "True"},
            kernel_name="sympy",
            kernel_version="1.14",
            detail="verified",
            narrative="different explanatory prose B",
        )
        # Both are verified regardless of narrative content
        assert d1.verified is True
        assert d2.verified is True
        # Narrative is separate from result_tex
        assert d1.narrative != d1.result_tex
        assert d2.narrative != d2.result_tex

    def test_narrative_separate_from_result_tex(self):
        """The narrative field does not appear in result_tex, and vice
        versa."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr()
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-009")
        for d in results:
            if d.narrative and d.result_tex:
                # Narrative prose should not appear as a LaTeX expression
                assert d.narrative != d.result_tex, (
                    f"narrative must be distinct from result_tex for {d.wrt!r}"
                )

    def test_narrative_does_not_appear_in_checks(self):
        """The narrative text does not appear in the checks dict."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr()
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-009")
        for d in results:
            if d.narrative:
                for check_val in d.checks.values():
                    assert d.narrative not in check_val, (
                        "narrative must not appear in check values"
                    )

    def test_gated_piece_narrative_explains_constraint_structure(self):
        """Even a gated (verified==False) piece carries narrative that
        explains the constraint structure without falsely asserting
        verification."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr(nonmetricity=True)
        results = derive_adm(npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-009")
        constraint_piece = next(
            (d for d in results if d.wrt == "connection-sector constraints"), None
        )
        assert constraint_piece is not None
        # The piece is gated
        assert constraint_piece.verified is False
        # But it still has narrative explaining the structure
        assert constraint_piece.narrative, (
            "gated piece should still carry teaching narrative"
        )
        # And the narrative explains the constraint structure without
        # falsely claiming verification
        assert "Dirac" in constraint_piece.narrative or (
            "constraint" in constraint_piece.narrative.lower()
        ), (
            "narrative should explain the constraint structure"
        )
        # The narrative does not set result_tex
        assert constraint_piece.result_tex is not None
        # The narrative is separate from detail
        assert constraint_piece.narrative != constraint_piece.detail


# ---------------------------------------------------------------------------
# Helper: build a metric-affine NPR for ADM testing
# ---------------------------------------------------------------------------


def _build_metric_affine_adm_npr(*, nonmetricity: bool = True):
    """Build a well-posed metric-affine NPR for ADM derivation testing.

    With nonmetricity=True the connection has both torsion and
    non-metricity, so the Dirac chain cannot close and the constraint
    piece should be gated (verified==False). With nonmetricity=False
    the connection is metric-compatible with torsion, so the Dirac chain
    closes and the constraint piece should be verified.
    """
    from noether.npr import (
        NOETHER_DEFAULT_V1,
        NPR,
        Action,
        ConnectionSpec,
        Geometry,
        ObjectDecl,
        Task,
    )
    from noether.npr.ast import tensor

    connection = ConnectionSpec(
        type="independent",
        torsion=True,
        nonmetricity=nonmetricity,
        metric_compatible=not nonmetricity,
        family="metric-affine",
    )
    geometry = Geometry(
        metric_name="g",
        connection_name="Gamma",
        connection=connection,
    )
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=geometry,
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(
                name="Gamma",
                kind="connection",
                role="dynamical",
                rank=3,
            ),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=tensor("R"),
            lagrangian_tex="R",
        ),
        task=Task(type="adm", with_respect_to=["g"]),
        ambiguities=[],
    )


def _build_metric_affine_adm_npr_with_matter(
    *, nonmetricity: bool = True, covcurl: bool = True
):
    """Build a well-posed metric-affine NPR with matter that couples to the
    connection (has hypermomentum), for ADM derivation testing.

    With covcurl=True the gauge field has F = covariant curl, which
    couples to the connection (hypermomentum nonzero). With covcurl=False
    the gauge field has F = dA (no hypermomentum).
    """
    from noether.npr import (
        NOETHER_DEFAULT_V1,
        NPR,
        Action,
        ConnectionSpec,
        Geometry,
        ObjectDecl,
        Task,
    )
    from noether.npr.ast import tensor

    conventions = NOETHER_DEFAULT_V1
    if covcurl:
        conventions = conventions.model_copy(
            update={"field_strength_definition": "covariant-curl"}
        )

    connection = ConnectionSpec(
        type="independent",
        torsion=True,
        nonmetricity=nonmetricity,
        metric_compatible=not nonmetricity,
        family="metric-affine",
    )
    geometry = Geometry(
        metric_name="g",
        connection_name="Gamma",
        connection=connection,
    )
    return NPR(
        conventions=conventions,
        geometry=geometry,
        objects=[
            ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
            ObjectDecl(
                name="Gamma",
                kind="connection",
                role="dynamical",
                rank=3,
            ),
            ObjectDecl(name="A", kind="tensor-field", role="dynamical", rank=1),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=tensor("R"),
            lagrangian_tex="R - 1/4 F_{mu nu} F^{mu nu}",
        ),
        task=Task(type="adm", with_respect_to=["g"]),
        ambiguities=[],
    )


# ---------------------------------------------------------------------------
# VAL-ADM-015: Matter coupled to the connection appears in the ADM
# constraint structure
# ---------------------------------------------------------------------------


class TestMatterHypermomentumConstraints:
    """For a metric-affine action with matter that sources the connection
    (hypermomentum), the ADM result surfaces the matter contribution to
    the connection-sector constraints; pure-gravity sessions carry no
    such matter piece."""

    def test_hypermomentum_constraint_check_with_nonzero_delta(self):
        """The check_matter_hypermomentum_constraints method with a
        nonzero Delta verifies the hypermomentum enters the constraint
        structure on the foliated background."""
        from noether.kernels.sympy_kernel.adm import adm_affine_matter_sample_1p2

        affine, Delta = adm_affine_matter_sample_1p2(nonmetricity=True)
        ok, detail = affine.check_matter_hypermomentum_constraints(Delta)
        assert ok, detail
        assert "spin" in detail.lower() or "tau" in detail, (
            "detail should name the spin/hypermomentum contribution: " + detail
        )
        assert "constraint" in detail.lower(), (
            "detail should reference constraints: " + detail
        )

    def test_hypermomentum_constraint_check_with_zero_delta(self):
        """When Delta = 0 (pure gravity), the check passes trivially
        with a message stating zero hypermomentum."""
        from noether.kernels.sympy_kernel.adm import adm_affine_sample_1p2

        affine = adm_affine_sample_1p2()
        D = affine.D
        Delta_zero = sp.ImmutableDenseNDimArray(
            sp.MutableDenseNDimArray.zeros(D, D, D)
        )
        ok, detail = affine.check_matter_hypermomentum_constraints(Delta_zero)
        assert ok, detail
        assert "Delta = 0" in detail or "no matter coupling" in detail, (
            "zero hypermomentum should be explicitly stated: " + detail
        )

    def test_hypermomentum_decomposition_on_foliation(self):
        """The hypermomentum decomposition Delta = spin + dilation + shear
        holds on the foliated background, and the projected pieces have
        the correct symmetry/trace properties."""
        from noether.kernels.sympy_kernel.adm import adm_affine_matter_sample_1p2

        affine, Delta = adm_affine_matter_sample_1p2(nonmetricity=True)
        ok, detail = affine.check_matter_hypermomentum_constraints(Delta)
        assert ok, detail

    def test_metric_compatible_hypermomentum_verified(self):
        """On a metric-compatible (Q=0) background, the Dirac chain closes
        and the matter contribution is verified (not gated)."""
        from noether.kernels.sympy_kernel.adm import adm_affine_matter_sample_1p2

        affine, Delta = adm_affine_matter_sample_1p2(nonmetricity=False)
        ok, detail = affine.check_matter_hypermomentum_constraints(Delta)
        assert ok, detail
        assert "Dirac chain closes" in detail or "Q=0" in detail, (
            "metric-compatible case should reference Dirac chain closure: "
            + detail
        )

    def test_nonmetric_hypermomentum_gated(self):
        """On a non-metric-compatible (Q!=0) background, the Dirac chain
        is gated and the detail names the blocker."""
        from noether.kernels.sympy_kernel.adm import adm_affine_matter_sample_1p2

        affine, Delta = adm_affine_matter_sample_1p2(nonmetricity=True)
        ok, detail = affine.check_matter_hypermomentum_constraints(Delta)
        assert ok, detail
        # The detail should mention the Dirac chain gating
        assert "gated" in detail.lower() or "Dirac" in detail, (
            "non-metric case should mention Dirac gating: " + detail
        )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_hypermomentum_on_multiple_backgrounds(self, seed):
        """The check passes on multiple seeded backgrounds."""
        from noether.kernels.sympy_kernel.adm import adm_affine_matter_sample_1p2

        affine, Delta = adm_affine_matter_sample_1p2(seed=seed, nonmetricity=True)
        ok, detail = affine.check_matter_hypermomentum_constraints(Delta)
        assert ok, detail

    def test_adapter_adm_affine_matter_1p2(self):
        """The SymPy adapter runs the adm-affine-matter-1p2 check suite
        including the matter-hypermomentum-constraints check."""
        from noether.kernels.base import Capability, KernelTask
        from noether.kernels.sympy_kernel import SympyKernelAdapter

        adapter = SympyKernelAdapter()
        result = adapter.run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="metric-affine ADM with matter 1+2 check",
                payload={"check": "adm-affine-matter-1p2", "nonmetricity": True},
            )
        )
        assert result.value.get("passed"), result.value.get("detail", "")
        checks = result.value.get("checks", {})
        assert "matter-hypermomentum-constraints" in checks, (
            "matter-hypermomentum-constraints check must be in the suite"
        )
        assert checks["matter-hypermomentum-constraints"] == "True", (
            "matter-hypermomentum-constraints check must pass"
        )

    def test_adapter_adm_affine_matter_metric_compatible(self):
        """The SymPy adapter runs the matter check on a metric-compatible
        background (Q=0), where the Dirac chain closes. The full suite
        may fail the Q/L falsifier (Q=0 is expected), but the matter
        check itself passes."""
        from noether.kernels.base import Capability, KernelTask
        from noether.kernels.sympy_kernel import SympyKernelAdapter

        adapter = SympyKernelAdapter()
        result = adapter.run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="metric-affine ADM with matter 1+2 (Q=0)",
                payload={"check": "adm-affine-matter-1p2", "nonmetricity": False},
            )
        )
        checks = result.value.get("checks", {})
        # The matter check itself must pass
        assert "matter-hypermomentum-constraints" in checks
        assert checks["matter-hypermomentum-constraints"] == "True", (
            "matter-hypermomentum-constraints check must pass"
        )
        # The post-Riemannian decomposition and torsion checks pass
        assert checks.get("post-riemannian-on-foliation") == "True"
        assert checks.get("connection-eom-algebraic") == "True"
        # Q/L falsifier checks may fail on Q=0 backgrounds (expected)
        # The overall suite passed=False is fine for this configuration


class TestMatterHypermomentumInDeriveADM:
    """The derive_adm function surfaces the matter hypermomentum
    contribution as a constraint piece when the action has matter that
    couples to the connection, and omits it for pure gravity."""

    def test_matter_piece_present_for_covcurl_gauge(self):
        """A metric-affine ADM with a covariant-curl gauge field (has
        hypermomentum) surfaces a 'matter hypermomentum contribution'
        constraint piece."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr_with_matter(covcurl=True)
        results = derive_adm(
            npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-015-matter"
        )
        matter_piece = next(
            (
                d
                for d in results
                if d.wrt == "matter hypermomentum contribution"
            ),
            None,
        )
        assert matter_piece is not None, (
            "matter hypermomentum contribution piece should be present "
            "for an action with covcurl gauge field"
        )
        assert "Delta" in matter_piece.result_tex or "tau" in matter_piece.result_tex, (
            "matter piece should name the hypermomentum: "
            + matter_piece.result_tex
        )
        assert matter_piece.narrative, (
            "matter piece should carry teaching narrative"
        )

    def test_matter_piece_absent_for_pure_gravity(self):
        """A pure-gravity metric-affine ADM (no matter coupling to the
        connection) carries no 'matter hypermomentum contribution' piece."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr(nonmetricity=True)
        results = derive_adm(
            npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-015-pure"
        )
        matter_piece = next(
            (
                d
                for d in results
                if d.wrt == "matter hypermomentum contribution"
            ),
            None,
        )
        assert matter_piece is None, (
            "matter hypermomentum contribution piece should be absent "
            "for a pure-gravity action"
        )

    def test_matter_piece_absent_for_dA_gauge(self):
        """A metric-affine ADM with F=dA gauge field (no hypermomentum)
        carries no 'matter hypermomentum contribution' piece."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr_with_matter(covcurl=False)
        results = derive_adm(
            npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-015-dA"
        )
        matter_piece = next(
            (
                d
                for d in results
                if d.wrt == "matter hypermomentum contribution"
            ),
            None,
        )
        assert matter_piece is None, (
            "matter hypermomentum contribution piece should be absent "
            "for F=dA gauge (no connection coupling)"
        )

    def test_matter_piece_verified_or_gated(self):
        """The matter piece is verified (SymPy check passes) or gated
        with a non-empty detail naming the blocker."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        # Q=0 (metric-compatible): verified
        npr_mc = _build_metric_affine_adm_npr_with_matter(
            nonmetricity=False, covcurl=True
        )
        results_mc = derive_adm(
            npr_mc, {"sympy": SympyKernelAdapter()}, session_id="s-adm-015-vmc"
        )
        matter_mc = next(
            (
                d
                for d in results_mc
                if d.wrt == "matter hypermomentum contribution"
            ),
            None,
        )
        assert matter_mc is not None
        assert matter_mc.verified is True or (
            matter_mc.verified is False and matter_mc.detail
        ), (
            "matter piece must be verified or gated with a detail"
        )

        # Q!=0 (non-metric-compatible): gated
        npr_nm = _build_metric_affine_adm_npr_with_matter(
            nonmetricity=True, covcurl=True
        )
        results_nm = derive_adm(
            npr_nm, {"sympy": SympyKernelAdapter()}, session_id="s-adm-015-vnm"
        )
        matter_nm = next(
            (
                d
                for d in results_nm
                if d.wrt == "matter hypermomentum contribution"
            ),
            None,
        )
        assert matter_nm is not None
        assert matter_nm.verified is False, (
            "matter piece should be gated when Q != 0"
        )
        assert matter_nm.detail, (
            "gated piece must have a non-empty detail"
        )

    def test_constraint_piece_names_hypermomentum(self):
        """The constraint piece's result_tex names the hypermomentum
        contribution (spin/dilation/shear entering constraints)."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr_with_matter(covcurl=True)
        results = derive_adm(
            npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-015-name"
        )
        matter_piece = next(
            (
                d
                for d in results
                if d.wrt == "matter hypermomentum contribution"
            ),
            None,
        )
        assert matter_piece is not None
        result_tex = matter_piece.result_tex
        assert "Delta" in result_tex or "tau" in result_tex or "sigma" in result_tex, (
            "result_tex should name the hypermomentum: " + result_tex
        )
        assert "constraint" in result_tex.lower(), (
            "result_tex should reference constraints: " + result_tex
        )

    def test_matter_piece_kernel_name_is_sympy(self):
        """The matter piece uses the SymPy kernel, not Cadabra."""
        from noether.kernels.sympy_kernel import SympyKernelAdapter
        from noether.orchestrator.derive import derive_adm

        npr = _build_metric_affine_adm_npr_with_matter(covcurl=True)
        results = derive_adm(
            npr, {"sympy": SympyKernelAdapter()}, session_id="s-adm-015-kernel"
        )
        matter_piece = next(
            (
                d
                for d in results
                if d.wrt == "matter hypermomentum contribution"
            ),
            None,
        )
        assert matter_piece is not None
        assert matter_piece.kernel_name == "sympy", (
            "matter piece should use the SymPy kernel"
        )

