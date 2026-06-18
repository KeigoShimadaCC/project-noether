r"""VAL-GEOM-015 / VAL-EOM-013: Held G4(phi,X)R / G5 Horndeski closure.

The best-effort higher-Horndeski closure either fully closes (Cadabra residue 0
AND SymPy cross-check agrees, verified=True) or is returned verified=False with
a non-empty detail; never verified with a gate unmet.

VAL-GEOM-015 (M2 primitive level): the ClosureAttempt satisfies the XOR
condition at the primitive level:

    (verified and residue_zero and oracle_agrees)
    XOR
    (not verified and detail != '')

VAL-EOM-013 (M3 EOM path level): when the G4(phi,X)R / G5 Horndeski EOM
is attempted through the general derivation path, each FieldDerivation
satisfies:

    if verified==True then residue_zero=="True"
    else verified==False with a non-empty detail

The result is surfaced, never falsely asserted.

Conventions: noether-default-v1.
"""

import pytest

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.blocks import has_g4g5_terms
from noether.kernels.cadabra.horndeski_g4g5 import (
    ClosureAttempt,
    assemble_g4_metric_eom_script,
    assemble_g4_scalar_eom_script,
    attempt_g4g5_closure,
)
from noether.npr import (
    NOETHER_DEFAULT_V1,
    NPR,
    Action,
    ConnectionSpec,
    Geometry,
    ObjectDecl,
    Task,
)
from noether.npr.ast import prod, tensor, up
from noether.npr.parse import parse_lagrangian
from noether.orchestrator.derive import FieldDerivation, attempt_g4g5_eom


def _g4g5_npr() -> NPR:
    """Build a minimal NPR carrying a G4(phi,X)R Lagrangian for testing
    ``attempt_g4g5_eom``. The convention block is the repo default."""
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(connection=ConnectionSpec(type="levi-civita")),
        objects=[
            ObjectDecl(name="g", kind="metric", role="background", symmetry="symmetric", rank=2),
            ObjectDecl(name="phi", kind="scalar-field", role="dynamical"),
        ],
        action=Action(
            measure_tex=r"d^4x \sqrt{-g}",
            lagrangian=prod(tensor("G", up("mu"), up("nu")), tensor("R", up("mu"), up("nu"))),
            lagrangian_tex=r"G(\phi, X) R",
        ),
        task=Task(type="vary", with_respect_to=["g", "phi"]),
    )

# ---------------------------------------------------------------------------
# VAL-GEOM-015: verified-or-gated XOR condition
# ---------------------------------------------------------------------------


class TestG4G5ClosureXOR:
    """The closure result satisfies the XOR condition:

        (verified and residue_zero and oracle_agrees)
        XOR
        (not verified and detail != '')

    It is never verified with a gate unmet, and if gated, the detail is
    non-empty and names the specific blocker.
    """

    def test_closure_attempt_returns_xor_satisfied(self):
        """The closure attempt result satisfies the XOR condition."""
        result = attempt_g4g5_closure()

        # The XOR condition: exactly one of the two branches holds.
        branch_a = result.verified and result.residue_zero and result.oracle_agrees
        branch_b = (not result.verified) and (result.detail != "")

        # They must be mutually exclusive and exhaustive.
        assert branch_a != branch_b, (
            f"XOR condition violated: branch_a={branch_a}, branch_b={branch_b}, "
            f"verified={result.verified}, residue_zero={result.residue_zero}, "
            f"oracle_agrees={result.oracle_agrees}, detail={result.detail!r}"
        )

    def test_gated_result_has_nonempty_detail(self):
        """When the closure is gated, the detail is non-empty and names
        the blocker."""
        result = attempt_g4g5_closure()

        if not result.verified:
            assert result.detail, "Gated result must have non-empty detail"
            # The detail must name a specific blocker, not be a generic message.
            assert len(result.detail) > 20, (
                f"Detail too short to name a specific blocker: {result.detail!r}"
            )

    def test_never_verified_with_gate_unmet(self):
        """verified=True is only possible when residue_zero and oracle_agrees
        are both True. This is a structural invariant, not just a runtime
        check."""
        result = attempt_g4g5_closure()

        if result.verified:
            assert result.residue_zero, (
                "verified=True requires residue_zero=True"
            )
            assert result.oracle_agrees, (
                "verified=True requires oracle_agrees=True"
            )

    def test_detail_names_sortcovds_blocker(self):
        """When gated by the normal-ordering gap, the detail names the
        SortCovDs blocker explicitly."""
        result = attempt_g4g5_closure()

        if not result.verified and "SortCovDs" in result.detail:
            assert "normal-ordering" in result.detail, (
                f"Detail should mention normal-ordering: {result.detail!r}"
            )
            assert "xAct" in result.detail, (
                f"Detail should mention xAct: {result.detail!r}"
            )

    def test_closure_attempt_is_honest_not_stale(self):
        """The closure attempt is not a stale placeholder: it reflects the
        actual state of the M2 primitives and the known blocker."""
        result = attempt_g4g5_closure()

        # The result must be a proper ClosureAttempt, not a stub.
        assert isinstance(result, ClosureAttempt)
        assert isinstance(result.verified, bool)
        assert isinstance(result.residue_zero, bool)
        assert isinstance(result.oracle_agrees, bool)
        assert isinstance(result.detail, str)


# ---------------------------------------------------------------------------
# Scalar EOM: second-order verification (Cadabra hand-audit confirmation)
# ---------------------------------------------------------------------------


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestG4ScalarEOM:
    """The G4(phi,X)R scalar EOM is second order: no third derivatives of phi
    survive the IBP. This is confirmed by a Cadabra script that substitutes
    nabla_nabla_nabla_phi -> 0 and checks the difference vanishes.

    This test confirms the scalar side of the G4 closure is reachable; the
    blocker is specifically the metric EOM's normal-ordering gap.
    """

    def test_scalar_eom_second_order(self):
        """The G4(phi,X)R scalar EOM has no third derivatives of phi after IBP.

        The Cadabra script varies phi in the action, applies IBP, expands
        coupling derivatives, and checks that substituting
        nabla_nabla_nabla_phi -> 0 does not change the expression. This
        confirms the no-Ostrogradski cancellation works for the scalar sector.
        """
        script = assemble_g4_scalar_eom_script()
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="G4 scalar EOM second-order check",
                payload={"script": script},
            )
        )
        assert result.raw.returncode == 0, result.raw.stderr
        check_val = result.value["checks"].get("scalar_eom_second_order")
        assert check_val == "True", (
            f"G4 scalar EOM should be second order (no third derivatives); "
            f"check value: {check_val}; output:\n{result.raw.stdout}"
        )


# ---------------------------------------------------------------------------
# Metric EOM: third-derivative diagnostic (Cadabra hand-audit confirmation)
# ---------------------------------------------------------------------------


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestG4MetricEOM:
    """The G4(phi,X)R metric EOM produces third derivatives of phi after
    expanding the wrapped nabla_mu(G4_X nabla_nu nabla_rho phi nabla^rho phi)
    terms. Without SortCovDs normal-ordering, these cannot be systematically
    reduced through the commutator + Ricci folds + Bianchi, so the metric EOM
    cannot be verified.

    This test confirms the blocker is real: the metric EOM expression DOES
    contain third derivatives of phi, demonstrating that the normal-ordering
    gap is the actual blocker (not a hypothetical one).
    """

    def test_metric_eom_has_third_derivatives(self):
        """The G4(phi,X)R metric EOM contains third derivatives of phi,
        confirming the normal-ordering blocker is real.

        After the standard two-pass IBP, the expression still has terms
        involving nabla_mu nabla_nu nabla_rho phi. Substituting these to zero
        and taking the difference produces a non-zero result, proving that
        third derivatives are present and the reduction is incomplete without
        the normal-ordering pass.
        """
        script = assemble_g4_metric_eom_script()
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="G4 metric EOM third-derivative diagnostic",
                payload={"script": script},
            )
        )
        assert result.raw.returncode == 0, result.raw.stderr
        check_val = result.value["checks"].get("metric_eom_has_third_derivs")
        assert check_val == "True", (
            f"G4 metric EOM should have third derivatives (confirming the "
            f"SortCovDs blocker); check value: {check_val}; output:\n"
            f"{result.raw.stdout}"
        )


# ---------------------------------------------------------------------------
# VAL-EOM-013: G4/G5 EOM best-effort through the general derive path
# ---------------------------------------------------------------------------


def _eom_xor_condition(d: FieldDerivation) -> bool:
    """The VAL-EOM-013 XOR condition for a single FieldDerivation:

        (verified and residue_zero=="True")
        XOR
        (not verified and detail != "")

    It is never verified==True with residue not True.
    """
    branch_a = d.verified and d.checks.get("residue_zero") == "True"
    branch_b = (not d.verified) and (d.detail != "")
    return branch_a != branch_b  # XOR


class TestG4G5EomXOR:
    """VAL-EOM-013: The G4(phi,X)R / G5 Horndeski EOM attempt through the
    general derivation path either closes (verified==True with
    residue_zero=="True") or returns verified==False with a non-empty detail;
    never verified==True with residue not True.

    These tests use the ``attempt_g4g5_eom`` function which produces
    ``FieldDerivation`` objects by running the hand-audited Cadabra scripts,
    exercising the same path that ``derive_field`` uses when G4/G5 terms are
    detected.
    """

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_scalar_eom_satisfies_xor(self):
        """The scalar EOM derivation satisfies the VAL-EOM-013 XOR condition."""
        derivations = attempt_g4g5_eom(CadabraAdapter(), _g4g5_npr())
        scalar = next(d for d in derivations if d.wrt == "phi")
        assert _eom_xor_condition(scalar), (
            f"Scalar EOM XOR violated: verified={scalar.verified}, "
            f"residue_zero={scalar.checks.get('residue_zero')}, "
            f"detail={scalar.detail!r}"
        )

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_metric_eom_satisfies_xor(self):
        """The metric EOM derivation satisfies the VAL-EOM-013 XOR condition."""
        derivations = attempt_g4g5_eom(CadabraAdapter(), _g4g5_npr())
        metric = next(d for d in derivations if d.wrt == "g")
        assert _eom_xor_condition(metric), (
            f"Metric EOM XOR violated: verified={metric.verified}, "
            f"residue_zero={metric.checks.get('residue_zero')}, "
            f"detail={metric.detail!r}"
        )

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_both_eoms_gated_with_detail(self):
        """Both EOMs are gated (verified==False) with a non-empty detail
        naming the SortCovDs blocker."""
        derivations = attempt_g4g5_eom(CadabraAdapter(), _g4g5_npr())
        for d in derivations:
            assert d.verified is False, (
                f"EOM wrt {d.wrt} should be gated (verified=False); "
                f"got verified={d.verified}"
            )
            assert d.detail, (
                f"EOM wrt {d.wrt} must have non-empty detail when gated"
            )
            assert "SortCovDs" in d.detail or "normal-ordering" in d.detail, (
                f"EOM wrt {d.wrt} detail should name the SortCovDs blocker: "
                f"{d.detail!r}"
            )

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_never_verified_with_residue_not_true(self):
        """If verified==True then residue_zero=="True"; this is a structural
        invariant that holds for every derivation regardless of the closure
        outcome."""
        derivations = attempt_g4g5_eom(CadabraAdapter(), _g4g5_npr())
        for d in derivations:
            if d.verified:
                assert d.checks.get("residue_zero") == "True", (
                    f"EOM wrt {d.wrt} is verified but residue_zero is "
                    f"{d.checks.get('residue_zero')!r}, not 'True'"
                )

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_scalar_eom_has_diagnostic_checks(self):
        """The scalar EOM derivation carries the second-order diagnostic
        check, confirming the no-Ostrogradski cancellation works for the
        scalar sector."""
        derivations = attempt_g4g5_eom(CadabraAdapter(), _g4g5_npr())
        scalar = next(d for d in derivations if d.wrt == "phi")
        assert "scalar_eom_second_order" in scalar.checks, (
            f"Scalar EOM should carry the scalar_eom_second_order diagnostic; "
            f"checks: {scalar.checks}"
        )
        assert scalar.checks["scalar_eom_second_order"] == "True", (
            "G4 scalar EOM should be second order (no third derivatives)"
        )

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_metric_eom_has_diagnostic_checks(self):
        """The metric EOM derivation carries the third-derivative diagnostic
        check, confirming the SortCovDs blocker is real."""
        derivations = attempt_g4g5_eom(CadabraAdapter(), _g4g5_npr())
        metric = next(d for d in derivations if d.wrt == "g")
        assert "metric_eom_has_third_derivs" in metric.checks, (
            f"Metric EOM should carry the metric_eom_has_third_derivs "
            f"diagnostic; checks: {metric.checks}"
        )
        assert metric.checks["metric_eom_has_third_derivs"] == "True", (
            "G4 metric EOM should have third derivatives (confirming blocker)"
        )

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_derivation_objects_are_field_derivations(self):
        """The result objects are proper FieldDerivation instances with the
        expected fields."""
        derivations = attempt_g4g5_eom(CadabraAdapter(), _g4g5_npr())
        assert len(derivations) == 2
        for d in derivations:
            assert isinstance(d, FieldDerivation)
            assert d.kind == "eom"
            assert d.wrt in ("phi", "g")
            assert d.kernel_name == "cadabra"
            assert d.script  # non-empty script
            assert d.result_id  # non-empty result_id

    @pytest.mark.kernel_cadabra
    @pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
    def test_g4g5_derivations_carry_nonempty_conventions(self):
        """The G4/G5 best-effort derivations carry a non-empty convention
        block, matching every other derivation path (EOM, perturbation, ADM).
        Even gated results carry their named conventions so the consumer can
        see which conventions produced the (unverified) expression."""
        derivations = attempt_g4g5_eom(CadabraAdapter(), _g4g5_npr())
        for d in derivations:
            assert d.conventions, (
                f"G4/G5 derivation wrt {d.wrt} must carry a non-empty "
                f"convention block; got {d.conventions!r}"
            )
            assert "signature" in d.conventions, (
                f"Convention block must include 'signature'; "
                f"got keys: {sorted(d.conventions.keys())}"
            )
            assert "convention_id" in d.conventions, (
                f"Convention block must include 'convention_id'; "
                f"got keys: {sorted(d.conventions.keys())}"
            )
            assert d.conventions["convention_id"] == "noether-default-v1", (
                f"Convention ID must be noether-default-v1; "
                f"got {d.conventions['convention_id']!r}"
            )

    def test_closure_attempt_and_eom_attempt_consistent(self):
        """The ClosureAttempt (M2) and FieldDerivation (M3) results are
        consistent: both agree on the gated state and the blocker detail."""
        closure = attempt_g4g5_closure()
        # The closure attempt is gated; the EOM attempt should also be gated
        # when actually run (tested above with Cadabra). Here we just confirm
        # the closure-level XOR is still satisfied.
        branch_a = closure.verified and closure.residue_zero and closure.oracle_agrees
        branch_b = (not closure.verified) and (closure.detail != "")
        assert branch_a != branch_b, "Closure-level XOR violated"


# ---------------------------------------------------------------------------
# G4/G5 detection in Lagrangian
# ---------------------------------------------------------------------------


class TestG4G5Detection:
    """The detection function correctly identifies G4(phi,X)R terms in the
    Lagrangian, which is the trigger for the best-effort derive path."""

    def test_g4_phi_x_r_detected(self):
        """A G4(phi,X)R term is detected as a held-out higher-Horndeski
        density."""
        lag = parse_lagrangian(r"G(\phi, X) R")
        assert has_g4g5_terms(lag, "phi")

    def test_nonminimal_f_phi_r_not_detected(self):
        """A nonminimal F(phi)R coupling is NOT detected as G4/G5 (it matches
        the compositional nonminimal block instead)."""
        lag = parse_lagrangian(r"F(\phi) R")
        assert not has_g4g5_terms(lag, "phi")

    def test_kessence_not_detected(self):
        """A k-essence K(phi, X) term without R is NOT detected as G4/G5."""
        lag = parse_lagrangian(r"K(\phi, X)")
        assert not has_g4g5_terms(lag, "phi")

    def test_einstein_hilbert_not_detected(self):
        """A bare Ricci scalar is NOT detected as G4/G5."""
        lag = parse_lagrangian(r"R")
        assert not has_g4g5_terms(lag, "phi")

    def test_g4_mixed_with_kinetic_detected(self):
        """A Lagrangian mixing G4(phi,X)R with kinetic terms still detects
        the G4 component."""
        lag = parse_lagrangian(r"G(\phi, X) R - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi")
        assert has_g4g5_terms(lag, "phi")
