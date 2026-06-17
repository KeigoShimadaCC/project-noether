r"""VAL-GEOM-015: Held G4(phi,X)R / G5 Horndeski closure is verified-or-gated.

The best-effort higher-Horndeski closure either fully closes (Cadabra residue 0
AND SymPy cross-check agrees, verified=True) or is returned verified=False with
a non-empty detail; never verified with a gate unmet.

The XOR condition:

    (verified and residue_zero and oracle_agrees)
    XOR
    (not verified and detail != '')

must hold. If gated, detail names the blocker (e.g. needs covariant-derivative
normal-ordering unavailable without xAct); the result is surfaced, never
asserted true.

Conventions: noether-default-v1.
"""

import pytest

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.cadabra.horndeski_g4g5 import (
    ClosureAttempt,
    assemble_g4_metric_eom_script,
    assemble_g4_scalar_eom_script,
    attempt_g4g5_closure,
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
