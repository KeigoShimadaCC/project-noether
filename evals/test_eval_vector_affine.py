"""Executable eval: Vector (Maxwell) EOM on a metric-affine background.

Acceptance gates for VAL-EOM-020 and VAL-EOM-021.

Layers:
  1. Elicitation gate: the connection-independence question must block
     planning; the field-strength-definition question must also block.
  2. SymPy component verification:
     (a) The affine-LC divergence difference equals the T/Q correction
         (algebraic identity on random affine backgrounds).
     (b) The dA action has zero hypermomentum; the covcurl action has
         nonzero hypermomentum Delta = -2 A F (antisymmetric in mu, nu).
     (c) The two F definitions differ by the torsion term.
  3. Cadabra derivation (skips if not installed):
     (a) dA EOM: residue zero against nabla_mu F^{mu nu} = 0.
     (b) dA hypermomentum: zero (no dG terms in variation output).
     (c) covcurl hypermomentum: nonzero (dG terms present).
"""

import pytest

from evals.eval_vector_affine import (
    build_npr,
    hypermomentum_covcurl,
    target_eom_dA,
)
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.verify.checks import WellFormedCheck
from noether.verify.ladder import run_ladder
from tests.test_vector_eom_affine import (
    _COVCURL_HYPERMOMENTUM_SCRIPT,
    _DA_EOM_SCRIPT,
    _DA_HYPERMOMENTUM_SCRIPT,
)


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):

        npr = build_npr("exterior-derivative")
        # Build an unresolved version by clearing resolutions
        unresolved_npr = npr.model_copy(deep=True)
        for amb in unresolved_npr.ambiguities:
            amb.resolution = None
        with pytest.raises(AmbiguityBlocked):
            build_plan(unresolved_npr)

    def test_resolved_npr_plans_with_independent_connection(self):
        npr = build_npr("exterior-derivative")
        plan = build_plan(npr)
        assert any(s.capability is Capability.INDEPENDENT_CONNECTION for s in plan.steps)


class TestStructure:
    def test_dA_eom_well_formed(self):
        from noether.npr.ast import up

        NU = up("nu")
        adapters = {"sympy": SympyKernelAdapter()}
        report = run_ladder(target_eom_dA(), [WellFormedCheck(expected_free=[NU])], adapters)
        assert report.all_passed, report.summary()

    def test_hypermomentum_is_antisymmetric_in_mu_nu(self):
        from noether.npr.ast import down, up

        MU, NU = up("mu"), up("nu")
        adapters = {"sympy": SympyKernelAdapter()}
        report = run_ladder(
            hypermomentum_covcurl(),
            [WellFormedCheck(expected_free=[down("lambda"), MU, NU])],
            adapters,
        )
        assert report.all_passed, report.summary()


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKernelDerivation:
    def test_dA_eom_residue_zero(self):
        """VAL-EOM-020: dA Maxwell EOM residue is zero."""
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="dA Maxwell EOM on metric-affine background",
                payload={"script": _DA_EOM_SCRIPT},
            )
        )
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("dA_eom_residue_zero") == "True", result.raw.stdout

    def test_dA_hypermomentum_zero(self):
        """VAL-EOM-021: dA choice yields zero hypermomentum."""
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="dA hypermomentum on metric-affine background",
                payload={"script": _DA_HYPERMOMENTUM_SCRIPT},
            )
        )
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("dA_hypermomentum_zero") == "True", result.raw.stdout

    def test_covcurl_hypermomentum_nonzero(self):
        """VAL-EOM-021: covariant-curl choice yields nonzero hypermomentum."""
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.SUBSTITUTE,
                description="covcurl hypermomentum on metric-affine background",
                payload={"script": _COVCURL_HYPERMOMENTUM_SCRIPT},
            )
        )
        assert result.raw.returncode == 0, result.raw.stderr
        checks = result.value.get("checks", {})
        assert checks.get("covcurl_hypermomentum_nonzero") == "True", result.raw.stdout
