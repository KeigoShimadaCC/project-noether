"""Executable eval 3g (perturbation sector): the graviton quadratic action.

Kernel gate (skips without cadabra2): the frozen `pert_metric_quadratic`
template expands the Einstein-Hilbert action to quadratic order about a flat
background and proves, two independent ways, that the fluctuation obeys the
linearized vacuum Einstein equation G^{(1)}_{mu nu} = 0.
"""

import pytest

from evals.eval3g_graviton_perturbation import TEMPLATE
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestGravitonQuadraticAction:
    def _run(self):
        return CadabraAdapter().run(
            KernelTask(
                capability=Capability.PERTURB,
                description="graviton quadratic-action expansion",
                payload={"template": TEMPLATE},
            )
        )

    def test_quadratic_action_eom_residue_zero(self):
        result = self._run()
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout

    def test_matches_linearized_einstein_tensor(self):
        result = self._run()
        assert result.value["checks"].get("linearized_eom_match") == "True", result.raw.stdout

    def test_quadratic_action_result_returned(self):
        result = self._run()
        # The printed NOETHER_RESULT is the quadratic Lagrangian L2.
        assert result.expression_tex
        assert "h" in result.expression_tex
