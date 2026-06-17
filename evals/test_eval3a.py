"""Executable eval 3a (perturbation sector): the Maxwell quadratic action.

Kernel gate (skips without cadabra2): the frozen `pert_gauge_quadratic`
template expands the Maxwell action to quadratic order about a background
potential and proves, two independent ways, that the fluctuation obeys the
source-free linearized Maxwell equation nabla_mu f^{mu nu} = 0.
"""

import pytest

from evals.eval3a_maxwell_perturbation import TEMPLATE
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestMaxwellQuadraticAction:
    def _run(self):
        return CadabraAdapter().run(
            KernelTask(
                capability=Capability.PERTURB,
                description="Maxwell quadratic-action expansion",
                payload={"template": TEMPLATE},
            )
        )

    def test_quadratic_action_eom_residue_zero(self):
        result = self._run()
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("residue_zero") == "True", result.raw.stdout

    def test_matches_linearized_full_eom(self):
        result = self._run()
        assert result.value["checks"].get("linearized_eom_match") == "True", result.raw.stdout

    def test_quadratic_action_result_returned(self):
        result = self._run()
        # S2 is the printed NOETHER_RESULT: the fluctuation Maxwell Lagrangian,
        # with the linearized strength written out as derivatives of a.
        assert result.expression_tex
        assert "nabla" in result.expression_tex and "a_" in result.expression_tex
