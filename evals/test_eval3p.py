"""Executable eval 3p (perturbation sector): the scalar quadratic action.

Kernel gate (skips without cadabra2): the frozen `pert_scalar_quadratic`
template expands the scalar action to quadratic order and proves, two
independent ways, that the fluctuation obeys box chi - V''(phibar) chi = 0.
"""

import pytest

from evals.eval3p_scalar_perturbation import TEMPLATE
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestScalarQuadraticAction:
    def _run(self):
        return CadabraAdapter().run(
            KernelTask(
                capability=Capability.PERTURB,
                description="scalar quadratic-action expansion",
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
        # S2 is the printed NOETHER_RESULT: the genuinely quadratic Lagrangian.
        assert result.expression_tex
        assert "chi" in result.expression_tex
