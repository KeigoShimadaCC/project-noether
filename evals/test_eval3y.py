"""Executable eval 3y (perturbation sector): the Yang-Mills quadratic action.

Kernel gate (skips without cadabra2): the frozen `pert_yang_mills_quadratic`
template expands the non-abelian gauge action to quadratic order about a
background potential and proves, two independent ways, that the fluctuation
obeys the linearized YM equation Dbar_mu f^{a mu nu} + g f^{abc} v^b Fbar^c = 0.
"""

import pytest

from evals.eval3y_yang_mills_perturbation import TEMPLATE
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestYangMillsQuadraticAction:
    def _run(self):
        return CadabraAdapter().run(
            KernelTask(
                capability=Capability.PERTURB,
                description="Yang-Mills quadratic-action expansion",
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
        # S2 is the printed NOETHER_RESULT: the quadratic YM Lagrangian (f1, Fbar).
        assert result.expression_tex
        assert "f1" in result.expression_tex or "Fbar" in result.expression_tex
