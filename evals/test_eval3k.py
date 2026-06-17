"""Executable eval 3k (perturbation sector): the k-essence quadratic action.

Kernel gate (skips without cadabra2): the frozen `pert_kessence_quadratic`
template expands an X-dependent scalar action to quadratic order about a
covariantly-constant-gradient background and proves, two independent ways, that
the fluctuation obeys the k-essence linearized equation, the sound-speed kinetic
mixing KXX (nabla phibar . nabla chi)^2 that distinguishes it from a plain
scalar (eval 3p).
"""

import pytest

from evals.eval3k_kessence_perturbation import TEMPLATE
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestKEssenceQuadraticAction:
    def _run(self):
        return CadabraAdapter().run(
            KernelTask(
                capability=Capability.PERTURB,
                description="k-essence quadratic-action expansion",
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
        # S2 is the printed NOETHER_RESULT: the fluctuation Lagrangian. The KXX
        # term is the X-kinetic mixing that gives a nontrivial sound speed.
        assert result.expression_tex
        assert "KXX" in result.expression_tex and "chi" in result.expression_tex
