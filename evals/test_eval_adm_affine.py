"""Executable eval: metric-affine ADM (3+1) decomposition.

Layers:
  1. Elicitation gate: foliation/K-sign/boundary questions must block planning;
     the documented answers unblock an "adm" plan.
  2. Metric-sector ADM checks (same as GR, via adm-gr-1p2).
  3. Connection-sector ADM checks (via adm-affine-1p2).
  4. Structural verification of the presented decomposition and constraints.
"""

import pytest

from evals.eval_adm_affine import (
    ELICITATION_ANSWERS,
    build_npr,
)
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.kernels.sympy_kernel.adm import adm_affine_sample_1p2
from noether.orchestrator.planner import AmbiguityBlocked, build_plan


@pytest.fixture(scope="module")
def affine_adm_checks() -> dict[str, tuple[bool, str]]:
    return adm_affine_sample_1p2().run_all_affine()


class TestElicitationGate:
    def test_unresolved_npr_cannot_plan(self):
        with pytest.raises(AmbiguityBlocked):
            build_plan(build_npr(resolved=False))

    def test_documented_answers_unblock_adm_plan(self):
        npr = build_npr(resolved=False)
        for amb in npr.ambiguities:
            amb.resolution = ELICITATION_ANSWERS[amb.id]
        plan = build_plan(npr)
        assert plan.task_type == "adm"
        assert "constraints-as-normal-projections" in plan.verification


class TestMetricSectorChecks:
    """The metric-sector GR ADM checks still pass."""

    def test_gr_adm_component_verification(self):
        from noether.kernels.base import Capability, KernelTask

        adapter = SympyKernelAdapter()
        result = adapter.run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="ADM GR 1+2 check",
                payload={"check": "adm-gr-1p2"},
            )
        )
        assert result.value.get("passed"), result.value.get("detail", "")


class TestConnectionSectorChecks:
    """The connection-sector metric-affine ADM checks pass."""

    def test_background_nondegenerate(self, affine_adm_checks):
        ok, detail = affine_adm_checks["background-nondegenerate-affine"]
        assert ok, detail

    def test_post_riemannian_on_foliation(self, affine_adm_checks):
        ok, detail = affine_adm_checks["post-riemannian-on-foliation"]
        assert ok, detail

    def test_torsion_nonmetricity_foliation(self, affine_adm_checks):
        ok, detail = affine_adm_checks["torsion-nonmetricity-foliation"]
        assert ok, detail

    def test_distortion_spatial_projections(self, affine_adm_checks):
        ok, detail = affine_adm_checks["distortion-spatial-projections"]
        assert ok, detail

    def test_connection_eom_algebraic(self, affine_adm_checks):
        ok, detail = affine_adm_checks["connection-eom-algebraic"]
        assert ok, detail

    def test_connection_sector_primary_constraints(self, affine_adm_checks):
        ok, detail = affine_adm_checks["connection-sector-primary-constraints"]
        assert ok, detail


class TestSympyAdapterAffineCheck:
    """The adm-affine-1p2 check suite runs through the adapter."""

    def test_adapter_adm_affine_1p2(self):
        from noether.kernels.base import Capability, KernelTask

        adapter = SympyKernelAdapter()
        result = adapter.run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="metric-affine ADM 1+2 check",
                payload={"check": "adm-affine-1p2"},
            )
        )
        assert result.value.get("passed"), result.value.get("detail", "")
