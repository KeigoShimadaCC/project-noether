"""Cadabra adapter: availability handling and the eval-1 golden run.

The golden run is the kernel-verified derivation of eval 1. It skips cleanly
when cadabra2 is not installed; CI must run it on an image that has the
kernel, otherwise Horizon 1 is not actually gated.
"""

import pytest

from noether.kernels.base import Capability, KernelTask, KernelUnavailable
from noether.kernels.cadabra import CadabraAdapter, templates

requires_cadabra = pytest.mark.skipif(
    not CadabraAdapter().available(), reason="cadabra2 not installed"
)


class TestAvailability:
    def test_unavailable_raises_kernel_unavailable(self):
        adapter = CadabraAdapter(executable="/nonexistent/cadabra2")
        adapter.executable = None  # force unavailable regardless of host
        task = KernelTask(
            capability=Capability.VARY, description="t", payload={"template": "eval1_eh_trace"}
        )
        with pytest.raises(KernelUnavailable):
            adapter.run(task)

    def test_unknown_template_rejected(self):
        with pytest.raises(KeyError):
            templates.get("not-a-template")

    def test_template_required(self):
        adapter = CadabraAdapter()
        if not adapter.available():
            pytest.skip("cadabra2 not installed")
        task = KernelTask(capability=Capability.VARY, description="t", payload={})
        with pytest.raises(ValueError, match="audited template"):
            adapter.run(task)


@requires_cadabra
@pytest.mark.kernel_cadabra
class TestGoldenEval1:
    def test_eh_trace_variation_residue_zero(self):
        adapter = CadabraAdapter()
        task = KernelTask(
            capability=Capability.VARY,
            description="EH trace-form metric variation",
            payload={"template": "eval1_eh_trace"},
        )
        result = adapter.run(task)
        assert result.raw.returncode == 0, result.raw.stderr
        assert result.value["checks"].get("residue_zero") == "True", (
            f"derivation residue nonzero:\n{result.raw.stdout}"
        )
