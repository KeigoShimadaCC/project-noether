"""SymPy kernel adapter: component checks through the adapter interface."""

import pytest

from noether.kernels.base import Capability, KernelTask
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.npr.ast import add, down, num, prod, tensor, up

FAST_METRIC = {"kind": "random-diagonal", "seed": 7, "dim": 4}


def run_check(payload):
    adapter = SympyKernelAdapter()
    task = KernelTask(capability=Capability.COMPONENT_EVAL, description="test", payload=payload)
    return adapter.run(task)


def einstein_lhs():
    return tensor("G", down("mu"), down("nu"))


def einstein_expanded():
    return add(
        tensor("R", down("mu"), down("nu")),
        prod(
            num(-1, 2),
            tensor("g", down("mu"), down("nu")),
            tensor("R", down("alpha"), down("beta")),
            tensor("g", up("alpha"), up("beta")),
        ),
    )


class TestChecks:
    def test_einstein_equals_expansion(self):
        result = run_check(
            {
                "check": "equal",
                "lhs": einstein_lhs().model_dump(),
                "rhs": einstein_expanded().model_dump(),
                "metric": FAST_METRIC,
            }
        )
        assert result.value["passed"], result.value["detail"]

    def test_einstein_symmetric(self):
        result = run_check(
            {
                "check": "symmetric",
                "expr": einstein_lhs().model_dump(),
                "metric": FAST_METRIC,
            }
        )
        assert result.value["passed"], result.value["detail"]

    def test_einstein_divergence_free(self):
        result = run_check(
            {
                "check": "divergence-zero",
                "expr": einstein_lhs().model_dump(),
                "metric": FAST_METRIC,
            }
        )
        assert result.value["passed"], result.value["detail"]

    def test_wrong_claim_is_falsified(self):
        # R_{mu nu} alone must FAIL the divergence check on a curved metric.
        result = run_check(
            {
                "check": "divergence-zero",
                "expr": tensor("R", down("mu"), down("nu")).model_dump(),
                "metric": FAST_METRIC,
            }
        )
        assert not result.value["passed"]

    def test_trace_of_einstein_is_minus_r(self):
        # g^{mu nu} G_{mu nu} = -R in d=4: the eval-1 observation, kernel-computed.
        lhs = prod(tensor("g", up("mu"), up("nu")), tensor("G", down("mu"), down("nu")))
        rhs = prod(num(-1), tensor("R"))
        result = run_check(
            {
                "check": "equal",
                "lhs": lhs.model_dump(),
                "rhs": rhs.model_dump(),
                "metric": FAST_METRIC,
            }
        )
        assert result.value["passed"], result.value["detail"]

    def test_provenance_script_is_executable_python(self):
        result = run_check(
            {
                "check": "symmetric",
                "expr": einstein_lhs().model_dump(),
                "metric": FAST_METRIC,
            }
        )
        compile(result.script.source, "<reproduction>", "exec")

    def test_unknown_check_rejected(self):
        with pytest.raises(ValueError, match="unknown check"):
            run_check({"check": "nope", "metric": FAST_METRIC})
