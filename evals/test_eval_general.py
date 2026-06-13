"""Acceptance gate for the general derivation path (orchestrator/derive.py).

Eval 3 proves the frozen golden templates reproduce the scalar-tensor EOMs.
This eval proves the *general* pipeline does too: for the same well-posed NPR,
`derive_eom` asks the model for a script, runs it in Cadabra, and trusts each
field equation only when the kernel's own residue check passes.

The model is stubbed to the audited script (a live model is non-deterministic
and cannot gate CI); what is under test here is the orchestration, namely that
generate -> run -> residue gate -> verified verdict reproduces eval 3's two
kernel-verified equations end to end, and that the no-guessing gate still
refuses an unresolved problem. The script content is the golden, golden-tested
template, so a green residue here is the kernel's verdict, not the stub's.
"""

import pytest

from evals.eval3_scalar_tensor import build_npr
from noether.kernels.cadabra import CadabraAdapter, templates
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.llm.base import StubLLMAdapter
from noether.orchestrator.derive import derive_eom, derive_perturbation
from noether.orchestrator.planner import AmbiguityBlocked


class _FieldAwareStub:
    """Returns the audited script that matches the field being varied. The
    generation prompt ends with 'vary with respect to <field>', so we branch
    on that to hand back the right golden template."""

    name = "stub"

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return "stub-1"

    def complete(self, system: str, prompt: str) -> str:
        if "with respect to phi" in prompt:
            return templates.get("eval3_scalar_tensor_scalar")
        return templates.get("eval3_scalar_tensor_metric")


class TestGeneralPathGate:
    def test_unresolved_npr_refuses_to_derive(self):
        adapters = {"cadabra": CadabraAdapter(), "sympy": SympyKernelAdapter()}
        with pytest.raises(AmbiguityBlocked):
            derive_eom(
                build_npr(resolved=False),
                _FieldAwareStub(),
                adapters,
                session_id="eval-general",
            )


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestGeneralPathReproducesEval3:
    def test_both_field_equations_are_kernel_verified(self, tmp_path):
        adapters = {"cadabra": CadabraAdapter(), "sympy": SympyKernelAdapter()}
        derivations = derive_eom(
            build_npr(resolved=True),
            _FieldAwareStub(),
            adapters,
            session_id="eval-general",
            results_root=tmp_path / "results",
        )
        assert [d.wrt for d in derivations] == ["g", "phi"]
        for d in derivations:
            assert d.verified is True, d.checks
            assert d.result_tex
            assert d.kernel_name == "cadabra"
            assert d.bundle_path  # provenance written for every run


class TestGeneralPerturbationGate:
    def test_unresolved_npr_refuses_to_perturb(self):
        adapters = {"cadabra": CadabraAdapter(), "sympy": SympyKernelAdapter()}
        with pytest.raises(AmbiguityBlocked):
            derive_perturbation(
                build_npr(resolved=False),
                StubLLMAdapter(reply=templates.get("pert_scalar_quadratic")),
                adapters,
                fields=["phi"],
                session_id="eval-general-pert",
            )


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestGeneralPathReproducesEval3p:
    """The general perturbation path drives the same kernel checks as eval 3p:
    the stubbed-but-audited quadratic-action script must pass both the residue
    gate and the independent linearized-EOM match before the orchestrator calls
    it verified."""

    def test_scalar_quadratic_action_is_kernel_verified(self, tmp_path):
        adapters = {"cadabra": CadabraAdapter(), "sympy": SympyKernelAdapter()}
        derivations = derive_perturbation(
            build_npr(resolved=True),
            StubLLMAdapter(reply=templates.get("pert_scalar_quadratic")),
            adapters,
            session_id="eval-general-pert",
            results_root=tmp_path / "results",
        )
        assert [d.wrt for d in derivations] == ["phi"]
        d = derivations[0]
        assert d.kind == "perturbation"
        assert d.verified is True, d.checks
        assert d.checks.get("linearized_eom_match") == "True"
        assert d.result_tex
        assert d.bundle_path
