"""General derivation pipeline (orchestrator.derive + kernels.cadabra.generate).

The model parameterizes a Cadabra script; the kernel runs it; the result is
trusted only when the kernel's residue check confirms it. We test the plumbing
deterministically with a StubLLMAdapter that returns a known script, so no
live model is needed; the kernel-backed cases skip when cadabra is absent.
"""

import pytest

from evals.eval1s_adm import build_npr as build_adm_npr
from evals.eval3_scalar_tensor import build_npr
from noether.kernels.cadabra import CadabraAdapter, templates
from noether.kernels.cadabra.generate import (
    build_generation_prompt,
    generate_script,
    strip_fences,
)
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.llm.base import StubLLMAdapter
from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Geometry, ObjectDecl, Task
from noether.npr.ast import tensor
from noether.orchestrator.derive import (
    derive_adm,
    derive_eom,
    derive_field,
    derive_perturbation,
)
from noether.orchestrator.planner import AmbiguityBlocked

requires_cadabra = pytest.mark.skipif(
    not CadabraAdapter().available(), reason="cadabra2 not installed"
)


class TestPromptGeneration:
    def test_prompt_carries_action_conventions_and_field(self):
        npr = build_npr(resolved=True)
        system, prompt = build_generation_prompt(npr, "g")
        assert "Cadabra2" in system
        assert "NOETHER_CHECK: residue_zero" in system
        assert npr.action.lagrangian_tex in prompt
        assert npr.conventions.id in prompt
        assert "with respect to g" in prompt
        # the worked example must be an audited frozen template
        assert templates.get("eval3_scalar_tensor_metric") in prompt

    def test_variation_key_picks_example_by_field_kind(self):
        npr = build_npr(resolved=True)
        _, metric_prompt = build_generation_prompt(npr, "g")
        _, scalar_prompt = build_generation_prompt(npr, "phi")
        assert templates.get("eval3_scalar_tensor_metric") in metric_prompt
        assert templates.get("eval3_scalar_tensor_scalar") in scalar_prompt

    def test_strip_fences_removes_markdown(self):
        fenced = "```cadabra\nex := A;\n```"
        assert strip_fences(fenced) == "ex := A;"
        assert strip_fences("ex := A;") == "ex := A;"

    def test_generate_script_uses_llm_output(self):
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply='ex := A;\nprint("NOETHER_RESULT: A")')
        generated = generate_script(npr, "g", stub)
        assert generated.source.startswith("ex := A;")
        assert generated.llm_name == "stub"
        assert generated.variation_key == "vary-metric"


class TestNoGuessingGate:
    def test_unresolved_npr_blocks_derivation(self):
        npr = build_npr(resolved=False)
        with pytest.raises(AmbiguityBlocked):
            derive_field(npr, "g", StubLLMAdapter(), {"cadabra": CadabraAdapter()}, session_id="s")


@requires_cadabra
@pytest.mark.kernel_cadabra
class TestVerifiedDerivation:
    def test_metric_eom_verified_when_script_is_correct(self):
        npr = build_npr(resolved=True)
        # The stub stands in for a model that produced a correct script.
        stub = StubLLMAdapter(reply=templates.get("eval3_scalar_tensor_metric"))
        result = derive_field(npr, "g", stub, {"cadabra": CadabraAdapter()}, session_id="s-test")
        assert result.verified, result.checks
        assert result.result_tex
        assert result.kernel_name == "cadabra"

    def test_bundle_written_with_provenance(self, tmp_path):
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("eval3_scalar_tensor_metric"))
        result = derive_field(
            npr,
            "g",
            stub,
            {"cadabra": CadabraAdapter()},
            session_id="s-test",
            results_root=tmp_path,
        )
        assert result.bundle_path is not None
        base = tmp_path / "s-test"
        assert base.exists()
        # provenance: the generated script and the kernel raw output are kept
        assert any(base.rglob("scripts/*.cdb"))
        assert any(base.rglob("checks.json"))

    def test_unverified_result_surfaced_as_such(self):
        npr = build_npr(resolved=True)
        # A script the kernel runs but whose residue does not vanish.
        broken = 'print("NOETHER_RESULT: x");\nprint("NOETHER_CHECK: residue_zero=False");\n'
        result = derive_field(
            npr, "g", StubLLMAdapter(reply=broken), {"cadabra": CadabraAdapter()}, session_id="s"
        )
        assert result.verified is False
        assert "unverified" in result.detail

    def test_derive_eom_covers_all_varied_fields(self):
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("eval3_scalar_tensor_metric"))
        results = derive_eom(npr, stub, {"cadabra": CadabraAdapter()}, session_id="s")
        assert [r.wrt for r in results] == ["g", "phi"]


class TestPerturbationPromptGeneration:
    def test_scalar_perturbation_prompt_uses_quadratic_scaffold(self):
        npr = build_npr(resolved=True)
        system, prompt = build_generation_prompt(npr, "phi", kind="perturbation")
        assert "keep_weight" in system
        assert "WeightInherit" in system
        assert "linearized_eom_match" in system
        assert "quadratic order" in prompt
        assert templates.get("pert_scalar_quadratic") in prompt

    def test_metric_perturbation_prompt_uses_quadratic_scaffold(self):
        npr = build_npr(resolved=True)
        _, prompt = build_generation_prompt(npr, "g", kind="perturbation")
        assert templates.get("pert_metric_quadratic") in prompt

    def test_perturbation_rejects_unsupported_field(self):
        from evals.eval4_maxwell import build_npr as build_maxwell_npr

        npr = build_maxwell_npr(resolved=True)
        with pytest.raises(NotImplementedError):
            build_generation_prompt(npr, "A", kind="perturbation")


class TestPerturbationGate:
    def test_unresolved_npr_blocks_perturbation(self):
        npr = build_npr(resolved=False)
        with pytest.raises(AmbiguityBlocked):
            derive_field(
                npr,
                "phi",
                StubLLMAdapter(),
                {"cadabra": CadabraAdapter()},
                kind="perturbation",
                session_id="s",
            )

    def test_perturbation_refuses_unsupported_field(self):
        from evals.eval4_maxwell import build_npr as build_maxwell_npr

        npr = build_maxwell_npr(resolved=True)
        with pytest.raises(NotImplementedError):
            derive_perturbation(
                npr,
                StubLLMAdapter(),
                {"cadabra": CadabraAdapter()},
                fields=["A"],
                session_id="s",
            )


@requires_cadabra
@pytest.mark.kernel_cadabra
class TestVerifiedPerturbation:
    def test_scalar_quadratic_action_verified(self):
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("pert_scalar_quadratic"))
        results = derive_perturbation(
            npr, stub, {"cadabra": CadabraAdapter()}, fields=["phi"], session_id="s-test"
        )
        assert [r.wrt for r in results] == ["phi"]
        d = results[0]
        assert d.kind == "perturbation"
        assert d.verified is True, d.checks
        assert d.checks.get("linearized_eom_match") == "True"
        assert d.result_tex

    def test_metric_quadratic_action_verified(self):
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("pert_metric_quadratic"))
        results = derive_perturbation(
            npr, stub, {"cadabra": CadabraAdapter()}, fields=["g"], session_id="s-test"
        )
        assert [r.wrt for r in results] == ["g"]
        d = results[0]
        assert d.kind == "perturbation"
        assert d.verified is True, d.checks
        assert d.checks.get("residue_zero") == "True"
        assert d.checks.get("linearized_eom_match") == "True"
        assert d.result_tex

    def test_perturbation_defaults_to_dynamical_fields(self):
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("pert_scalar_quadratic"))
        results = derive_perturbation(npr, stub, {"cadabra": CadabraAdapter()}, session_id="s")
        # both the dynamical metric and the dynamical scalar are perturbable
        assert [r.wrt for r in results] == ["g", "phi"]


def _scalar_only_npr() -> NPR:
    """A well-posed action with no metric, to exercise the ADM refusal."""
    return NPR(
        conventions=NOETHER_DEFAULT_V1,
        geometry=Geometry(),
        objects=[ObjectDecl(name="phi", kind="scalar-field", role="dynamical", rank=0)],
        action=Action(measure_tex="d^4x", lagrangian=tensor("phi"), lagrangian_tex="phi"),
        task=Task(type="adm", with_respect_to=["phi"]),
        ambiguities=[],
    )


class TestVerifiedAdm:
    """ADM writes no model script: the SymPy component kernel verifies the
    Gauss-Codazzi split and the Einstein-tensor projections, so these run
    without cadabra or an LLM."""

    def test_adm_decomposition_verified(self):
        results = derive_adm(
            build_adm_npr(resolved=True), {"sympy": SympyKernelAdapter()}, session_id="s-adm"
        )
        assert len(results) == 3
        assert all(d.kind == "adm" for d in results)
        assert all(d.verified for d in results), [d.checks for d in results]
        assert all(d.result_tex for d in results)
        assert results[0].kernel_name == "sympy"
        # every named component check passed
        assert results[0].checks.get("lagrangian-split") == "True"
        assert results[0].checks.get("hamiltonian-projection") == "True"
        assert results[0].checks.get("momentum-projection") == "True"

    def test_adm_writes_provenance_bundle(self, tmp_path):
        results = derive_adm(
            build_adm_npr(resolved=True),
            {"sympy": SympyKernelAdapter()},
            session_id="s-adm",
            results_root=tmp_path,
        )
        assert results[0].bundle_path is not None
        assert (tmp_path / "s-adm").exists()
        assert any((tmp_path / "s-adm").rglob("checks.json"))

    def test_adm_refuses_action_without_metric(self):
        with pytest.raises(NotImplementedError):
            derive_adm(_scalar_only_npr(), {"sympy": SympyKernelAdapter()}, session_id="s")

    def test_adm_blocked_when_questions_open(self):
        with pytest.raises(AmbiguityBlocked):
            derive_adm(
                build_adm_npr(resolved=False), {"sympy": SympyKernelAdapter()}, session_id="s"
            )
