"""General derivation pipeline (orchestrator.derive + kernels.cadabra.generate).

The model parameterizes a Cadabra script; the kernel runs it; the result is
trusted only when the kernel's residue check confirms it. We test the plumbing
deterministically with a StubLLMAdapter that returns a known script, so no
live model is needed; the kernel-backed cases skip when cadabra is absent.
"""

import pytest

from evals.eval1s_adm import build_npr as build_adm_npr
from evals.eval3_scalar_tensor import build_npr
from noether.kernels.base import Capability, ComputedResult, KernelRawOutput, KernelScript
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
    _compositional_decomposition,
    _result_detail,
    derive_adm,
    derive_eom,
    derive_field,
    derive_perturbation,
)
from noether.orchestrator.planner import AmbiguityBlocked, build_plan

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

    def test_box_coupling_routes_scalar_to_cubic_galileon_example(self):
        from evals.eval6_cubic_galileon import build_npr as build_galileon_npr

        npr = build_galileon_npr(resolved=True)
        _, scalar_prompt = build_generation_prompt(npr, "phi")
        # the K(phi) box phi term routes to the audited cubic scaffold, and not
        # the plain scalar-tensor example, which has no double-IBP idiom
        assert templates.get("eom_cubic_galileon_scalar") in scalar_prompt
        assert templates.get("eval3_scalar_tensor_scalar") not in scalar_prompt


class TestConnectionVariationRouting:
    """VAL-EOM-005/006/007: connection variation must route through the general
    derivation path (not only the frozen template), must never fall through to
    the metric worked example, and the Cadabra adapter must advertise the
    INDEPENDENT_CONNECTION capability."""

    @pytest.fixture()
    def palatini_npr(self):
        from evals.eval2_palatini import build_npr as build_palatini_npr

        return build_palatini_npr(resolved=True)

    def test_connection_field_selects_vary_connection_key(self, palatini_npr):
        from noether.kernels.cadabra.generate import _variation_key

        key = _variation_key(palatini_npr, "Gamma", "eom")
        assert key == "vary-connection", (
            "connection field must select vary-connection, not fall through "
            f"to vary-metric (got {key!r})"
        )

    def test_connection_variation_uses_palatini_connection_template(self, palatini_npr):
        _, prompt = build_generation_prompt(palatini_npr, "Gamma")
        # the worked example for connection variation is the audited Palatini
        # connection template, never the metric one
        assert templates.get("eval2_palatini_connection") in prompt
        assert templates.get("eval3_scalar_tensor_metric") not in prompt

    def test_cadabra_adapter_advertises_independent_connection(self):
        adapter = CadabraAdapter()
        assert Capability.INDEPENDENT_CONNECTION in adapter.capabilities(), (
            "CadabraAdapter.capabilities() must include INDEPENDENT_CONNECTION"
        )

    def test_connection_derive_uses_independent_connection_capability(self, palatini_npr):
        from noether.orchestrator.derive import _compositional_decomposition

        # Connection fields have no compositional decomposition; they always
        # route to the model-written script path (with vary-connection example)
        dec = _compositional_decomposition(palatini_npr, "Gamma", "eom")
        assert dec is None, (
            "connection fields must not route through the compositional path"
        )

    def test_derive_field_connection_capability(self, palatini_npr):
        # derive_field should set the capability to INDEPENDENT_CONNECTION
        # for a connection wrt field, not the generic VARY.
        # Verify that the routing picks the right example.
        from noether.kernels.cadabra.generate import _variation_key

        npr = palatini_npr
        key = _variation_key(npr, "Gamma", "eom")
        assert key == "vary-connection"

    def test_metric_field_never_routes_to_connection_example(self, palatini_npr):
        """VAL-EOM-006: a metric field must never be routed to the
        connection-variation worked example. For a Palatini NPR with an
        independent connection, the metric routes to vary-metric-palatini,
        not the standard vary-metric."""
        from noether.kernels.cadabra.generate import _variation_key

        key = _variation_key(palatini_npr, "g", "eom")
        assert key != "vary-connection"
        assert key == "vary-metric-palatini"

    @requires_cadabra
    @pytest.mark.kernel_cadabra
    def test_connection_derive_field_sets_capability(self, palatini_npr):
        """When derive_field runs for a connection, the resulting
        FieldDerivation carries INDEPENDENT_CONNECTION, not VARY."""
        stub = StubLLMAdapter(reply=templates.get("eval2_palatini_connection"))
        result = derive_field(
            palatini_npr,
            "Gamma",
            stub,
            {"cadabra": CadabraAdapter()},
            session_id="s-conn-test",
        )
        assert result.capability is Capability.INDEPENDENT_CONNECTION, (
            f"expected INDEPENDENT_CONNECTION, got {result.capability!r}"
        )
        assert result.wrt == "Gamma"


class TestCompositionalRouting:
    def test_general_scalar_action_routes_compositional(self):
        from evals.eval7_kessence import build_npr as build_kessence_npr

        dec = _compositional_decomposition(build_kessence_npr(resolved=True), "phi", "eom")
        assert dec is not None and dec.full

    def test_kessence_metric_eom_falls_back(self):
        # k-essence has no metric-sector block yet, so its metric EOM decomposes
        # only partially and the model path runs rather than guessing.
        from evals.eval7_kessence import build_npr as build_kessence_npr

        dec = _compositional_decomposition(build_kessence_npr(resolved=True), "g", "eom")
        assert dec is not None and not dec.full

    def test_perturbation_is_not_compositional(self):
        from evals.eval7_kessence import build_npr as build_kessence_npr

        npr = build_kessence_npr(resolved=True)
        assert _compositional_decomposition(npr, "phi", "perturbation") is None

    def test_scalar_tensor_scalar_eom_composes(self):
        # nonminimal F(phi)R now contributes F_phi R to the scalar EOM, so the
        # scalar-tensor scalar equation decomposes fully.
        from evals.eval8_nonminimal import build_npr as build_st_npr

        dec = _compositional_decomposition(build_st_npr(resolved=True), "phi", "eom")
        assert dec is not None and dec.full

    def test_scalar_tensor_metric_eom_composes(self):
        # F(phi)R + kinetic + potential is a full metric-sector decomposition.
        from evals.eval8_nonminimal import build_npr as build_st_npr

        dec = _compositional_decomposition(build_st_npr(resolved=True), "g", "eom")
        assert dec is not None and dec.full

    def test_g4_curvature_term_stays_partial(self):
        # an X-dependent curvature coupling G(phi, X) R is Horndeski G4, held out
        # until it verifies; it matches no block in either sector.
        from noether.kernels.cadabra.blocks import decompose_metric, decompose_scalar
        from noether.npr.parse import parse_lagrangian

        lag = parse_lagrangian(r"G(\phi, X) R - V(\phi)")
        assert not decompose_metric(lag, "phi").full
        assert not decompose_scalar(lag, "phi").full

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


def _computed(checks: dict[str, str], *, stderr: str = "", returncode: int = 0) -> ComputedResult:
    return ComputedResult(
        kernel_name="cadabra",
        kernel_version="test",
        script=KernelScript(kernel_name="cadabra", language="cadabra", source="ex := A;"),
        raw=KernelRawOutput(stdout="", stderr=stderr, returncode=returncode),
        value={"checks": checks},
    )


class TestUnverifiedReason:
    """An unverified run must say *why*: a script that never reached the
    kernel's residue check is a different failure from one that ran and found a
    nonzero residue. Collapsing them hides which happened."""

    def test_no_check_emitted_reports_script_failure_and_stderr(self):
        computed = _computed({}, stderr="RuntimeError: boom in canonicalise", returncode=1)
        detail = _result_detail("eom", False, {}, computed)
        assert "no residue check" in detail
        assert "kernel exit 1" in detail
        assert "boom in canonicalise" in detail

    def test_nonzero_residue_reports_mismatch(self):
        computed = _computed({"residue_zero": "False"})
        detail = _result_detail("eom", False, {"residue_zero": "False"}, computed)
        assert "nonzero residue" in detail
        assert "candidate equation" in detail

    def test_perturbation_nonzero_residue_names_quadratic_action(self):
        checks = {"residue_zero": "False"}
        detail = _result_detail("perturbation", False, checks, _computed(checks))
        assert "quadratic action" in detail

    def test_perturbation_linearized_mismatch_distinguished(self):
        checks = {"residue_zero": "True", "linearized_eom_match": "False"}
        detail = _result_detail("perturbation", False, checks, _computed(checks))
        assert "linearized-EOM" in detail

    def test_verified_message_unchanged(self):
        assert "matches the candidate" in _result_detail("eom", True, {}, _computed({}))


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

    def test_maxwell_perturbation_prompt_uses_gauge_scaffold(self):
        from evals.eval4_maxwell import build_npr as build_maxwell_npr

        npr = build_maxwell_npr(resolved=True)
        _, prompt = build_generation_prompt(npr, "A", kind="perturbation")
        # an abelian gauge potential routes to the Maxwell quadratic scaffold
        assert templates.get("pert_gauge_quadratic") in prompt
        assert templates.get("pert_yang_mills_quadratic") not in prompt

    def test_yang_mills_perturbation_prompt_uses_non_abelian_scaffold(self):
        from evals.eval3y_yang_mills_perturbation import build_npr as build_ym_npr

        npr = build_ym_npr(resolved=True)
        _, prompt = build_generation_prompt(npr, "A", kind="perturbation")
        # the SU(N) marker routes to the Yang-Mills quadratic scaffold
        assert templates.get("pert_yang_mills_quadratic") in prompt

    def test_kessence_perturbation_prompt_uses_x_expansion_scaffold(self):
        from evals.eval7_kessence import build_npr as build_kessence_npr

        npr = build_kessence_npr(resolved=True)
        _, prompt = build_generation_prompt(npr, "phi", kind="perturbation")
        # an X-dependent coupling K(phi, X) routes to the k-essence scaffold,
        # which carries the sound-speed kinetic mixing, not the plain scalar one
        assert templates.get("pert_kessence_quadratic") in prompt
        assert templates.get("pert_scalar_quadratic") not in prompt

    def test_perturbation_rejects_unsupported_field(self):
        from evals.eval4_maxwell import build_npr as build_maxwell_npr

        # the field-strength shorthand F is not a dynamical field kind, so it
        # has no quadratic-action scaffold and the refusal still holds
        npr = build_maxwell_npr(resolved=True)
        with pytest.raises(NotImplementedError):
            build_generation_prompt(npr, "F", kind="perturbation")


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

        # the field-strength shorthand F has no quadratic-action scaffold; the
        # refusal must name it rather than guess (the gauge potential A is
        # supported and exercised in TestVerifiedPerturbation)
        npr = build_maxwell_npr(resolved=True)
        with pytest.raises(NotImplementedError):
            derive_perturbation(
                npr,
                StubLLMAdapter(),
                {"cadabra": CadabraAdapter()},
                fields=["F"],
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

    def test_maxwell_quadratic_action_verified(self):
        from evals.eval4_maxwell import build_npr as build_maxwell_npr

        npr = build_maxwell_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("pert_gauge_quadratic"))
        results = derive_perturbation(
            npr, stub, {"cadabra": CadabraAdapter()}, fields=["A"], session_id="s-test"
        )
        assert [r.wrt for r in results] == ["A"]
        d = results[0]
        assert d.kind == "perturbation"
        assert d.verified is True, d.checks
        assert d.checks.get("residue_zero") == "True"
        assert d.checks.get("linearized_eom_match") == "True"
        assert d.result_tex

    def test_yang_mills_quadratic_action_verified(self):
        from evals.eval3y_yang_mills_perturbation import build_npr as build_ym_npr

        npr = build_ym_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("pert_yang_mills_quadratic"))
        results = derive_perturbation(
            npr, stub, {"cadabra": CadabraAdapter()}, fields=["A"], session_id="s-test"
        )
        assert [r.wrt for r in results] == ["A"]
        d = results[0]
        assert d.kind == "perturbation"
        assert d.verified is True, d.checks
        assert d.checks.get("residue_zero") == "True"
        assert d.checks.get("linearized_eom_match") == "True"
        assert d.result_tex

    def test_kessence_quadratic_action_verified(self):
        from evals.eval7_kessence import build_npr as build_kessence_npr

        npr = build_kessence_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("pert_kessence_quadratic"))
        results = derive_perturbation(
            npr, stub, {"cadabra": CadabraAdapter()}, fields=["phi"], session_id="s-test"
        )
        assert [r.wrt for r in results] == ["phi"]
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


# ---------------------------------------------------------------------------
# VAL-EOM-018: Palatini elicitation gate blocks the connection EOM until
# the connection ambiguity is resolved
# ---------------------------------------------------------------------------


class TestPalatiniElicitationGate:
    """VAL-EOM-018: an unresolved Palatini NPR cannot derive the connection
    EOM (build_plan raises AmbiguityBlocked); resolving to independent
    enables the INDEPENDENT_CONNECTION plan step."""

    @pytest.fixture()
    def palatini_npr_unresolved(self):
        from evals.eval2_palatini import build_npr as build_palatini_npr

        return build_palatini_npr(resolved=False)

    @pytest.fixture()
    def palatini_npr_resolved(self):
        from evals.eval2_palatini import build_npr as build_palatini_npr

        return build_palatini_npr(resolved=True)

    def test_unresolved_palatini_blocks_build_plan(self, palatini_npr_unresolved):
        """build_plan raises AmbiguityBlocked on an unresolved Palatini NPR."""
        with pytest.raises(AmbiguityBlocked):
            build_plan(palatini_npr_unresolved)

    def test_unresolved_palatini_blocks_derive_field(self, palatini_npr_unresolved):
        """derive_field raises AmbiguityBlocked on an unresolved Palatini
        NPR (the no-guessing gate)."""
        with pytest.raises(AmbiguityBlocked):
            derive_field(
                palatini_npr_unresolved,
                "Gamma",
                StubLLMAdapter(),
                {"cadabra": CadabraAdapter()},
                session_id="s",
            )

    def test_unresolved_palatini_blocks_derive_eom(self, palatini_npr_unresolved):
        """derive_eom raises AmbiguityBlocked on an unresolved Palatini
        NPR."""
        with pytest.raises(AmbiguityBlocked):
            derive_eom(
                palatini_npr_unresolved,
                StubLLMAdapter(),
                {"cadabra": CadabraAdapter()},
                session_id="s",
            )

    def test_resolved_independent_plan_includes_connection_step(self, palatini_npr_resolved):
        """Resolving to independent enables the INDEPENDENT_CONNECTION plan
        step."""
        plan = build_plan(palatini_npr_resolved)
        capabilities = [s.capability for s in plan.steps]
        assert Capability.INDEPENDENT_CONNECTION in capabilities, (
            f"plan must include INDEPENDENT_CONNECTION; got {[c.value for c in capabilities]}"
        )

    def test_resolved_independent_connection_step_description(self, palatini_npr_resolved):
        """The INDEPENDENT_CONNECTION step description must reflect the
        torsion and non-metricity flags."""
        plan = build_plan(palatini_npr_resolved)
        conn_step = next(
            s for s in plan.steps if s.capability is Capability.INDEPENDENT_CONNECTION
        )
        desc = conn_step.description.lower()
        assert "torsion" in desc or "nonmetricity" in desc, (
            f"step description must mention torsion/nonmetricity; got: {conn_step.description}"
        )

    def test_resolved_levi_civita_no_connection_step(self):
        """A well-posed NPR with connection=levi-civita must NOT have an
        INDEPENDENT_CONNECTION step."""
        npr = build_npr(resolved=True)
        # build_npr from eval3 creates a Levi-Civita NPR
        plan = build_plan(npr)
        capabilities = [s.capability for s in plan.steps]
        assert Capability.INDEPENDENT_CONNECTION not in capabilities


# ---------------------------------------------------------------------------
# VAL-EOM-017: Gated EOM results carry a visible, non-empty reason
# ---------------------------------------------------------------------------


class TestGatedEomDetail:
    """VAL-EOM-017: any unverified/gated EOM result returns verified==False
    with a non-empty detail identifying the blocker."""

    def test_nonzero_residue_gated_with_detail(self):
        """A derivation with nonzero residue must be verified==False with a
        non-empty detail naming the mismatch."""
        npr = build_npr(resolved=True)
        broken = (
            'print("NOETHER_RESULT: x");\n'
            'print("NOETHER_CHECK: residue_zero=False");\n'
        )
        result = derive_field(
            npr, "g", StubLLMAdapter(reply=broken), {"cadabra": CadabraAdapter()},
            session_id="s",
        )
        assert result.verified is False
        assert result.detail, "gated result must have non-empty detail"
        assert "unverified" in result.detail.lower() or "nonzero" in result.detail.lower()
        assert "residue" in result.detail.lower() or "mismatch" in result.detail.lower()

    def test_no_residue_check_gated_with_detail(self):
        """A derivation that never reaches the residue check must be
        verified==False with a non-empty detail about the script failure."""
        npr = build_npr(resolved=True)
        result = derive_field(
            npr, "g", StubLLMAdapter(reply="ex := 1;\n"), {"cadabra": CadabraAdapter()},
            session_id="s",
        )
        assert result.verified is False
        assert result.detail, "gated result must have non-empty detail"
        assert "no residue check" in result.detail or "did not run" in result.detail

    def test_verified_result_has_detail_too(self):
        """A verified result also has a detail (the success message), which
        is distinguishable from a gated one."""
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("eval3_scalar_tensor_metric"))
        result = derive_field(
            npr, "g", stub, {"cadabra": CadabraAdapter()}, session_id="s",
        )
        assert result.verified is True
        # verified results also carry a detail (the success message)
        assert result.detail

    def test_perturbation_nonzero_residue_gated_with_detail(self):
        """A perturbation with nonzero residue must be verified==False
        with detail naming the quadratic action mismatch."""
        npr = build_npr(resolved=True)
        broken = (
            'print("NOETHER_RESULT: x");\n'
            'print("NOETHER_CHECK: residue_zero=False");\n'
        )
        result = derive_field(
            npr, "phi", StubLLMAdapter(reply=broken), {"cadabra": CadabraAdapter()},
            kind="perturbation", session_id="s",
        )
        assert result.verified is False
        assert result.detail
        assert "quadratic action" in result.detail
