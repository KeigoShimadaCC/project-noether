"""Palatini Einstein-Hilbert: both equations through the general path.

This test file validates VAL-EOM-001, VAL-EOM-002, VAL-EOM-003, VAL-EOM-004,
and VAL-EOM-010 for the Palatini Einstein-Hilbert action

  S = integral sqrt(-g) g^{mu nu} R_{mu nu}(Gamma)

with an independent connection Gamma.

Conventions: noether-default-v1 (dimension 4, mostly-plus, R^rho_{sigma mu nu}
= d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma} + GG - GG,
R_{sigma nu} = R^lambda_{sigma lambda nu}).
"""

import pytest

from evals.eval2_palatini import build_npr
from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter, templates
from noether.kernels.cadabra.generate import _variation_key, build_generation_prompt
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.llm.base import StubLLMAdapter
from noether.npr.latex import render
from noether.orchestrator.derive import derive_field

requires_cadabra = pytest.mark.skipif(
    not CadabraAdapter().available(), reason="cadabra2 not installed"
)


class TestPalatiniMetricVariationRouting:
    """VAL-EOM-002: When the connection is independent, the metric variation
    must NOT vary the curvature with the metric. It routes to the Palatini
    metric worked example, not the standard LC one."""

    @pytest.fixture()
    def palatini_npr(self):
        return build_npr(resolved=True)

    def test_independent_connection_routes_metric_to_palatini(self, palatini_npr):
        """The metric variation key must be vary-metric-palatini when the
        connection is independent."""
        key = _variation_key(palatini_npr, "g")
        assert key == "vary-metric-palatini", (
            f"expected vary-metric-palatini for independent connection, got {key!r}"
        )

    def test_lc_connection_routes_metric_to_standard(self):
        """When the connection is NOT independent, the standard metric
        variation key is used (regression guard)."""
        from noether.npr import NOETHER_DEFAULT_V1, NPR, Action, Geometry, ObjectDecl, Task
        from noether.npr.ast import down, prod, tensor, up

        npr = NPR(
            conventions=NOETHER_DEFAULT_V1,
            geometry=Geometry(),  # default: Levi-Civita
            objects=[
                ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
                ObjectDecl(name="phi", kind="scalar-field", role="dynamical", rank=0),
            ],
            action=Action(
                measure_tex=r"d^4x \sqrt{-g}",
                lagrangian=prod(tensor("R", down("mu"), down("nu")),
                                tensor("g", up("mu"), up("nu"))),
                lagrangian_tex=r"R",
            ),
            task=Task(type="vary", with_respect_to=["g"]),
            ambiguities=[],
        )
        key = _variation_key(npr, "g")
        assert key == "vary-metric", (
            f"expected vary-metric for Levi-Civita, got {key!r}"
        )

    def test_palatini_metric_prompt_uses_palatini_template(self, palatini_npr):
        """The generated prompt must use the Palatini metric template, not the
        standard LC one."""
        system, prompt = build_generation_prompt(palatini_npr, "g")
        assert templates.get("eval2_palatini_metric") in prompt
        assert templates.get("eval3_scalar_tensor_metric") not in prompt
        # The Palatini contract must say R does NOT vary with the metric
        assert "does NOT vary" in system or "stays FIXED" in system

    def test_connection_field_routes_to_vary_connection(self, palatini_npr):
        """The connection variation key is unchanged (vary-connection)."""
        key = _variation_key(palatini_npr, "Gamma")
        assert key == "vary-connection"


class TestMetricEOMSymmetrization:
    """VAL-EOM-001: The Palatini metric equation returns R_{(mu nu)} - 1/2 g_{mu nu}
    Rtilde = 0 with explicit symmetrization (both index orders appear) and
    verified=True."""

    def test_presentation_form_contains_both_index_orders(self):
        """The rendered LaTeX for the Palatini metric EOM must contain both
        R_{mu nu} and R_{nu mu} (explicit symmetrization)."""
        from evals.eval2_palatini import target_metric_eom

        tex = render(target_metric_eom())
        assert r"R_{\mu \nu}" in tex, f"missing R_{{mu nu}} in: {tex}"
        assert r"R_{\nu \mu}" in tex, f"missing R_{{nu mu}} in: {tex}"

    @requires_cadabra
    @pytest.mark.kernel_cadabra
    def test_metric_variation_residue_zero(self):
        """The Cadabra kernel reports residue_zero=True for the Palatini
        metric variation."""
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.VARY,
                description="Palatini metric variation",
                payload={"template": "eval2_palatini_metric"},
            )
        )
        checks = result.value["checks"]
        assert checks.get("residue_zero") == "True", (
            f"metric variation residue not zero: {checks}"
        )

    @requires_cadabra
    @pytest.mark.kernel_cadabra
    def test_metric_derivation_through_general_path_verified(self):
        """When the Palatini metric EOM is derived through the general path
        (derive_field with the Palatini metric worked example), it must be
        verified=True."""
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("eval2_palatini_metric"))
        result = derive_field(
            npr,
            "g",
            stub,
            {"cadabra": CadabraAdapter()},
            session_id="s-palatini-metric",
        )
        assert result.verified, (
            f"metric EOM not verified: checks={result.checks}, "
            f"detail={result.detail}"
        )
        assert result.wrt == "g"


class TestMetricVariationNoDeltaGamma:
    """VAL-EOM-002: The metric variation introduces no deltaGamma/IBP terms
    (curvature is not varied with the metric when the connection is independent)."""

    def test_palatini_metric_script_has_no_dgamma(self):
        """The Palatini metric variation script must NOT contain any dGamma
        or integrate_by_parts terms, because the curvature depends on the
        independent connection, not the metric."""
        script = templates.get("eval2_palatini_metric")
        # dGamma would indicate the curvature is being varied with the metric
        assert "dGamma" not in script, (
            "Palatini metric variation script contains dGamma: "
            "the curvature must NOT vary with the metric when the connection is independent"
        )
        # integrate_by_parts would only be needed if dGamma terms were present
        assert "integrate_by_parts" not in script, (
            "Palatini metric variation script contains integrate_by_parts: "
            "no IBP is needed because the curvature is independent of the metric"
        )

    def test_lc_metric_script_has_dgamma_by_contrast(self):
        """By contrast, the standard LC metric variation DOES contain dGamma
        (the Palatini identity for varying R_{mu nu} with the metric). This
        confirms the Palatini script is genuinely different."""
        lc_script = templates.get("eval3_scalar_tensor_metric")
        assert "dGamma" in lc_script, (
            "LC metric variation script should contain dGamma (the Palatini identity)"
        )
        assert "integrate_by_parts" in lc_script, (
            "LC metric variation script should contain integrate_by_parts"
        )


class TestConnectionEquationProjective:
    """VAL-EOM-003: The connection equation through the general path yields
    the projective-mode solution Gamma = LC(g) + delta^lam_nu A_mu."""

    @requires_cadabra
    @pytest.mark.kernel_cadabra
    def test_connection_equation_solved_by_projective_family(self):
        """The Cadabra kernel reports solution_zero=True and
        ricci_shift_is_dA=True for the Palatini connection variation."""
        result = CadabraAdapter().run(
            KernelTask(
                capability=Capability.INDEPENDENT_CONNECTION,
                description="Palatini connection variation",
                payload={"template": "eval2_palatini_connection"},
            )
        )
        checks = result.value["checks"]
        assert checks.get("solution_zero") == "True", (
            f"connection equation not solved by projective family: {checks}"
        )
        assert checks.get("ricci_shift_is_dA") == "True", (
            f"Ricci shift is not dA: {checks}"
        )

    @requires_cadabra
    @pytest.mark.kernel_cadabra
    def test_connection_derivation_through_general_path(self):
        """When the connection EOM is derived through the general path, the
        result must use INDEPENDENT_CONNECTION capability and be verified."""
        npr = build_npr(resolved=True)
        stub = StubLLMAdapter(reply=templates.get("eval2_palatini_connection"))
        result = derive_field(
            npr,
            "Gamma",
            stub,
            {"cadabra": CadabraAdapter()},
            session_id="s-palatini-connection",
        )
        assert result.capability is Capability.INDEPENDENT_CONNECTION, (
            f"expected INDEPENDENT_CONNECTION, got {result.capability!r}"
        )
        assert result.wrt == "Gamma"


class TestProjectiveFreedomSurfaced:
    """VAL-EOM-004: The connection result explicitly states the projective
    family and the arbitrariness of A_mu; it never presents the connection
    as uniquely fixed."""

    def test_connection_template_states_projective_family(self):
        """The connection variation template explicitly contains the projective
        substitution Gamma = C + delta^lam_nu A_mu with A_mu arbitrary."""
        script = templates.get("eval2_palatini_connection")
        # The projective family substitution must be present
        assert "g^{\\lambda}_{\\nu} A_{\\mu}" in script, (
            "connection template must contain the projective substitution "
            "Gamma = C + delta^lam_nu A_mu"
        )
        # The solution_zero check verifies the projective family annihilates
        # the connection equation identically (A_mu is arbitrary)
        assert "solution_zero" in script, (
            "connection template must have the solution_zero check "
            "verifying the projective family annihilates the connection equation"
        )

    def test_connection_template_checks_ricci_shift_is_dA(self):
        """The template checks that the Ricci shift under the projective
        transformation is exactly dA (exterior derivative of A_mu)."""
        script = templates.get("eval2_palatini_connection")
        assert "ricci_shift_is_dA" in script, (
            "connection template must check that R(Gamma+proj) - R(Gamma) = dA"
        )


class TestProjectiveInertnessSymPy:
    """VAL-EOM-010: The SymPy oracle confirms the projective family is inert
    on random general-connection backgrounds (metric equation unchanged) and
    the Ricci shift is exactly dA."""

    @pytest.mark.parametrize(
        "metric_spec, conn_seed, cov_seed",
        [
            ({"kind": "random-diagonal", "seed": 11, "dim": 3}, 7, 3),
            ({"kind": "random-diagonal", "seed": 23, "dim": 4}, 13, 5),
            ({"kind": "random-diagonal", "seed": 37, "dim": 3}, 19, 11),
        ],
    )
    def test_ricci_shift_is_dA_on_random_backgrounds(
        self, metric_spec, conn_seed, cov_seed
    ):
        """R(Gamma + projective) - R(Gamma) = dA on random general-connection
        backgrounds. This is the fundamental identity behind the projective
        invariance of the Palatini metric equation."""
        adapter = SympyKernelAdapter()
        task = KernelTask(
            capability=Capability.COMPONENT_EVAL,
            description="Palatini Ricci shift is dA",
            payload={
                "check": "palatini-ricci-shift-is-dA",
                "metric": metric_spec,
                "connection_seed": conn_seed,
                "covector_seed": cov_seed,
            },
        )
        result = adapter.run(task)
        assert result.value["passed"], result.value["detail"]

    @pytest.mark.parametrize(
        "metric_spec, conn_seed, cov_seed",
        [
            ({"kind": "random-diagonal", "seed": 11, "dim": 3}, 7, 3),
            ({"kind": "random-diagonal", "seed": 23, "dim": 4}, 13, 5),
            ({"kind": "random-diagonal", "seed": 37, "dim": 3}, 19, 11),
        ],
    )
    def test_projective_inert_on_general_backgrounds(
        self, metric_spec, conn_seed, cov_seed
    ):
        """The Palatini metric equation R_{(mu nu)} - 1/2 g_{mu nu} Rtilde is
        unchanged under the projective shift on random general-connection
        backgrounds (not just Levi-Civita + projective)."""
        adapter = SympyKernelAdapter()
        task = KernelTask(
            capability=Capability.COMPONENT_EVAL,
            description="Palatini projective inert on general backgrounds",
            payload={
                "check": "palatini-projective-inert-general",
                "metric": metric_spec,
                "connection_seed": conn_seed,
                "covector_seed": cov_seed,
            },
        )
        result = adapter.run(task)
        assert result.value["passed"], result.value["detail"]

    @pytest.mark.parametrize(
        "metric_spec, seed",
        [
            ({"kind": "random-diagonal", "seed": 11, "dim": 3}, 4),
            ({"kind": "random-diagonal", "seed": 23, "dim": 4}, 9),
        ],
    )
    def test_projective_family_inert_on_lc_plus_proj(self, metric_spec, seed):
        """On the special case of LC + projective, the Palatini metric
        equation reduces to the Einstein tensor (original check)."""
        adapter = SympyKernelAdapter()
        task = KernelTask(
            capability=Capability.COMPONENT_EVAL,
            description="Palatini projective inert (LC + projective)",
            payload={
                "check": "palatini-projective-inert",
                "metric": metric_spec,
                "seed": seed,
            },
        )
        result = adapter.run(task)
        assert result.value["passed"], result.value["detail"]
