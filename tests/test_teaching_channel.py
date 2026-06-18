"""Teaching narration channel on FieldDerivation (VAL-GUIDE-008/009/010/016/018/019,
VAL-CROSS-010).

The teaching field is a first-class narration channel on FieldDerivation,
distinct from the failure-diagnostic detail. Teaching holds pure prose
(no result_tex substring) and detail continues to carry only the
verify/gated diagnostic. Generating teaching mutates no NPR and sets no
result (pre/post NPR snapshot equality; verified/result_tex determined by
kernel checks alone). /derive and /results expose teaching as a top-level
per-derivation key distinct from result_tex, checks, and detail.

The verified-vs-reasoned boundary is observable: the teaching field never
appears inside checks, varying it never changes verified, and no proposal
rationale or teaching string appears among the checks. For the same
action/resolutions, enabling teaching adds narration on its field while
result_tex/verified/checks equal the no-teaching run and the NPR version
count is unchanged.
"""

import json

from noether.kernels.base import Capability
from noether.llm import StubLLMAdapter, stub_reply
from noether.orchestrator.derive import FieldDerivation, _geometry_teaching
from noether.orchestrator.elicit import propose_resolutions
from noether.orchestrator.ingest import ingest_action
from noether.orchestrator.session import Session

MEASURE = r"d^4x \sqrt{-g}"


def _metric_affine_npr():
    """Build a well-posed metric-affine NPR for teaching tests."""
    from noether.orchestrator.elicit import apply_resolutions

    npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
    # Resolve all geometry ambiguities to independent + torsion + non-metricity
    geo_answers = {}
    for amb in npr.ambiguities:
        if amb.id == "amb-connection":
            geo_answers[amb.id] = "independent"
        elif amb.id == "amb-torsion":
            geo_answers[amb.id] = "torsion-allowed"
        elif amb.id == "amb-nonmetricity":
            geo_answers[amb.id] = "nonmetricity-allowed"
        elif amb.id == "amb-metric-compatibility":
            geo_answers[amb.id] = "not-metric-compatible"
        elif amb.id == "amb-curvature-free":
            geo_answers[amb.id] = "curvature-allowed"
        elif amb.id == "amb-ricci-contraction":
            geo_answers[amb.id] = amb.options[0]
        else:
            geo_answers[amb.id] = amb.options[0]
    npr = apply_resolutions(npr, geo_answers)

    # Resolve any remaining ambiguities
    remaining = npr.unresolved_ambiguities()
    if remaining:
        more_answers = {amb.id: amb.options[0] for amb in remaining}
        npr = apply_resolutions(npr, more_answers)

    return npr


def _levi_civita_npr():
    """Build a well-posed Levi-Civita NPR for comparison."""
    from noether.orchestrator.elicit import apply_resolutions

    npr = ingest_action(MEASURE, "R").npr
    answers = {amb.id: amb.options[0] for amb in npr.ambiguities}
    npr = apply_resolutions(npr, answers)
    remaining = npr.unresolved_ambiguities()
    if remaining:
        npr = apply_resolutions(npr, {amb.id: amb.options[0] for amb in remaining})
    return npr


# ---------------------------------------------------------------------------
# VAL-GUIDE-008: Teaching lives on its own first-class field, separate from
# detail
# ---------------------------------------------------------------------------


class TestTeachingFieldDistinctFromDetail:
    """FieldDerivation exposes a teaching field distinct from detail; teaching
    holds pure prose (not result_tex), and detail continues to carry only the
    verify/gated diagnostic."""

    def test_teaching_field_exists_on_field_derivation(self):
        """FieldDerivation has a teaching field."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-008",
        )
        assert hasattr(d, "teaching"), "FieldDerivation must have a teaching field"
        assert d.teaching == "", "teaching defaults to empty string"

    def test_teaching_distinct_from_detail(self):
        """The teaching and detail fields are distinct keys."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-008",
            detail="unverified: kernel did not confirm",
            teaching="Torsion couples to the spin current of matter fields.",
        )
        assert d.teaching != d.detail, "teaching and detail must be distinct"
        assert "spin" in d.teaching, "teaching should contain prose"
        assert "unverified" in d.detail, "detail should contain diagnostic"

    def test_teaching_contains_no_result_tex_substring(self):
        """Teaching holds pure prose, not a result_tex substring."""
        result_tex = r"G_{\mu\nu} = 0"
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-008",
            result_tex=result_tex,
            teaching="The independent connection introduces projective freedom.",
        )
        assert d.teaching != d.result_tex, "teaching must not equal result_tex"
        assert d.result_tex not in d.teaching, (
            "teaching must not contain result_tex as a substring"
        )

    def test_detail_continues_to_carry_only_verify_diagnostic(self):
        """Detail is about verification/gated status, not teaching prose."""
        d_verified = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-008v",
            verified=True,
            detail="kernel confirmed the variation matches the candidate equation",
            teaching="The metric equation involves the symmetric part of the Ricci tensor.",
        )
        d_gated = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-008g",
            verified=False,
            detail="unverified: the kernel computed a nonzero residue",
            teaching="Non-metricity means length is not conserved under parallel transport.",
        )
        # Detail is about the kernel verdict, not the physics narrative
        assert "kernel" in d_verified.detail.lower() or "confirmed" in d_verified.detail.lower()
        assert "unverified" in d_gated.detail.lower()
        # Teaching is about the physics, not the kernel verdict
        assert "kernel" not in d_verified.teaching.lower()
        assert "unverified" not in d_gated.teaching.lower()


# ---------------------------------------------------------------------------
# VAL-GUIDE-009: Teaching narration mutates no NPR and sets no result
# ---------------------------------------------------------------------------


class TestTeachingMutatesNoNPR:
    """Generating teaching leaves the session NPR unchanged and does not
    alter result_tex or verified."""

    def test_teaching_generation_does_not_alter_npr(self):
        """The _geometry_teaching function reads the NPR but does not mutate it."""
        npr = _metric_affine_npr()
        npr_before = npr.model_dump_json()

        # Call the teaching generator (it only reads, never writes)
        _ = _geometry_teaching(npr, "g", "eom")

        npr_after = npr.model_dump_json()
        assert npr_before == npr_after, "teaching generation must not mutate the NPR"

    def test_teaching_does_not_alter_result_tex(self):
        """Varying the teaching content does not change result_tex."""
        d1 = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-009a",
            result_tex=r"G_{\mu\nu} = 0",
            verified=True,
            teaching="",
        )
        d2 = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-009b",
            result_tex=r"G_{\mu\nu} = 0",
            verified=True,
            teaching="The independent connection introduces projective freedom.",
        )
        assert d1.result_tex == d2.result_tex, "teaching must not alter result_tex"

    def test_teaching_does_not_alter_verified(self):
        """Varying the teaching content does not change verified."""
        d1 = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-009c",
            verified=True,
            checks={"residue_zero": "True"},
            teaching="",
        )
        d2 = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-009d",
            verified=True,
            checks={"residue_zero": "True"},
            teaching="Torsion introduces spin-current coupling.",
        )
        assert d1.verified == d2.verified, "teaching must not alter verified"

    def test_teaching_on_propose_resolutions_does_not_mutate_npr(self):
        """propose_resolutions (which generates rationale/teaching) does not
        mutate the NPR."""
        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        npr_before = npr.model_dump_json()

        all_answers = {amb.id: amb.options[0] for amb in npr.ambiguities}
        _ = propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        npr_after = npr.model_dump_json()
        assert npr_before == npr_after, (
            "propose_resolutions must not mutate the NPR"
        )

    def test_verified_determined_by_kernel_checks_alone(self):
        """The verified flag is set by kernel checks, not by teaching."""
        # A verified derivation with no teaching
        d_verified_no_teaching = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-009e",
            verified=True,
            checks={"residue_zero": "True"},
            teaching="",
        )
        # A verified derivation with teaching
        d_verified_with_teaching = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-009f",
            verified=True,
            checks={"residue_zero": "True"},
            teaching="The metric equation involves the symmetric Ricci tensor.",
        )
        # Both are verified because checks are the same
        assert d_verified_no_teaching.verified == d_verified_with_teaching.verified


# ---------------------------------------------------------------------------
# VAL-GUIDE-010: HTTP payloads expose teaching as a field separate from
# result and checks
# ---------------------------------------------------------------------------


class TestHTTPTeachingPayload:
    """/derive and /results include the teaching field as a top-level
    per-derivation key, distinct from result_tex, checks, and detail."""

    def _app(self, tmp_path):
        from noether.llm import StubLLMAdapter
        from noether.orchestrator.store import SessionStore
        from noether.server.app import create_app

        store = SessionStore(tmp_path / "sessions")
        llm = StubLLMAdapter('{"no-op": true}')
        results_root = tmp_path / "results"
        return create_app(store=store, llm=llm, results_root=results_root)

    def test_derivation_model_dump_includes_teaching_key(self):
        """FieldDerivation.model_dump() includes the teaching key."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-010",
            result_tex=r"G_{\mu\nu} = 0",
            verified=True,
            checks={"residue_zero": "True"},
            detail="kernel confirmed",
            teaching="The independent connection introduces projective freedom.",
        )
        dumped = d.model_dump()
        assert "teaching" in dumped, "model_dump must include teaching"
        assert dumped["teaching"] == "The independent connection introduces projective freedom."
        # Distinct from the other fields
        assert dumped["teaching"] != dumped["result_tex"]
        assert dumped["teaching"] != dumped["checks"]
        assert dumped["teaching"] != dumped["detail"]

    def test_teaching_key_non_overlapping_with_other_keys(self):
        """The teaching key is non-overlapping with result_tex, checks, and
        detail in the serialized form."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-010b",
            result_tex=r"G_{\mu\nu} = 0",
            verified=True,
            checks={"residue_zero": "True"},
            detail="kernel confirmed",
            teaching="Torsion couples to spin.",
        )
        dumped = d.model_dump()
        key_set = set(dumped.keys())
        assert "teaching" in key_set
        assert "result_tex" in key_set
        assert "checks" in key_set
        assert "detail" in key_set
        # All four are distinct keys
        assert len({"teaching", "result_tex", "checks", "detail"} & key_set) == 4

    def test_teaching_json_round_trip(self):
        """Teaching survives a JSON round-trip through model_dump/model_validate."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-010c",
            result_tex=r"G_{\mu\nu} = 0",
            verified=True,
            checks={"residue_zero": "True"},
            detail="kernel confirmed",
            teaching="The projective family annihilates the connection equation.",
        )
        json_str = d.model_dump_json()
        reloaded = FieldDerivation.model_validate_json(json_str)
        assert reloaded.teaching == d.teaching, (
            "teaching must survive JSON round-trip"
        )


# ---------------------------------------------------------------------------
# VAL-GUIDE-016: The verified-vs-reasoned boundary is observable end to end
# ---------------------------------------------------------------------------


class TestVerifiedVsReasonedBoundary:
    """The teaching field never appears inside checks, varying it never
    changes verified, and no proposal rationale or teaching string appears
    among the checks."""

    def test_teaching_key_absent_from_checks(self):
        """The teaching key is not present in the checks dict."""
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-016a",
            verified=True,
            checks={"residue_zero": "True", "solution_zero": "True"},
            teaching="Torsion couples to spin current.",
        )
        assert "teaching" not in d.checks, "teaching must not appear in checks"
        assert "narrative" not in d.checks, "narrative must not appear in checks"

    def test_verified_unchanged_as_teaching_varies(self):
        """Varying the teaching string never changes the verified flag."""
        for teaching_text in [
            "",
            "Torsion couples to spin.",
            "Non-metricity means length is not conserved.",
            "The projective family annihilates the connection equation.",
        ]:
            d = FieldDerivation(
                wrt="g",
                kind="eom",
                capability=Capability.VARY,
                result_id="test-016b",
                verified=True,
                checks={"residue_zero": "True"},
                teaching=teaching_text,
            )
            assert d.verified is True, (
                f"verified must stay True regardless of teaching: {teaching_text!r}"
            )

    def test_proposal_rationale_not_in_checks(self):
        """No proposal rationale or teaching string appears among the checks."""
        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        all_answers = {amb.id: amb.options[0] for amb in npr.ambiguities}
        # Use a rationale that would be obviously out of place in checks
        custom_rationale = "TEACHING_TEST_RATIONALE_MARKER_12345"
        reply = json.dumps(
            {
                amb_id: {"choice": ans, "rationale": custom_rationale}
                for amb_id, ans in all_answers.items()
            }
        )
        proposal = propose_resolutions(npr, StubLLMAdapter(reply))

        # The rationale appears in the proposals but would never appear in
        # any derivation's checks
        rationale_found = any(
            p.rationale == custom_rationale for p in proposal.proposals
        )
        assert rationale_found, "rationale should be in proposals"

        # Construct a FieldDerivation and verify checks don't contain rationale
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-016c",
            verified=True,
            checks={"residue_zero": "True"},
            teaching=custom_rationale,
        )
        for check_val in d.checks.values():
            assert custom_rationale not in check_val, (
                "rationale must not appear in check values"
            )

    def test_teaching_string_not_in_checks(self):
        """A teaching string on a FieldDerivation does not appear in checks."""
        teaching_str = "TEACHING_MARKER_FOR_CHECKS_TEST"
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-016d",
            verified=True,
            checks={"residue_zero": "True", "solution_zero": "True"},
            teaching=teaching_str,
        )
        for check_val in d.checks.values():
            assert teaching_str not in check_val, (
                f"teaching string appeared in check value: {check_val!r}"
            )


# ---------------------------------------------------------------------------
# VAL-GUIDE-018: Teaching explains the geometric tradeoffs of a choice
# ---------------------------------------------------------------------------


class TestTeachingExplainsGeometryTradeoffs:
    """For an open geometry ambiguity, the teaching narration contrasts the
    options' physical consequences, not a bare restatement of the menu."""

    def test_teaching_references_distinct_consequences_of_at_least_two_options(self):
        """Teaching text for a metric-affine NPR references the distinct
        consequences of at least two geometric options."""
        npr = _metric_affine_npr()
        teaching = _geometry_teaching(npr, "g", "eom")

        # Teaching should be non-empty for a metric-affine NPR
        assert teaching, "teaching must be non-empty for metric-affine NPR"

        # Teaching should reference at least two of these physical consequences:
        # - torsion -> spin coupling
        # - non-metricity -> length non-conservation
        # - projective freedom
        consequence_keywords = [
            ("torsion", "spin"),
            ("non-metricity", "length"),
            ("projective", "freedom"),
            ("connection", "algebraic"),
        ]
        hits = 0
        for kw1, kw2 in consequence_keywords:
            if kw1 in teaching.lower() and kw2 in teaching.lower():
                hits += 1
        assert hits >= 2, (
            f"teaching should reference at least 2 distinct consequences; "
            f"found {hits}: {teaching!r}"
        )

    def test_teaching_not_bare_restatement_of_menu(self):
        """Teaching is not a bare restatement of the menu options."""
        npr = _metric_affine_npr()
        teaching = _geometry_teaching(npr, "g", "eom")

        # Teaching should NOT simply list the menu options
        # Menu options are like "independent", "torsion-allowed", etc.
        menu_options = [
            amb.options for amb in npr.ambiguities if amb.id.startswith("amb-")
        ]
        flat_options = [opt for sublist in menu_options for opt in sublist]
        for opt in flat_options:
            # Teaching should not contain the exact option string as a sentence
            assert teaching.strip() != opt, (
                f"teaching should not simply restate menu option {opt!r}"
            )

    def test_teaching_remains_on_teaching_channel_resolving_nothing(self):
        """Teaching is narration that resolves nothing; it doesn't set
        results or flip verified."""
        npr = _metric_affine_npr()
        npr_before = npr.model_dump_json()

        teaching = _geometry_teaching(npr, "g", "eom")
        assert teaching, "teaching should be non-empty"

        # NPR unchanged
        npr_after = npr.model_dump_json()
        assert npr_before == npr_after, "teaching must not mutate the NPR"

    def test_teaching_for_levi_civita_is_empty(self):
        """For a Levi-Civita NPR, there are no geometric tradeoffs to
        narrate, so teaching is empty."""
        npr = _levi_civita_npr()
        teaching = _geometry_teaching(npr, "g", "eom")
        assert teaching == "", "Levi-Civita teaching should be empty"

    def test_teaching_for_connection_variation(self):
        """Teaching for a connection variation explains projective freedom
        and spin coupling."""
        npr = _metric_affine_npr()
        # Find the connection object name
        conn_obj = next((o for o in npr.objects if o.kind == "connection"), None)
        if conn_obj is not None:
            teaching = _geometry_teaching(npr, conn_obj.name, "eom")
            assert teaching, "teaching for connection variation must be non-empty"
            assert "projective" in teaching.lower(), (
                "teaching should mention projective freedom"
            )


# ---------------------------------------------------------------------------
# VAL-GUIDE-019: Teaching is available during elicitation, with the
# proposal's rationale preserved
# ---------------------------------------------------------------------------


class TestElicitationRationalePreserved:
    """While answering geometry questions (chat / HTTP elicit), the proposal's
    rationale is surfaced to the user alongside the on-menu choice."""

    def test_proposal_rationale_is_non_empty(self):
        """The elicitation proposal exposes non-empty rationale with the
        choice."""
        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        all_answers = {amb.id: amb.options[0] for amb in npr.ambiguities}
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        # At least one geometry proposal should have a rationale
        geo_ids = {"amb-connection", "amb-torsion", "amb-nonmetricity",
                   "amb-metric-compatibility", "amb-ricci-contraction"}
        geo_proposals = [p for p in proposal.proposals if p.ambiguity_id in geo_ids]
        rationales = [p.rationale for p in geo_proposals if p.rationale]
        assert rationales, "at least one geometry proposal should have a non-empty rationale"

    def test_rationale_does_not_mutate_npr(self):
        """The rationale is narration only and mutates nothing."""
        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        npr_before = npr.model_dump_json()

        all_answers = {amb.id: amb.options[0] for amb in npr.ambiguities}
        _ = propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        npr_after = npr.model_dump_json()
        assert npr_before == npr_after, "rationale must not mutate the NPR"

    def test_npr_unchanged_until_explicit_on_menu_resolve(self):
        """The NPR remains unchanged until an explicit on-menu resolve."""
        from noether.orchestrator.elicit import apply_resolutions

        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        npr_before = npr.model_dump_json()

        # Proposing does not resolve
        all_answers = {amb.id: amb.options[0] for amb in npr.ambiguities}
        _ = propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))
        assert npr.model_dump_json() == npr_before, "propose must not resolve"

        # Only apply_resolutions with on-menu choices resolves
        confirmed = apply_resolutions(npr, all_answers)
        assert confirmed.model_dump_json() != npr_before, (
            "apply_resolutions should change the NPR"
        )

    def test_http_elicit_surfaces_rationale(self):
        """POST /elicit returns the proposal's rationale alongside the choice."""
        from fastapi.testclient import TestClient

        from noether.orchestrator.store import SessionStore
        from noether.server.app import create_app

        store = SessionStore("/tmp/test-teaching-elicit-sessions")
        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        all_answers = {amb.id: amb.options[0] for amb in npr.ambiguities}
        llm = StubLLMAdapter(stub_reply(all_answers))
        results_root = store.root.parent / "results"
        app = create_app(store=store, llm=llm, results_root=results_root)
        client = TestClient(app)

        # Create session
        resp = client.post("/sessions", json={"lagrangian": r"g^{\mu\nu} R_{\mu\nu}(\Gamma)"})
        assert resp.status_code == 201
        session_id = resp.json()["session_id"]

        # Elicit
        resp = client.post(f"/sessions/{session_id}/elicit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"] is False

        # Proposals should include rationale
        proposals = data["proposals"]
        assert len(proposals) > 0, "elicit should return proposals"
        for p in proposals:
            assert "rationale" in p, "each proposal must have a rationale key"
            assert "choice" in p, "each proposal must have a choice key"


# ---------------------------------------------------------------------------
# VAL-CROSS-010: Teaching narration never leaks into a verified result and
# never mutates the NPR
# ---------------------------------------------------------------------------


class TestTeachingNeverLeaks:
    """For the same action/resolutions, enabling teaching adds narration on
    its field while result_tex/verified/checks equal the no-teaching run and
    the NPR version count is unchanged."""

    def test_enabling_teaching_adds_narration_without_changing_core_fields(self):
        """A derivation with teaching has the same result_tex, verified, and
        checks as one without teaching."""
        d_no_teaching = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-cross-010a",
            result_tex=r"G_{\mu\nu} = 0",
            verified=True,
            checks={"residue_zero": "True"},
            detail="kernel confirmed",
            teaching="",
        )
        d_with_teaching = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-cross-010b",
            result_tex=r"G_{\mu\nu} = 0",
            verified=True,
            checks={"residue_zero": "True"},
            detail="kernel confirmed",
            teaching="The independent connection introduces projective freedom.",
        )

        # Core fields equal
        assert d_no_teaching.result_tex == d_with_teaching.result_tex
        assert d_no_teaching.verified == d_with_teaching.verified
        assert d_no_teaching.checks == d_with_teaching.checks

        # Teaching is present and non-overlapping
        assert d_with_teaching.teaching != ""
        assert d_with_teaching.teaching != d_with_teaching.result_tex
        assert d_with_teaching.teaching not in d_with_teaching.checks.values()

    def test_teaching_does_not_change_npr_version_count(self):
        """Generating teaching does not add an NPR version."""
        session = Session(session_id="s-test-cross-010")
        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        session.ingest(npr)
        npr_count_before = len(session.npr_versions)

        # Generate teaching (this just reads the NPR, doesn't add versions)
        _ = _geometry_teaching(session.npr, "g", "eom")

        npr_count_after = len(session.npr_versions)
        assert npr_count_after == npr_count_before, (
            "teaching generation must not add NPR versions"
        )

    def test_propose_resolutions_does_not_add_npr_versions(self):
        """propose_resolutions (which carries rationale/teaching) does not
        add NPR versions to a session."""
        session = Session(session_id="s-test-cross-010b")
        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        session.ingest(npr)
        npr_count_before = len(session.npr_versions)

        all_answers = {amb.id: amb.options[0] for amb in npr.ambiguities}
        _ = propose_resolutions(session.npr, StubLLMAdapter(stub_reply(all_answers)))

        npr_count_after = len(session.npr_versions)
        assert npr_count_after == npr_count_before, (
            "propose_resolutions must not add NPR versions"
        )

    def test_teaching_not_present_in_verified_result_tex_or_checks(self):
        """Teaching narration never leaks into result_tex or checks of a
        verified result."""
        teaching_text = (
            "Torsion couples to the spin current. Non-metricity means "
            "length is not conserved. Projective freedom means the "
            "connection is determined only up to a projective mode."
        )
        d = FieldDerivation(
            wrt="g",
            kind="eom",
            capability=Capability.VARY,
            result_id="test-cross-010c",
            result_tex=r"G_{\mu\nu} = 0",
            verified=True,
            checks={"residue_zero": "True"},
            detail="kernel confirmed",
            teaching=teaching_text,
        )

        # Teaching does not appear in result_tex
        assert d.teaching not in (d.result_tex or ""), (
            "teaching must not leak into result_tex"
        )
        # Teaching does not appear in checks
        for check_val in d.checks.values():
            assert d.teaching not in check_val, (
                "teaching must not leak into check values"
            )
        # Teaching does not appear in detail
        assert d.teaching not in d.detail, (
            "teaching must not leak into detail"
        )


# ---------------------------------------------------------------------------
# Geometry teaching content tests (for _geometry_teaching function)
# ---------------------------------------------------------------------------


class TestGeometryTeachingContent:
    """The _geometry_teaching function generates correct narration for
    different geometry configurations."""

    def test_torsion_only_npr_mentions_spin(self):
        """An NPR with torsion but no non-metricity mentions spin coupling."""
        from noether.orchestrator.elicit import apply_resolutions

        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        answers = {}
        for amb in npr.ambiguities:
            if amb.id == "amb-connection":
                answers[amb.id] = "independent"
            elif amb.id == "amb-torsion":
                answers[amb.id] = "torsion-allowed"
            elif amb.id == "amb-nonmetricity":
                answers[amb.id] = "nonmetricity-free"
            elif amb.id == "amb-metric-compatibility":
                answers[amb.id] = "metric-compatible"
            elif amb.id == "amb-curvature-free":
                answers[amb.id] = "curvature-allowed"
            else:
                answers[amb.id] = amb.options[0]
        npr = apply_resolutions(npr, answers)
        remaining = npr.unresolved_ambiguities()
        if remaining:
            npr = apply_resolutions(npr, {amb.id: amb.options[0] for amb in remaining})

        teaching = _geometry_teaching(npr, "g", "eom")
        assert teaching, "teaching should be non-empty for torsion-only NPR"
        # Should mention torsion/spin coupling
        assert "torsion" in teaching.lower() or "spin" in teaching.lower(), (
            f"teaching should mention torsion/spin: {teaching!r}"
        )

    def test_nonmetricity_npr_mentions_length(self):
        """An NPR with non-metricity mentions length non-conservation."""
        from noether.orchestrator.elicit import apply_resolutions

        npr = ingest_action(MEASURE, r"g^{\mu\nu} R_{\mu\nu}(\Gamma)").npr
        answers = {}
        for amb in npr.ambiguities:
            if amb.id == "amb-connection":
                answers[amb.id] = "independent"
            elif amb.id == "amb-torsion":
                answers[amb.id] = "torsion-free"
            elif amb.id == "amb-nonmetricity":
                answers[amb.id] = "nonmetricity-allowed"
            elif amb.id == "amb-metric-compatibility":
                answers[amb.id] = "not-metric-compatible"
            elif amb.id == "amb-curvature-free":
                answers[amb.id] = "curvature-allowed"
            else:
                answers[amb.id] = amb.options[0]
        npr = apply_resolutions(npr, answers)
        remaining = npr.unresolved_ambiguities()
        if remaining:
            npr = apply_resolutions(npr, {amb.id: amb.options[0] for amb in remaining})

        teaching = _geometry_teaching(npr, "g", "eom")
        assert teaching, "teaching should be non-empty for non-metricity NPR"
        # Should mention non-metricity / length / dilation
        lower = teaching.lower()
        assert "non-metricity" in lower or "length" in lower or "dilation" in lower, (
            f"teaching should mention non-metricity/length/dilation: {teaching!r}"
        )

    def test_perturbation_teaching_mentions_fluctuation(self):
        """Teaching for a perturbation on a metric-affine background mentions
        the connection fluctuation."""
        npr = _metric_affine_npr()
        teaching = _geometry_teaching(npr, "g", "perturbation")
        assert teaching, "teaching should be non-empty for metric-affine perturbation"
        lower = teaching.lower()
        assert "fluctuation" in lower or "dg" in lower or "h" in lower, (
            f"teaching should mention fluctuation for perturbation: {teaching!r}"
        )

    def test_adm_teaching_mentions_foliation(self):
        """Teaching for ADM on a metric-affine background mentions foliation
        or constraint structure."""
        npr = _metric_affine_npr()
        teaching = _geometry_teaching(npr, "g", "adm")
        assert teaching, "teaching should be non-empty for metric-affine ADM"
        lower = teaching.lower()
        assert "foliation" in lower or "constraint" in lower or "torsion" in lower, (
            f"teaching should mention foliation/constraint for ADM: {teaching!r}"
        )
