"""Geometry inference: the model proposes, the human confirms (VAL-GUIDE-001..007).

The contract: propose_resolutions returns one ProposedResolution per open
geometry ambiguity; every non-null choice is in that ambiguity's options;
off-menu suggestions yield choice=None (rationale may survive).  Proposing
never mutates the NPR.  Only apply_resolutions with an on-menu choice
mutates geometry.connection.  Off-menu and unknown-id confirmations raise
ValueError.  The HTTP surface mirrors this: /elicit returns confirmed:false
with proposals (off-menu nulled); /resolve enforces the menu (400 on
off-menu).  Driven deterministically with StubLLMAdapter.
"""

import pytest

from noether.llm import StubLLMAdapter, stub_reply
from noether.orchestrator.elicit import apply_resolutions, propose_resolutions
from noether.orchestrator.ingest import ingest_action

MEASURE = r"d^4x \sqrt{-g}"

# Actions that trigger the geometry questionnaire
CURVATURE_ACTION = "R"
PALATINI_ACTION = r"g^{\mu\nu} R_{\mu\nu}(\Gamma)"

# Actions that do NOT trigger the geometry questionnaire
SCALAR_ACTION = "X"

# Geometry ambiguity IDs
GEOMETRY_AMBIGUITY_IDS = {
    "amb-connection",
    "amb-torsion",
    "amb-nonmetricity",
    "amb-metric-compatibility",
    "amb-curvature-free",
}


def _geometry_ambiguities(npr):
    """Return the geometry ambiguities from the NPR's ambiguity list."""
    return [amb for amb in npr.ambiguities if amb.id in GEOMETRY_AMBIGUITY_IDS]


def _on_menu_geometry_answers(npr) -> dict[str, str]:
    """Build a dict of on-menu answers for all geometry ambiguities."""
    return {amb.id: amb.options[0] for amb in _geometry_ambiguities(npr)}


def _off_menu_geometry_answers(npr) -> dict[str, str]:
    """Build a dict of off-menu answers for all geometry ambiguities."""
    return {amb.id: "absolutely-not-an-option" for amb in _geometry_ambiguities(npr)}


def _all_on_menu_answers(npr) -> dict[str, str]:
    """Build a dict of on-menu answers for ALL ambiguities (geometry + others)."""
    return {amb.id: amb.options[0] for amb in npr.ambiguities}


# ---------------------------------------------------------------------------
# VAL-GUIDE-001: Geometry inference proposes only on-menu choices
# ---------------------------------------------------------------------------


class TestGeometryInferenceOnMenu:
    """VAL-GUIDE-001: propose_resolutions returns one ProposedResolution per
    open geometry ambiguity and every non-null choice is in that ambiguity's
    options."""

    def test_each_geometry_ambiguity_gets_a_proposal(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        geo_amb = _geometry_ambiguities(npr)
        assert geo_amb, "curvature action must raise geometry ambiguities"

        # Include non-geometry answers too so the LLM output is complete
        all_answers = _all_on_menu_answers(npr)
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        proposal_ids = {p.ambiguity_id for p in proposal.proposals}
        for amb in geo_amb:
            assert amb.id in proposal_ids, f"no proposal for {amb.id}"

    def test_every_non_null_choice_is_in_options(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        all_answers = _all_on_menu_answers(npr)
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        for p in proposal.proposals:
            if p.choice is not None:
                amb = next(a for a in npr.ambiguities if a.id == p.ambiguity_id)
                assert p.choice in amb.options, (
                    f"proposed choice {p.choice!r} not in {amb.id} options {amb.options}"
                )

    def test_on_menu_geometry_choices_are_returned_as_proposed(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        geo_answers = _on_menu_geometry_answers(npr)
        all_answers = _all_on_menu_answers(npr)
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        for p in proposal.proposals:
            if p.ambiguity_id in geo_answers:
                assert p.choice is not None, (
                    f"on-menu answer for {p.ambiguity_id} should not be nulled"
                )
                assert p.choice == geo_answers[p.ambiguity_id]

    def test_palatini_action_geometry_proposals_are_on_menu(self):
        npr = ingest_action(MEASURE, PALATINI_ACTION).npr
        all_answers = _all_on_menu_answers(npr)
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        for p in proposal.proposals:
            if p.ambiguity_id in GEOMETRY_AMBIGUITY_IDS and p.choice is not None:
                amb = next(a for a in npr.ambiguities if a.id == p.ambiguity_id)
                assert p.choice in amb.options


# ---------------------------------------------------------------------------
# VAL-GUIDE-002: An off-menu geometry suggestion is discarded
# ---------------------------------------------------------------------------


class TestOffMenuGeometrySuggestionDiscarded:
    """VAL-GUIDE-002: A stubbed off-menu choice yields choice is None for
    that ambiguity (rationale may survive)."""

    def test_off_menu_geometry_choice_is_nulled(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        # Mix: off-menu for geometry, on-menu for the rest
        mixed = _all_on_menu_answers(npr)
        mixed.update(_off_menu_geometry_answers(npr))
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(mixed)))

        for p in proposal.proposals:
            if p.ambiguity_id in GEOMETRY_AMBIGUITY_IDS:
                assert p.choice is None, (
                    f"off-menu choice for {p.ambiguity_id} should be None, got {p.choice!r}"
                )

    def test_off_menu_rationale_may_survive(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        off_menu = _off_menu_geometry_answers(npr)
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(off_menu)))

        for p in proposal.proposals:
            if p.ambiguity_id in GEOMETRY_AMBIGUITY_IDS:
                # Rationale is allowed but not required to survive
                # (the stub always returns "stub rationale")
                assert isinstance(p.rationale, str)

    def test_mixed_on_and_off_menu_geometry_choices(self):
        """When some geometry answers are on-menu and others off-menu,
        on-menu survives and off-menu is nulled."""
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        geo_amb = _geometry_ambiguities(npr)
        assert len(geo_amb) >= 2, "need at least two geometry ambiguities for mixed test"

        answers = _all_on_menu_answers(npr)
        # Make the first geometry ambiguity off-menu
        first_geo_id = geo_amb[0].id
        answers[first_geo_id] = "totally-wrong-option"

        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(answers)))

        for p in proposal.proposals:
            if p.ambiguity_id == first_geo_id:
                assert p.choice is None
            elif p.ambiguity_id in GEOMETRY_AMBIGUITY_IDS and p.choice is not None:
                amb = next(a for a in npr.ambiguities if a.id == p.ambiguity_id)
                assert p.choice in amb.options


# ---------------------------------------------------------------------------
# VAL-GUIDE-003: Inference is a suggestion, never auto-applied
# ---------------------------------------------------------------------------


class TestInferenceIsSuggestion:
    """VAL-GUIDE-003: After propose_resolutions (even all on-menu), the NPR
    is unchanged: is_well_posed()==False, geometry ambiguities unresolved,
    geometry.connection unchanged."""

    def test_propose_does_not_change_well_posedness(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        assert not npr.is_well_posed()

        all_answers = _all_on_menu_answers(npr)
        propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        assert not npr.is_well_posed()

    def test_propose_does_not_resolve_geometry_ambiguities(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        geo_before = {a.id: a.resolution for a in _geometry_ambiguities(npr)}
        assert all(v is None for v in geo_before.values())

        all_answers = _all_on_menu_answers(npr)
        propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        geo_after = {a.id: a.resolution for a in _geometry_ambiguities(npr)}
        assert geo_after == geo_before, "propose must not resolve geometry ambiguities"

    def test_propose_does_not_change_connection(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        connection_before = npr.geometry.connection.model_copy()

        all_answers = _all_on_menu_answers(npr)
        propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        assert npr.geometry.connection == connection_before, (
            "propose must not change geometry.connection"
        )

    def test_propose_all_on_menu_still_does_not_apply(self):
        """Even when the stub returns all on-menu choices, the NPR must not
        be mutated.  The no-guessing contract: inference is a suggestion."""
        npr = ingest_action(MEASURE, PALATINI_ACTION).npr
        connection_before = npr.geometry.connection.model_copy()

        geo_answers = {
            "amb-connection": "independent",
            "amb-torsion": "torsion-allowed",
            "amb-nonmetricity": "nonmetricity-allowed",
            "amb-metric-compatibility": "not-metric-compatible",
            "amb-curvature-free": "curvature-allowed",
        }
        all_answers = _all_on_menu_answers(npr)
        all_answers.update(geo_answers)
        propose_resolutions(npr, StubLLMAdapter(stub_reply(all_answers)))

        assert npr.geometry.connection == connection_before
        assert not npr.is_well_posed()
        for amb in _geometry_ambiguities(npr):
            assert amb.resolution is None


# ---------------------------------------------------------------------------
# VAL-GUIDE-004: The geometry question is raised for any
# curvature/connection action; a scalar action has none
# ---------------------------------------------------------------------------


class TestGeometryQuestionTrigger:
    """VAL-GUIDE-004: Ingesting any curvature/connection action yields at
    least one geometry ambiguity with a non-empty options menu; a scalar
    action does not."""

    def test_curvature_action_yields_geometry_ambiguity(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        geo = _geometry_ambiguities(npr)
        assert geo, "curvature action must raise at least one geometry ambiguity"
        for amb in geo:
            assert amb.options, f"{amb.id} must have a non-empty options menu"

    def test_connection_action_yields_geometry_ambiguity(self):
        npr = ingest_action(MEASURE, PALATINI_ACTION).npr
        geo = _geometry_ambiguities(npr)
        assert geo, "connection action must raise at least one geometry ambiguity"
        # The connection action should place "independent" first in options
        conn_amb = next(a for a in geo if a.id == "amb-connection")
        assert "independent" in conn_amb.options

    def test_scalar_action_has_no_geometry_ambiguity(self):
        npr = ingest_action(MEASURE, SCALAR_ACTION).npr
        geo = _geometry_ambiguities(npr)
        assert not geo, "scalar action must not raise geometry ambiguities"

    def test_scalar_action_defaults_to_levi_civita(self):
        npr = ingest_action(MEASURE, SCALAR_ACTION).npr
        assert npr.geometry.connection.type == "levi-civita"

    def test_torsion_action_yields_geometry_ambiguity(self):
        npr = ingest_action(MEASURE, r"T^{\lambda}_{\mu\nu} T_{\lambda}^{\mu\nu}").npr
        geo = _geometry_ambiguities(npr)
        assert geo, "action with explicit torsion must raise geometry ambiguities"

    def test_nonmetricity_action_yields_geometry_ambiguity(self):
        npr = ingest_action(MEASURE, r"Q_{\lambda\mu\nu} Q^{\lambda\mu\nu}").npr
        geo = _geometry_ambiguities(npr)
        assert geo, "action with explicit non-metricity must raise geometry ambiguities"


# ---------------------------------------------------------------------------
# VAL-GUIDE-005: Only a human-confirmed answer mutates the geometry
# ---------------------------------------------------------------------------


class TestGeometryMutationOnlyOnConfirm:
    """VAL-GUIDE-005: geometry.connection changes only through
    apply_resolutions/resolve with an on-menu choice; propose_resolutions
    leaves it untouched and the input NPR is unchanged."""

    def test_propose_does_not_mutate_connection(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        original_connection = npr.geometry.connection.model_copy()
        assert original_connection.type == "levi-civita"

        answers = {**_all_on_menu_answers(npr), "amb-connection": "independent"}
        propose_resolutions(npr, StubLLMAdapter(stub_reply(answers)))

        assert npr.geometry.connection == original_connection

    def test_apply_on_menu_changes_connection(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        assert npr.geometry.connection.type == "levi-civita"

        confirmed = apply_resolutions(npr, {"amb-connection": "independent"})

        assert confirmed.geometry.connection.type == "independent"
        assert npr.geometry.connection.type == "levi-civita"  # input unchanged

    def test_apply_on_menu_torsion_changes_connection(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr

        confirmed = apply_resolutions(npr, {
            "amb-connection": "independent",
            "amb-torsion": "torsion-allowed",
        })

        assert confirmed.geometry.connection.torsion is True
        assert npr.geometry.connection.torsion is False  # input unchanged

    def test_full_geometry_resolution_through_apply(self):
        npr = ingest_action(MEASURE, PALATINI_ACTION).npr

        confirmed = apply_resolutions(npr, {
            "amb-connection": "independent",
            "amb-torsion": "torsion-allowed",
            "amb-nonmetricity": "nonmetricity-allowed",
            "amb-metric-compatibility": "not-metric-compatible",
            "amb-curvature-free": "curvature-allowed",
        })

        c = confirmed.geometry.connection
        assert c.type == "independent"
        assert c.torsion is True
        assert c.nonmetricity is True
        assert c.metric_compatible is False
        # Input NPR untouched
        assert npr.geometry.connection.type == "levi-civita"


# ---------------------------------------------------------------------------
# VAL-GUIDE-006: An off-menu confirmation is a hard error, never silent
# ---------------------------------------------------------------------------


class TestOffMenuConfirmationRejected:
    """VAL-GUIDE-006: apply_resolutions with an off-menu geometry value raises
    ValueError ("not a listed option"); an unknown ambiguity id raises; the
    NPR is never mutated."""

    def test_off_menu_connection_value_raises(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        with pytest.raises(ValueError, match="not a listed option"):
            apply_resolutions(npr, {"amb-connection": "non-metric-affine"})

    def test_off_menu_torsion_value_raises(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        with pytest.raises(ValueError, match="not a listed option"):
            apply_resolutions(npr, {"amb-torsion": "torsion-everywhere"})

    def test_unknown_geometry_ambiguity_id_raises(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        with pytest.raises(ValueError, match="no ambiguity"):
            apply_resolutions(npr, {"amb-spin-connection": "SO(3)"})

    def test_npr_never_mutated_on_rejection(self):
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        connection_before = npr.geometry.connection.model_copy()

        with pytest.raises(ValueError):
            apply_resolutions(npr, {"amb-connection": "wrong"})
        assert npr.geometry.connection == connection_before

        with pytest.raises(ValueError):
            apply_resolutions(npr, {"amb-phantom": "ghost"})
        assert npr.geometry.connection == connection_before

    def test_off_menu_ricci_contraction_raises(self):
        """The Ricci contraction ambiguity (opened after connection=independent)
        also rejects off-menu answers."""
        npr = ingest_action(MEASURE, CURVATURE_ACTION).npr
        independent = apply_resolutions(npr, {"amb-connection": "independent"})

        with pytest.raises(ValueError, match="not a listed option"):
            apply_resolutions(independent, {"amb-ricci-contraction": "second-fourth"})


# ---------------------------------------------------------------------------
# VAL-GUIDE-007: HTTP elicit proposes without applying; resolve enforces
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from noether.orchestrator.store import SessionStore  # noqa: E402
from noether.server import create_app  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    return SessionStore(tmp_path / "sessions")


@pytest.fixture()
def client(store):
    return TestClient(create_app(store=store))


def _create_session(client, lagrangian="R"):
    response = client.post("/sessions", json={"lagrangian": lagrangian})
    assert response.status_code == 201, response.text
    return response.json()


class TestHttpGeometryInference:
    """VAL-GUIDE-007: POST /elicit returns confirmed:false with proposals
    (off-menu nulled) and the ambiguity stays unresolved; mutation happens
    only after an on-menu POST /resolve; off-menu resolve is 400."""

    def test_elicit_returns_unconfirmed_with_geometry_proposals(self, store):
        npr = ingest_action(MEASURE, PALATINI_ACTION).npr
        answers = {a.id: a.options[0] for a in npr.ambiguities}
        client = TestClient(create_app(store=store, llm=StubLLMAdapter(stub_reply(answers))))

        body = _create_session(client, PALATINI_ACTION)
        response = client.post(f"/sessions/{body['session_id']}/elicit")
        assert response.status_code == 200

        payload = response.json()
        assert payload["confirmed"] is False

        # Proposals exist for geometry ambiguities
        proposal_ids = {p["ambiguity_id"] for p in payload["proposals"]}
        assert GEOMETRY_AMBIGUITY_IDS & proposal_ids, (
            "proposals must include geometry ambiguities"
        )

    def test_elicit_off_menu_geometry_proposals_are_nulled(self, store):
        npr = ingest_action(MEASURE, PALATINI_ACTION).npr
        # Build answers where geometry choices are off-menu
        answers = {a.id: "bogus-option" for a in npr.ambiguities}
        client = TestClient(create_app(store=store, llm=StubLLMAdapter(stub_reply(answers))))

        body = _create_session(client, PALATINI_ACTION)
        response = client.post(f"/sessions/{body['session_id']}/elicit")
        assert response.status_code == 200

        payload = response.json()
        for p in payload["proposals"]:
            if p["ambiguity_id"] in GEOMETRY_AMBIGUITY_IDS:
                assert p["choice"] is None, (
                    f"off-menu geometry proposal for {p['ambiguity_id']} must be nulled"
                )

    def test_elicit_does_not_resolve_ambiguities(self, store):
        npr = ingest_action(MEASURE, PALATINI_ACTION).npr
        answers = {a.id: a.options[0] for a in npr.ambiguities}
        client = TestClient(create_app(store=store, llm=StubLLMAdapter(stub_reply(answers))))

        body = _create_session(client, PALATINI_ACTION)
        client.post(f"/sessions/{body['session_id']}/elicit")

        after = client.get(f"/sessions/{body['session_id']}").json()
        assert after["well_posed"] is False
        for q in after["questions"]:
            if q["id"] in GEOMETRY_AMBIGUITY_IDS:
                assert q["resolution"] is None, (
                    f"elicit must not resolve {q['id']}"
                )

    def test_resolve_on_menu_applies_geometry(self, store):
        client = TestClient(create_app(store=store))
        body = _create_session(client, PALATINI_ACTION)
        sid = body["session_id"]

        response = client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": {
                "amb-conventions": "noether-default-v1",
                "amb-vary-wrt": "g and Gamma",
                "amb-connection": "independent",
                "amb-torsion": "torsion-allowed",
                "amb-nonmetricity": "nonmetricity-allowed",
                "amb-metric-compatibility": "not-metric-compatible",
                "amb-curvature-free": "curvature-allowed",
            }},
        )
        assert response.status_code == 200

        session = store.get(sid)
        c = session.npr.geometry.connection
        assert c.type == "independent"
        assert c.torsion is True
        assert c.nonmetricity is True
        assert c.metric_compatible is False

    def test_resolve_off_menu_geometry_is_400(self, client):
        body = _create_session(client, PALATINI_ACTION)
        sid = body["session_id"]

        response = client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": {"amb-connection": "teleparallel"}},
        )
        assert response.status_code == 400
        assert "not a listed option" in response.json()["detail"]

    def test_resolve_off_menu_torsion_is_400(self, client):
        body = _create_session(client, PALATINI_ACTION)
        sid = body["session_id"]

        response = client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": {"amb-torsion": "torsion-dominated"}},
        )
        assert response.status_code == 400

    def test_off_menu_resolve_leaves_ambiguity_unresolved(self, client):
        body = _create_session(client, PALATINI_ACTION)
        sid = body["session_id"]

        # Attempt off-menu resolve
        client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": {"amb-connection": "metric-affine"}},
        )

        after = client.get(f"/sessions/{sid}").json()
        conn_q = next(q for q in after["questions"] if q["id"] == "amb-connection")
        assert conn_q["resolution"] is None

    def test_elicit_then_resolve_full_flow(self, store):
        """End-to-end: elicit proposes, resolve confirms with on-menu."""
        npr = ingest_action(MEASURE, PALATINI_ACTION).npr
        answers = {a.id: a.options[0] for a in npr.ambiguities}
        # Force specific geometry answers
        answers["amb-connection"] = "independent"
        answers["amb-torsion"] = "torsion-allowed"
        answers["amb-nonmetricity"] = "nonmetricity-allowed"
        answers["amb-metric-compatibility"] = "not-metric-compatible"
        answers["amb-curvature-free"] = "curvature-allowed"

        client = TestClient(create_app(store=store, llm=StubLLMAdapter(stub_reply(answers))))
        body = _create_session(client, PALATINI_ACTION)
        sid = body["session_id"]

        # Step 1: elicit
        elicit_resp = client.post(f"/sessions/{sid}/elicit")
        assert elicit_resp.json()["confirmed"] is False

        # Step 2: confirm on-menu
        confirm = {
            p["ambiguity_id"]: p["choice"]
            for p in elicit_resp.json()["proposals"]
            if p["choice"]
        }
        resolve_resp = client.post(
            f"/sessions/{sid}/resolve",
            json={"resolutions": confirm},
        )
        assert resolve_resp.status_code == 200

        # Step 3: verify geometry mutated
        session = store.get(sid)
        c = session.npr.geometry.connection
        assert c.type == "independent"

    def test_scalar_action_no_geometry_in_elicit(self, store):
        """A scalar action has no geometry ambiguities, so /elicit proposals
        contain no geometry entries."""
        npr = ingest_action(MEASURE, SCALAR_ACTION).npr
        answers = {a.id: a.options[0] for a in npr.ambiguities}
        client = TestClient(create_app(store=store, llm=StubLLMAdapter(stub_reply(answers))))

        body = _create_session(client, SCALAR_ACTION)
        response = client.post(f"/sessions/{body['session_id']}/elicit")
        assert response.status_code == 200

        payload = response.json()
        proposal_ids = {p["ambiguity_id"] for p in payload["proposals"]}
        assert not (GEOMETRY_AMBIGUITY_IDS & proposal_ids), (
            "scalar action must not produce geometry proposals"
        )
