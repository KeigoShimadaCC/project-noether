"""Resolution propagation onto the NPR and its dynamic geometry follow-ups."""

import pytest

from noether.orchestrator.elicit import apply_resolutions
from noether.orchestrator.ingest import ingest_action

MEASURE = r"d^4x \sqrt{-g}"


def _question_ids(npr) -> set[str]:
    return {amb.id for amb in npr.ambiguities}


class TestGeometryResolutionPropagation:
    def test_connection_resolution_sets_independent_type_and_surfaces_ricci_question(self):
        npr = ingest_action(MEASURE, "R").npr

        confirmed = apply_resolutions(npr, {"amb-connection": "independent"})

        assert confirmed.geometry.connection.type == "independent"
        assert confirmed.geometry.connection.family == "metric-affine"
        ricci = next(amb for amb in confirmed.ambiguities if amb.id == "amb-ricci-contraction")
        assert ricci.resolution is None
        assert len(ricci.options) > 1
        assert "first-third" in ricci.options
        assert "first-fourth" in ricci.options
        assert "amb-ricci-contraction" not in _question_ids(npr)

    def test_geometry_answers_update_connection_flags(self):
        npr = ingest_action(MEASURE, "R").npr

        confirmed = apply_resolutions(
            npr,
            {
                "amb-connection": "independent",
                "amb-torsion": "torsion-allowed",
                "amb-nonmetricity": "nonmetricity-allowed",
                "amb-metric-compatibility": "not-metric-compatible",
            },
        )

        connection = confirmed.geometry.connection
        assert connection.type == "independent"
        assert connection.torsion is True
        assert connection.nonmetricity is True
        assert connection.metric_compatible is False

    def test_off_menu_geometry_answer_raises_and_leaves_input_npr_unchanged(self):
        npr = ingest_action(MEASURE, "R").npr

        with pytest.raises(ValueError, match="not a listed option"):
            apply_resolutions(npr, {"amb-connection": "metric-affine"})

        assert npr.geometry.connection.type == "levi-civita"
        assert npr.geometry.connection.torsion is False
        assert npr.geometry.connection.nonmetricity is False
        assert "amb-ricci-contraction" not in _question_ids(npr)

    def test_reresolving_back_to_levi_civita_resets_dependent_flags(self):
        npr = ingest_action(MEASURE, "R").npr
        independent = apply_resolutions(
            npr,
            {
                "amb-connection": "independent",
                "amb-torsion": "torsion-allowed",
                "amb-nonmetricity": "nonmetricity-allowed",
                "amb-metric-compatibility": "not-metric-compatible",
            },
        )

        reverted = apply_resolutions(independent, {"amb-connection": "levi-civita"})

        connection = reverted.geometry.connection
        assert connection.type == "levi-civita"
        assert connection.torsion is False
        assert connection.nonmetricity is False
        assert connection.metric_compatible is True
        assert connection.family == "riemannian"
        assert "amb-ricci-contraction" not in _question_ids(reverted)
