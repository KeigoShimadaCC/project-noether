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
                "amb-curvature-free": "curvature-allowed",
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
                "amb-curvature-free": "curvature-allowed",
            },
        )

        reverted = apply_resolutions(independent, {"amb-connection": "levi-civita"})

        connection = reverted.geometry.connection
        assert connection.type == "levi-civita"
        assert connection.torsion is False
        assert connection.nonmetricity is False
        assert connection.metric_compatible is True
        assert connection.curvature_free is False
        assert connection.family == "riemannian"
        assert "amb-ricci-contraction" not in _question_ids(reverted)


class TestTeleparallelFamilyRouting:
    """Teleparallel and symmetric-teleparallel family classification from
    curvature_free + torsion/nonmetricity/metric-compatibility flags."""

    def test_teleparallel_family_set_when_curvature_free_and_torsionful(self):
        npr = ingest_action(MEASURE, "R").npr

        confirmed = apply_resolutions(
            npr,
            {
                "amb-connection": "independent",
                "amb-torsion": "torsion-allowed",
                "amb-nonmetricity": "nonmetricity-free",
                "amb-metric-compatibility": "metric-compatible",
                "amb-curvature-free": "curvature-free",
            },
        )

        connection = confirmed.geometry.connection
        assert connection.curvature_free is True
        assert connection.metric_compatible is True
        assert connection.torsion is True
        assert connection.nonmetricity is False
        assert connection.family == "teleparallel"

    def test_symmetric_teleparallel_family_set_when_curvature_free_and_nonmetric(self):
        npr = ingest_action(MEASURE, "R").npr

        confirmed = apply_resolutions(
            npr,
            {
                "amb-connection": "independent",
                "amb-torsion": "torsion-free",
                "amb-nonmetricity": "nonmetricity-allowed",
                "amb-metric-compatibility": "not-metric-compatible",
                "amb-curvature-free": "curvature-free",
            },
        )

        connection = confirmed.geometry.connection
        assert connection.curvature_free is True
        assert connection.metric_compatible is False
        assert connection.torsion is False
        assert connection.nonmetricity is True
        assert connection.family == "symmetric-teleparallel"

    def test_riemann_cartan_family_when_curvature_not_free(self):
        """Same flags as teleparallel but curvature_free=False gives riemann-cartan."""
        npr = ingest_action(MEASURE, "R").npr

        confirmed = apply_resolutions(
            npr,
            {
                "amb-connection": "independent",
                "amb-torsion": "torsion-allowed",
                "amb-nonmetricity": "nonmetricity-free",
                "amb-metric-compatibility": "metric-compatible",
                "amb-curvature-free": "curvature-allowed",
            },
        )

        connection = confirmed.geometry.connection
        assert connection.curvature_free is False
        assert connection.metric_compatible is True
        assert connection.torsion is True
        assert connection.family == "riemann-cartan"

    def test_metric_affine_family_when_curvature_free_but_mixed_flags(self):
        """Curvature-free but with both torsion and non-metricity: metric-affine."""
        npr = ingest_action(MEASURE, "R").npr

        confirmed = apply_resolutions(
            npr,
            {
                "amb-connection": "independent",
                "amb-torsion": "torsion-allowed",
                "amb-nonmetricity": "nonmetricity-allowed",
                "amb-metric-compatibility": "not-metric-compatible",
                "amb-curvature-free": "curvature-free",
            },
        )

        connection = confirmed.geometry.connection
        assert connection.curvature_free is True
        assert connection.family == "metric-affine"
