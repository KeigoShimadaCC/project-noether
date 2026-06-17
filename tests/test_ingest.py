"""INGEST: raw action LaTeX -> blocked NPR draft (noether.orchestrator.ingest).

The contract under test is the no-guessing guarantee (AGENTS.md rule 4): for
every one of the five acceptance actions, ingest produces a draft NPR whose
ambiguity ledger is open, so build_plan() refuses to plan. Ingest classifies
objects syntactically but assigns no physics meaning; the tests check the
detected object kinds and the questions raised, never a resolved answer.
"""

import pytest

from evals import (
    eval1_eh_trace,
    eval2_palatini,
    eval3_scalar_tensor,
    eval4_maxwell,
    eval5_gauss_bonnet,
)
from noether.orchestrator import AmbiguityBlocked, build_plan, ingest_action


def _ingest(mod):
    action = mod.build_npr().action
    return ingest_action(action.measure_tex, action.lagrangian_tex)


ALL_EVALS = [
    eval1_eh_trace,
    eval2_palatini,
    eval3_scalar_tensor,
    eval4_maxwell,
    eval5_gauss_bonnet,
]


class TestNoGuessing:
    @pytest.mark.parametrize("mod", ALL_EVALS, ids=lambda m: m.__name__.split(".")[-1])
    def test_ingest_blocks_planning(self, mod):
        result = _ingest(mod)
        assert not result.npr.is_well_posed()
        assert result.npr.unresolved_ambiguities() == result.npr.ambiguities
        with pytest.raises(AmbiguityBlocked):
            build_plan(result.npr)

    @pytest.mark.parametrize("mod", ALL_EVALS, ids=lambda m: m.__name__.split(".")[-1])
    def test_conventions_and_vary_always_asked(self, mod):
        ids = {a.id for a in _ingest(mod).npr.ambiguities}
        assert {"amb-conventions", "amb-vary-wrt"} <= ids

    @pytest.mark.parametrize("mod", ALL_EVALS, ids=lambda m: m.__name__.split(".")[-1])
    def test_metric_always_present(self, mod):
        names = {o.name: o.kind for o in _ingest(mod).npr.objects}
        assert names.get("g") == "metric"


class TestObjectDiscovery:
    def test_eval1_einstein_shorthand(self):
        kinds = {o.name: o.kind for o in _ingest(eval1_eh_trace).npr.objects}
        assert kinds == {"g": "metric", "G": "shorthand"}

    def test_eval3_functions_and_scalar(self):
        kinds = {o.name: o.kind for o in _ingest(eval3_scalar_tensor).npr.objects}
        assert kinds == {
            "g": "metric",
            "F": "function",
            "V": "function",
            "R": "shorthand",
            "phi": "scalar-field",
        }

    def test_eval4_field_strength_is_tensor_field(self):
        kinds = {o.name: o.kind for o in _ingest(eval4_maxwell).npr.objects}
        assert kinds == {"g": "metric", "F": "tensor-field"}

    def test_torsion_and_nonmetricity_are_geometric_shorthands_with_symmetry(self):
        npr = ingest_action(
            r"d^4x \sqrt{-g}",
            r"T^{\lambda}_{\mu\nu} T_{\lambda}^{\mu\nu}"
            r" + Q_{\lambda\mu\nu} Q^{\lambda\mu\nu}",
        ).npr

        torsion = npr.object_named("T")
        nonmetricity = npr.object_named("Q")

        assert torsion.kind == "shorthand"
        assert torsion.role == "shorthand"
        assert torsion.rank == 3
        assert torsion.symmetry == "antisymmetric"

        assert nonmetricity.kind == "shorthand"
        assert nonmetricity.role == "shorthand"
        assert nonmetricity.rank == 3
        assert nonmetricity.symmetry == "symmetric"


class TestAmbiguityShape:
    def test_eval1_composite_question_for_G(self):
        ids = {a.id for a in _ingest(eval1_eh_trace).npr.ambiguities}
        assert "amb-composite-G" in ids

    def test_eval2_connection_question_from_annotation(self):
        ids = {a.id for a in _ingest(eval2_palatini).npr.ambiguities}
        assert "amb-connection" in ids

    def test_eval3_coupling_questions_for_both_functions(self):
        ids = {a.id for a in _ingest(eval3_scalar_tensor).npr.ambiguities}
        assert {"amb-coupling-F", "amb-coupling-V"} <= ids

    def test_eval5_dimension_question_from_symbolic_measure(self):
        ids = {a.id for a in _ingest(eval5_gauss_bonnet).npr.ambiguities}
        assert "amb-dimension" in ids

    def test_eval4_minimal_ledger_has_no_spurious_questions(self):
        ids = {a.id for a in _ingest(eval4_maxwell).npr.ambiguities}
        assert ids == {"amb-conventions", "amb-vary-wrt"}


class TestKineticScalarShorthand:
    """A bare X is the convention-named kinetic shorthand of the scalar, not an
    independent field to vary (AGENTS.md section 5). The reading is still put to
    the human as amb-kinetic-X rather than silently assumed."""

    HORNDESKI = (
        r"d^4x \sqrt{-g}",
        r"G_2(\phi,X) + G_3(\phi,X)\Box\phi + G_4(\phi,X) R "
        r"- \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi)",
    )

    def test_x_is_a_shorthand_not_a_dynamical_field(self):
        npr = ingest_action(*self.HORNDESKI).npr
        x = npr.object_named("X")
        assert x.kind == "shorthand" and x.role == "shorthand"
        assert x.definition_tex and "nabla" in x.definition_tex

    def test_x_is_not_offered_as_a_vary_candidate(self):
        npr = ingest_action(*self.HORNDESKI).npr
        assert "X" not in npr.task.with_respect_to
        assert npr.task.with_respect_to == ["g", "phi"]

    def test_kinetic_question_raised_and_no_curvature_question_for_x(self):
        ids = {a.id for a in ingest_action(*self.HORNDESKI).npr.ambiguities}
        assert "amb-kinetic-X" in ids
        assert "amb-composite-X" not in ids

    def test_subscripted_couplings_are_functions(self):
        kinds = {o.name: o.kind for o in ingest_action(*self.HORNDESKI).npr.objects}
        assert kinds["G_2"] == "function"
        assert kinds["G_3"] == "function"
        assert kinds["G_4"] == "function"

    def test_x_without_a_scalar_stays_independent(self):
        # No dynamical scalar to anchor the kinetic reading: do not reclassify.
        npr = ingest_action(r"d^4x \sqrt{-g}", r"G(X) R").npr
        assert npr.object_named("X").kind == "scalar-field"
        assert "amb-kinetic-X" not in {a.id for a in npr.ambiguities}


class TestResolutionUnblocks:
    """Answering every question makes the draft plannable: the gate is the
    ledger, not a hard-coded refusal."""

    def test_resolved_ingest_plans(self):
        result = _ingest(eval4_maxwell)
        for amb in result.npr.ambiguities:
            amb.resolution = amb.options[0]
        plan = build_plan(result.npr)
        assert plan.task_type == "vary"
