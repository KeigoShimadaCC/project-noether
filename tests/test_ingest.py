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


class TestResolutionUnblocks:
    """Answering every question makes the draft plannable: the gate is the
    ledger, not a hard-coded refusal."""

    def test_resolved_ingest_plans(self):
        result = _ingest(eval4_maxwell)
        for amb in result.npr.ambiguities:
            amb.resolution = amb.options[0]
        plan = build_plan(result.npr)
        assert plan.task_type == "vary"
