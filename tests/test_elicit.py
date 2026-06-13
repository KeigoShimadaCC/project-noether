"""ELICIT: the model proposes, the human confirms (AGENTS.md rule 4).

The contract under test: proposing never mutates the NPR and never resolves an
ambiguity; off-menu suggestions are discarded; only apply_resolutions with
listed options unblocks planning. Driven by the deterministic StubLLMAdapter.
"""

import pytest

from evals import (
    eval1_eh_trace,
    eval2_palatini,
    eval3_scalar_tensor,
    eval4_maxwell,
    eval5_gauss_bonnet,
)
from noether.llm import StubLLMAdapter, stub_reply
from noether.orchestrator import build_plan, ingest_action
from noether.orchestrator.elicit import (
    apply_resolutions,
    build_elicitation_prompt,
    propose_resolutions,
)

ALL_EVALS = [
    eval1_eh_trace,
    eval2_palatini,
    eval3_scalar_tensor,
    eval4_maxwell,
    eval5_gauss_bonnet,
]


def _ingest(mod):
    action = mod.build_npr().action
    return ingest_action(action.measure_tex, action.lagrangian_tex).npr


def _first_option_answers(npr) -> dict[str, str]:
    return {amb.id: amb.options[0] for amb in npr.ambiguities}


class TestPromptBuilding:
    def test_prompt_mentions_every_question_and_the_action(self):
        npr = _ingest(eval3_scalar_tensor)
        prompt = build_elicitation_prompt(npr, npr.unresolved_ambiguities())
        for amb in npr.ambiguities:
            assert amb.id in prompt
            for opt in amb.options:
                assert opt in prompt
        assert npr.action.lagrangian_tex in prompt


class TestProposeIsPure:
    def test_proposing_does_not_resolve_anything(self):
        npr = _ingest(eval1_eh_trace)
        llm = StubLLMAdapter(stub_reply(_first_option_answers(npr)))
        proposal = propose_resolutions(npr, llm)
        assert all(p.choice is not None for p in proposal.proposals)
        # The NPR is untouched: still blocked.
        assert not npr.is_well_posed()
        assert npr.unresolved_ambiguities() == npr.ambiguities

    def test_off_menu_suggestion_is_discarded(self):
        npr = _ingest(eval4_maxwell)
        bogus = {amb.id: "totally-not-an-option" for amb in npr.ambiguities}
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(bogus)))
        assert all(p.choice is None for p in proposal.proposals)

    def test_records_model_provenance(self):
        npr = _ingest(eval1_eh_trace)
        proposal = propose_resolutions(npr, StubLLMAdapter(stub_reply(_first_option_answers(npr))))
        assert proposal.llm_name == "stub"
        assert proposal.llm_version == "stub-1"


class TestApplyResolutions:
    @pytest.mark.parametrize("mod", ALL_EVALS, ids=lambda m: m.__name__.split(".")[-1])
    def test_confirmed_proposals_unblock_planning(self, mod):
        npr = _ingest(mod)
        llm = StubLLMAdapter(stub_reply(_first_option_answers(npr)))
        proposal = propose_resolutions(npr, llm)
        confirmations = {p.ambiguity_id: p.choice for p in proposal.proposals if p.choice}
        confirmed = apply_resolutions(npr, confirmations)
        assert confirmed.is_well_posed()
        assert build_plan(confirmed).task_type == "vary"

    def test_off_menu_confirmation_rejected(self):
        npr = _ingest(eval4_maxwell)
        with pytest.raises(ValueError, match="not a listed option"):
            apply_resolutions(npr, {"amb-conventions": "nope"})

    def test_unknown_ambiguity_rejected(self):
        npr = _ingest(eval4_maxwell)
        with pytest.raises(ValueError, match="no ambiguity"):
            apply_resolutions(npr, {"amb-does-not-exist": "x"})

    def test_vary_wrt_confirmation_propagates_to_task(self):
        npr = _ingest(eval3_scalar_tensor)
        assert npr.task.with_respect_to == ["g", "phi"]
        confirmed = apply_resolutions(npr, {"amb-vary-wrt": "phi"})
        assert confirmed.task.with_respect_to == ["phi"]
        assert npr.task.with_respect_to == ["g", "phi"]  # input NPR untouched
