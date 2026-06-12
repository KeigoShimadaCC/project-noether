"""ELICIT: the model PROPOSES, a human CONFIRMS.

This module keeps AGENTS.md rule 4 structural. `propose_resolutions` asks the
LLM to choose, for each open ambiguity, one of the listed options. It then
validates every suggestion against the allowed options and discards anything
off-menu (choice becomes None). It NEVER mutates the NPR and never sets a
resolution. Resolutions only take effect through `apply_resolutions`, which
takes human-confirmed choices. So an LLM, however persuasive, cannot make the
system plan on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from noether.llm.base import LLMAdapter, LLMError
from noether.npr.schema import NPR, Ambiguity

SYSTEM_PROMPT = (
    "You assist a physicist using Noether, a symbolic-physics tool. You do not "
    "make final decisions. For each tagged question, PROPOSE exactly one of the "
    "listed options and give a one-sentence rationale. A human confirms or "
    "overrides every choice. Never invent an option that is not listed. Respond "
    "with ONLY a JSON object mapping each question id to "
    '{"choice": <one listed option>, "rationale": <short string>}. '
    "Output no prose outside the JSON."
)


@dataclass
class ProposedResolution:
    ambiguity_id: str
    choice: str | None
    rationale: str = ""


@dataclass
class ElicitationProposal:
    proposals: list[ProposedResolution]
    llm_name: str
    llm_version: str
    raw: str = ""


def build_elicitation_prompt(npr: NPR, ambiguities: list[Ambiguity]) -> str:
    lines: list[str] = []
    lines.append(f"Conventions: {npr.conventions.id}")
    lines.append(f"Action: \\int {npr.action.measure_tex} ( {npr.action.lagrangian_tex} )")
    lines.append("Objects:")
    for obj in npr.objects:
        lines.append(f"  - {obj.name} ({obj.kind})")
    lines.append("Questions:")
    for amb in ambiguities:
        lines.append(f"  [{amb.id}] {amb.question}")
        lines.append(f"      options: {', '.join(amb.options)}")
    lines.append("Return a JSON object keyed by question id.")
    return "\n".join(lines)


def parse_llm_json(text: str) -> dict:
    """Extract the first top-level JSON object from possibly noisy model output."""
    import json

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise LLMError("no parseable JSON object found in LLM output")


def propose_resolutions(npr: NPR, llm: LLMAdapter) -> ElicitationProposal:
    """Ask the model to propose an option per open ambiguity. Pure suggestion:
    the returned NPR is unchanged and remains un-plannable."""
    unresolved = npr.unresolved_ambiguities()
    prompt = build_elicitation_prompt(npr, unresolved)
    raw = llm.complete(SYSTEM_PROMPT, prompt)
    parsed = parse_llm_json(raw)

    proposals: list[ProposedResolution] = []
    for amb in unresolved:
        entry = parsed.get(amb.id)
        choice: str | None = None
        rationale = ""
        if isinstance(entry, dict):
            rationale = str(entry.get("rationale", ""))
            candidate = entry.get("choice")
            if candidate in amb.options:  # off-menu suggestions are discarded
                choice = candidate
        proposals.append(ProposedResolution(amb.id, choice, rationale))

    return ElicitationProposal(
        proposals=proposals,
        llm_name=getattr(llm, "name", "unknown"),
        llm_version=llm.version(),
        raw=raw,
    )


def apply_resolutions(npr: NPR, confirmations: dict[str, str]) -> NPR:
    """Return a copy of `npr` with human-confirmed resolutions applied.

    Each confirmation must name a listed option for its ambiguity; an off-menu
    answer is a hard error, never a silent acceptance.
    """
    updated = npr.model_copy(deep=True)
    by_id = {amb.id: amb for amb in updated.ambiguities}
    for amb_id, choice in confirmations.items():
        if amb_id not in by_id:
            raise ValueError(f"no ambiguity {amb_id!r} in NPR")
        amb = by_id[amb_id]
        if amb.options and choice not in amb.options:
            raise ValueError(f"{choice!r} is not a listed option for {amb_id!r}: {amb.options}")
        amb.resolution = choice
    return updated
