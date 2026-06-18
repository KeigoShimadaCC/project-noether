"""ELICIT: the model PROPOSES, a human CONFIRMS.

This module keeps AGENTS.md rule 4 structural. `propose_resolutions` asks the
LLM to choose, for each open ambiguity, one of the listed options. It then
validates every suggestion against the allowed options and discards anything
off-menu (choice becomes None). It NEVER mutates the NPR and never sets a
resolution. Resolutions only take effect through `apply_resolutions`, which
takes human-confirmed choices. So an LLM, however persuasive, cannot make the
system plan on its own.

The inference prompt embeds the action's geometric cues (VAL-GUIDE-017) so
the model's proposed geometry choices are grounded in the action, not a fixed
default. A scalar action carries no such geometry cue. Convention proposals
(Ricci-contraction, field-strength definition) are on-menu with rationale,
never auto-applied (VAL-GUIDE-020).
"""

from __future__ import annotations

from dataclasses import dataclass

from noether.llm.base import LLMAdapter, LLMError
from noether.npr.ast import Deriv, Expr, Func, Num, Pow, Prod, Sum, Sym, Tensor
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


@dataclass
class _GeometricCues:
    """Structural cues extracted from the action's AST that ground geometry
    inference in the action's actual content, not a fixed default."""

    has_curvature: bool = False
    has_connection: bool = False
    has_torsion: bool = False
    has_nonmetricity: bool = False
    has_curvature_free_cue: bool = False  # T or Q present, R absent
    # f(Q)/f(T) family detection: a function of Q or T (e.g. f(Q), V(T))
    has_fq_family: bool = False
    has_ft_family: bool = False


_CURVATURE_NAMES = {"R", "G", "C", "W"}
_EXPLICIT_CONNECTION_NAMES = {"Gamma"}


def _detect_geometric_cues(expr: Expr) -> _GeometricCues:
    """Walk the action AST and detect structural geometric cues.

    These cues ground the model's proposed geometry choices in the action
    rather than a fixed default (VAL-GUIDE-017).
    """
    cues = _GeometricCues()

    def walk(node: Expr) -> None:
        match node:
            case Num() | Sym():
                return
            case Func(name=name, args=args):
                # Detect f(Q)/f(T) family: any function with Q or T as argument
                arg_names = set()
                for arg in args:
                    match arg:
                        case Sym(name=n):
                            arg_names.add(n)
                        case Tensor(name=n):
                            arg_names.add(n)
                        case _:
                            pass
                if "Q" in arg_names:
                    cues.has_fq_family = True
                if "T" in arg_names:
                    cues.has_ft_family = True
                for arg in args:
                    walk(arg)
            case Tensor(name=name, connection=connection):
                cues.has_curvature |= name in _CURVATURE_NAMES
                cues.has_connection |= name in _EXPLICIT_CONNECTION_NAMES or connection is not None
                cues.has_torsion |= name == "T"
                cues.has_nonmetricity |= name == "Q"
            case Deriv(expr=inner, connection=connection):
                cues.has_connection |= connection not in (None, "metric")
                walk(inner)
            case Pow(base=base):
                walk(base)
            case Prod(factors=factors):
                for factor in factors:
                    walk(factor)
            case Sum(terms=terms):
                for term in terms:
                    walk(term)
            case _:
                raise TypeError(f"unhandled expr node {node!r}")

    walk(expr)
    # Curvature-free cue: torsion or non-metricity present, curvature absent.
    cues.has_curvature_free_cue = (
        (cues.has_torsion or cues.has_nonmetricity) and not cues.has_curvature
    )
    return cues


def _geometry_cue_text(npr: NPR) -> str:
    """Produce a human-readable description of the geometric cues found in
    the action, or an empty string if no cues are present (scalar control).

    The cue text is embedded in the inference prompt so the model's proposed
    geometry choices are grounded in the action, not a fixed default
    (VAL-GUIDE-017).
    """
    cues = _detect_geometric_cues(npr.action.lagrangian)

    # No cues at all: scalar action or action with no geometric content.
    if not any(
        (
            cues.has_curvature,
            cues.has_connection,
            cues.has_torsion,
            cues.has_nonmetricity,
            cues.has_curvature_free_cue,
            cues.has_fq_family,
            cues.has_ft_family,
        )
    ):
        return ""

    lines: list[str] = []
    lines.append("Geometric cues from the action:")

    if cues.has_curvature and cues.has_connection:
        lines.append(
            "  - The action contains curvature built from an independent "
            "connection R(\\Gamma); the curvature depends on the connection, "
            "not the metric alone."
        )
    elif cues.has_curvature:
        lines.append(
            "  - The action contains curvature (R); the geometry may be "
            "Levi-Civita or use an independent connection."
        )

    if cues.has_connection and not cues.has_curvature:
        lines.append(
            "  - The action carries an explicit connection (\\Gamma) without "
            "curvature; consider whether this is an independent connection."
        )

    if cues.has_torsion:
        lines.append(
            "  - The action uses explicit torsion T; the connection may carry "
            "torsion (spin coupling, teleparallel geometry)."
        )

    if cues.has_nonmetricity:
        lines.append(
            "  - The action uses explicit non-metricity Q; the connection may "
            "be non-metric-compatible (length non-conservation, symmetric "
            "teleparallel geometry)."
        )

    if cues.has_curvature_free_cue:
        lines.append(
            "  - The action uses torsion or non-metricity but no curvature "
            "tensor, suggesting a curvature-free (teleparallel or symmetric "
            "teleparallel) geometry."
        )

    if cues.has_ft_family:
        lines.append(
            "  - The action is in the f(T) family: a function of the torsion "
            "scalar T, characteristic of metric teleparallel gravity."
        )

    if cues.has_fq_family:
        lines.append(
            "  - The action is in the f(Q) family: a function of the "
            "non-metricity scalar Q, characteristic of symmetric teleparallel "
            "gravity."
        )

    return "\n".join(lines)


def build_elicitation_prompt(npr: NPR, ambiguities: list[Ambiguity]) -> str:
    lines: list[str] = []
    lines.append(f"Conventions: {npr.conventions.id}")
    lines.append(f"Action: \\int {npr.action.measure_tex} ( {npr.action.lagrangian_tex} )")

    # Embed geometric cues so the model's proposals are grounded in the
    # action (VAL-GUIDE-017). Empty string for scalar actions.
    cue_text = _geometry_cue_text(npr)
    if cue_text:
        lines.append(cue_text)

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
    from noether.orchestrator.resolutions import propagate_resolution

    updated = npr.model_copy(deep=True)
    by_id = {amb.id: amb for amb in updated.ambiguities}
    for amb_id, choice in confirmations.items():
        if amb_id not in by_id:
            raise ValueError(f"no ambiguity {amb_id!r} in NPR")
        amb = by_id[amb_id]
        if amb.options and choice not in amb.options:
            raise ValueError(f"{choice!r} is not a listed option for {amb_id!r}: {amb.options}")
        amb.resolution = choice
        propagate_resolution(updated, amb)
    return updated
