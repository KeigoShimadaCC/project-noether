"""Readability shorthand proposals (good form, docs/03_METHODOLOGY.md).

When an action couples through a function of a field, F(phi), its derivatives
appear all over the variation and the equations of motion. Writing them out as
dF/dphi every time is correct but noisy; physicists shorthand them as F_phi,
F_phiphi, and so on. This module PROPOSES those shorthands so the human can
adopt cleaner notation.

What this is and is not:

  - These are DEFINITIONS, not computed results. `F_phi` is introduced as a
    name for the derivative partial F / partial phi; nothing here claims what
    the variation of any particular action evaluates to. So this stays inside
    AGENTS.md rule 1: no asserted result without a kernel, because there is no
    asserted result, only notation.
  - Like every other notation choice, adoption is the human's call (rule 4).
    propose_definitions never mutates the NPR; Session.add_definition applies
    only what the human accepts, as a new immutable NPR version.

Proposals are deterministic functions of the declared function couplings and
their arguments: for each non-constant function coupling we offer the first
derivative in each argument and the (diagonal) second derivative. Couplings
the human has already pinned to a constant are skipped, since their
derivatives vanish. Symbols already declared are never re-proposed.
"""

from __future__ import annotations

from dataclasses import dataclass

from noether.npr.schema import NPR, ObjectDecl

# Greek argument names render as symbols in subscripts and operators.
_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
    "upsilon", "phi", "chi", "psi", "omega",
}  # fmt: skip


def _sym(name: str) -> str:
    return f"\\{name}" if name in _GREEK else name


@dataclass(frozen=True)
class DefinitionProposal:
    """A proposed readability shorthand. `symbol` is the NPR object name;
    `symbol_tex` and `meaning_tex` render the definition for the human."""

    id: str
    symbol: str
    symbol_tex: str
    meaning_tex: str
    rationale: str

    def as_object(self) -> ObjectDecl:
        return ObjectDecl(
            name=self.symbol,
            kind="shorthand",
            role="shorthand",
            rank=0,
            definition_tex=f"{self.symbol_tex} \\equiv {self.meaning_tex}",
        )


def _coupling_is_constant(npr: NPR, func_name: str) -> bool:
    for amb in npr.ambiguities:
        if amb.id == f"amb-coupling-{func_name}":
            return amb.resolution == "constant"
    return False


def propose_definitions(npr: NPR) -> list[DefinitionProposal]:
    """Propose derivative shorthands for each non-constant function coupling.
    Pure: the NPR is not modified. Symbols already declared are skipped, so
    calling this repeatedly converges (accepted proposals stop reappearing)."""
    existing = {obj.name for obj in npr.objects}
    proposals: list[DefinitionProposal] = []
    for obj in npr.objects:
        if obj.kind != "function" or not obj.args:
            continue
        if _coupling_is_constant(npr, obj.name):
            continue
        f_tex = _sym(obj.name)
        for arg in obj.args:
            a_tex = _sym(arg)
            first = f"{obj.name}_{arg}"
            if first not in existing:
                proposals.append(
                    DefinitionProposal(
                        id=f"def-{obj.name}-{arg}",
                        symbol=first,
                        symbol_tex=f"{f_tex}_{{{a_tex}}}",
                        meaning_tex=f"\\frac{{\\partial {f_tex}}}{{\\partial {a_tex}}}",
                        rationale=(
                            f"notation for the first derivative of {f_tex} in {a_tex}, "
                            f"which appears whenever {f_tex} is varied or differentiated"
                        ),
                    )
                )
                existing.add(first)
            second = f"{obj.name}_{arg}{arg}"
            if second not in existing:
                proposals.append(
                    DefinitionProposal(
                        id=f"def-{obj.name}-{arg}{arg}",
                        symbol=second,
                        symbol_tex=f"{f_tex}_{{{a_tex}{a_tex}}}",
                        meaning_tex=f"\\frac{{\\partial^2 {f_tex}}}{{\\partial {a_tex}^2}}",
                        rationale=(
                            f"notation for the second derivative of {f_tex} in {a_tex}, "
                            "which appears in the linearized equations of motion"
                        ),
                    )
                )
                existing.add(second)
    return proposals
