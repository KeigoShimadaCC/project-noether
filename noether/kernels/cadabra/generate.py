"""Parameterize an audited Cadabra derivation scaffold with the LLM.

This is the general-derivation counterpart to the frozen golden templates in
`templates.py`. For an arbitrary (well-posed) action there is no pre-written
script, so the model writes one. Crucially, the model has NO authority over
the answer (AGENTS.md rules 1, 3):

  - it emits only a Cadabra *script*, never a "remembered" field equation;
  - the script must DERIVE the equation of motion by `vary()` from the action
    AND state an independent candidate `target`, then let the kernel compute
    the residue and report `residue_zero`;
  - `derive.py` trusts the result only when the kernel reports
    `residue_zero=True`. A script that cannot make the residue vanish yields
    an UNVERIFIED result, which is surfaced as such, never as truth.

The audited frozen templates double as in-context examples, so the model has
a concrete, golden-tested pattern to follow rather than inventing the dialect.
"""

from __future__ import annotations

from dataclasses import dataclass

from noether.kernels.cadabra import templates
from noether.llm.base import LLMAdapter
from noether.npr.schema import NPR

# One golden template per derivation capability, used as the worked example in
# the prompt. These are frozen and golden-tested (see tests/test_cadabra_*).
_EXAMPLE_TEMPLATE: dict[str, str] = {
    "vary-metric": "eval3_scalar_tensor_metric",
    "vary-scalar": "eval3_scalar_tensor_scalar",
    "vary-gauge": "eval4_maxwell",
}

CADABRA_CONTRACT = r"""You are a Cadabra2 scripting backend for Noether, a symbolic-physics tool.
Your ONLY output is a complete Cadabra2 script. No prose, no markdown fences.

You never assert a field equation from memory. The script must DERIVE the
equation of motion from the action and let the kernel check it. Specifically:

1. Reproduce the declaration block exactly as in the worked example
   (Indices, Integer range, Coordinate, Derivative, Metric/InverseMetric,
   KroneckerDelta, the sqrt(-g) shorthand `sg`, and a `::Depends(\nabla{#})`
   line listing every differentiated symbol). Add antisymmetry/symmetry
   property declarations for any field that has them.
2. Build the integrand:  ex := \int{ <action integrand in sg, g, R, fields> }{x};
   using `sg` for sqrt(-g) and the listed curvature/field symbols.
3. Derive delta S / delta(<field>) with a single `vary(ex, $...$)` call using
   the standard variation rules, then `distribute`, `product_rule`,
   metric-compatibility substitutions `\nabla_{\mu}{g...} -> 0`,
   `integrate_by_parts`, `canonicalise`, `rename_dummies`, and finally
   print("NOETHER_RESULT: " + str(ex)).
4. State the candidate equation of motion INDEPENDENTLY as `target := ...`
   built from curvature tensors and field derivatives (do NOT set it equal to
   `ex`), put it in the same canonical form, then:
       residue := @(ex) - @(target);
       distribute(residue); canonicalise(residue); rename_dummies(residue); meld(residue);
       print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
   If your derivation and your candidate agree, the kernel will report True;
   that, not your say-so, is what makes the result trusted.

Metric variation rules (mostly-plus, noether-default-v1):
  g^{\alpha\beta} -> -h^{\alpha\beta},  sg -> 1/2 sg g^{\mu\nu} h_{\mu\nu},
  R_{\alpha\beta} -> \nabla_{\lambda}{dGamma^{\lambda}_{\beta\alpha}}
                     - \nabla_{\beta}{dGamma^{\lambda}_{\lambda\alpha}},
  dGamma^{\lambda}_{\nu\sigma} -> 1/2 g^{\lambda\rho}(
       \nabla_{\nu}{h_{\rho\sigma}} + \nabla_{\sigma}{h_{\rho\nu}}
       - \nabla_{\rho}{h_{\nu\sigma}} ).
Scalar variation rules: \phi -> dphi, and for any coupling C(\phi): C -> Cp dphi.
Output ONLY the script."""


@dataclass
class GeneratedScript:
    source: str
    variation_key: str
    llm_name: str
    llm_version: str
    raw: str


def _variation_key(npr: NPR, wrt: str) -> str:
    """Pick the worked example that best matches the field being varied."""
    by_name = {obj.name: obj for obj in npr.objects}
    obj = by_name.get(wrt)
    if obj is None:
        return "vary-metric"
    if obj.kind == "metric":
        return "vary-metric"
    if obj.kind == "scalar-field":
        return "vary-scalar"
    if obj.kind == "tensor-field":
        return "vary-gauge"
    return "vary-metric"


def build_generation_prompt(npr: NPR, wrt: str) -> tuple[str, str]:
    """Return (system, prompt) for generating a variation script wrt `wrt`."""
    key = _variation_key(npr, wrt)
    example = templates.get(_EXAMPLE_TEMPLATE[key])
    objs = "\n".join(f"  - {o.name} ({o.kind}, {o.role}, rank {o.rank})" for o in npr.objects)
    prompt = (
        f"Conventions: {npr.conventions.id} (dimension {npr.conventions.dimension}, "
        f"signature {npr.conventions.signature}).\n"
        f"Action: S = \\int {npr.action.measure_tex} \\, ( {npr.action.lagrangian_tex} )\n"
        f"Objects:\n{objs}\n"
        f"Task: derive the equation of motion delta S / delta {wrt} = 0 "
        f"(vary with respect to {wrt}).\n\n"
        f"Worked example for this kind of variation (follow its structure):\n"
        f"-----\n{example}\n-----\n"
        f"Now write the script for the action above, varying with respect to {wrt}."
    )
    return CADABRA_CONTRACT, prompt


def strip_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def generate_script(npr: NPR, wrt: str, llm: LLMAdapter) -> GeneratedScript:
    """Ask the model to write a Cadabra variation script for `wrt`.

    Pure script generation: the returned source is run and verified elsewhere;
    nothing here is trusted as a physics result.
    """
    system, prompt = build_generation_prompt(npr, wrt)
    raw = llm.complete(system, prompt)
    return GeneratedScript(
        source=strip_fences(raw),
        variation_key=_variation_key(npr, wrt),
        llm_name=getattr(llm, "name", "unknown"),
        llm_version=llm.version(),
        raw=raw,
    )
