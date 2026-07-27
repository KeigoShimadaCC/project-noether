# Background

This section holds the rationale behind Project Noether's load-bearing design choices and the traps a contributor can fall into while working inside them. It is the "why" layer: the architecture page describes the boundaries these decisions create, and the patterns-and-conventions page describes the rules in operational form, but neither explains the failure modes the rules exist to prevent.

## Why these decisions matter

Noether's central promise is a bright line between what the model reasoned about and what a kernel computed. An LLM is good at language, planning, and turning a LaTeX action into a kernel script; it is bad at tensor algebra and will, given the chance, hallucinate a tensor identity with perfect confidence. So the system is built so that no human, no orchestrator, and no model can place an unverified expression into a result. Only a kernel's own check sets `verified`. Around that line sit a second set of guarantees: ambiguity is resolved by asking, not guessing; provenance ships with every result; and the backend is never locked to one CAS.

These are not policy. Each is enforced mechanically, in code, by a gate or a type that makes the violation impossible to merge silently. The design-decisions page names the choices and the files that enforce them; the pitfalls page names the ways a contributor can still go wrong while working inside the gates.

## Pages

- [Design decisions](./design-decisions.md) - the load-bearing choices (the NPR boundary, the provenance boundary, the ambiguity gate, the dual gate for metric-affine results, ambient-auth LLM, immutable NPR versions, frozen golden templates, the teaching channel, honest gating) and the failure mode each one exists to prevent.
- [Pitfalls](./pitfalls.md) - the danger zones and known traps: the torsion trap, raising and lowering across `nabla`, missing AST cue cases, kernel version pins, model-injected answers, silent conventions, NPR version edits, the Palatini projective mode, non-sentinel stdout, and caching that can change a symbolic answer.

## Related pages

- [../overview/architecture.md](../overview/architecture.md) - the boundaries these decisions create.
- [../how-to-contribute/patterns-and-conventions.md](../how-to-contribute/patterns-and-conventions.md) - the rules in operational form.
- [../systems/verification.md](../systems/verification.md) - the dual gate.
- [../systems/llm.md](../systems/llm.md) - ambient-auth LLM transport.
- [../systems/npr.md](../systems/npr.md) - immutable NPR versions.
- [../systems/kernels/cadabra.md](../systems/kernels/cadabra.md) - frozen templates and sentinels.
