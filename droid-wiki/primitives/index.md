# Primitives

Active contributors: KeigoShimadaCC

## Purpose

This section documents the foundational domain objects that cross three or more systems in Noether. They are the bedrock types that the orchestrator, kernel adapters, verification layer, provenance bundles, and the NPR (Noether Problem Representation) all depend on. Because they sit beneath the system boundary, a change to any of them ripples through the whole tool, and a misuse of any of them breaks one of the two mechanically-enforced promises (no unearned assertions, no silent guessing).

The primitives are deliberately small, typed, and pydantic-backed. They are not services and not orchestration logic. They are the contracts that services pass to each other.

## The primitives

| Primitive | Source | One-line summary |
|---|---|---|
| Conventions | `noether/npr/conventions.py` | Frozen pydantic model carrying the full convention block (signature, curvature and torsion signs, Ricci contraction, field-strength definition, K-sign, foliation normal, and the rest) that travels with every expression crossing a kernel boundary. |
| Expression AST | `noether/npr/ast.py` | The typed tree (`Num`, `Sym`, `Func`, `Tensor`, `Deriv`, `Pow`, `Prod`, `Sum`, with abstract `Index`) that holds the Lagrangian and every kernel result. LaTeX is a rendering of this tree, never the source of truth. |
| Computed result | `noether/kernels/base.py` | The kernel adapter contract: `Capability`, `KernelTask`, `KernelScript`, `KernelRawOutput`, `ComputedResult`, the `KernelAdapter` Protocol, and `KernelUnavailable`. The only object that can carry a computed expression into a result. |

## Where each appears

| Primitive | NPR schema | Orchestrator | Kernels | Verification | Provenance | LaTeX renderer |
|---|---|---|---|---|---|---|
| Conventions | `NPR.conventions` | threaded through derive and elicitation | pinned to the script that runs | V0 reads `metric_compatible` | bundled with every result | signature-aware rendering |
| Expression AST | `Action.lagrangian` | elicitation walks it for geometric cues | results returned as trees | V0 validates index balance | derivations stored as trees | `noether/npr/latex.py` renders it |
| Computed result | n/a (output, not input) | derive wraps kernel output in it | adapters produce it | ladder checks its `notes` | bundle writer reads it | n/a |

## Related pages

- [Conventions](conventions.md) - the convention block and `noether-default-v1`
- [Expression AST](expression-ast.md) - the typed expression tree
- [Computed result](computed-result.md) - the kernel adapter contract

## Cross-references

- [NPR system](../systems/npr.md) - the schema that holds these primitives
- [Kernels system](../systems/kernels/index.md) - adapters that produce `ComputedResult`
- [Verification system](../systems/verification.md) - V0 validates the AST
- [Provenance system](../systems/provenance.md) - bundles carry conventions and derivations
- [Patterns and conventions](../how-to-contribute/patterns-and-conventions.md) - the rules these primitives enforce
- [Glossary](../overview/glossary.md) and [Architecture](../overview/architecture.md)
