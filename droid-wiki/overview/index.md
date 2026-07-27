# Project Noether

Noether is an agentic symbolic-physics collaborator. A physicist pastes a LaTeX action, answers a short set of clarifying questions, and gets back the equations of motion, the 3+1 (ADM) decomposition, and the perturbed theory in clean, canonical, publication-ready LaTeX. An LLM orchestrates the conversation; established computer-algebra kernels (Cadabra2 and SymPy) do the computing; a verification layer checks every result; the human decides what the problem actually is.

## What problem it solves

Symbolic tensor computation forces a trade between rigor and ergonomics. Tools like xAct, Cadabra, and SymPy are correct and powerful, but using them means hand-translating physics notation into an unforgiving CAS dialect, declaring every symmetry and convention by hand, and reading the answer back out of that dialect. The hard part is not the mathematics. It is the translation-and-judgment layer between how a physicist thinks and what the machine demands. Noether is that layer.

## The core loop

The tool runs a four-beat cycle that mirrors a good human collaboration:

1. **State the action.** Paste a LaTeX action `S = \int d^4x \sqrt{-g}\,(\dots)`. Messy and convention-light is fine.
2. **Noether interrogates, you answer.** Which symbols are dynamical? Is the connection Levi-Civita or independent with torsion and non-metricity? Signature, curvature sign, vary with respect to what? Menu or plain-language answers turn an ambiguous string into a well-posed problem.
3. **The kernels compute.** With the problem well-defined, Noether plans the computation, generates audited kernel scripts, runs them in a sandbox, and drives the result toward good form.
4. **Noether returns the answer and its provenance.** LaTeX out, alongside the assumptions used, the checks that passed, and the exact kernel script that produced it.

The loop is resumable and stateful. The problem definition, conventions, and intermediate results persist across a session.

## What it can do today

Both Horizon 1 (equations of motion) and Horizon 2 (ADM, perturbation) evals pass. The geometry spans the general metric-affine case: the independent affine connection with torsion `T` and non-metricity `Q` is the default setting, and Levi-Civita is the `T = Q = 0` special case.

- **Ingest** a LaTeX action into a backend-agnostic problem representation (the NPR).
- **Elicit** field roles, symmetries, geometry, and conventions through a propose-then-confirm dialogue that never guesses.
- **Derive equations of motion** (`vary`) for the metric, scalar, independent connection, and gauge-field classes, including full Palatini/metric-affine connection variation, the Einstein-Cartan case, and a no-template compositional path for scalar-tensor and cubic Galileon sectors.
- **Derive teleparallel field equations**: `f(Q)` symmetric teleparallel and `f(T)` metric teleparallel, kernel-verified.
- **Perturb** to quadratic order (`perturb`) for scalars, the metric, abelian and non-abelian gauge potentials, and metric-affine backgrounds.
- **ADM 3+1 decomposition** (`adm`) of the gravitational sector, including the metric-affine connection-sector foliation split.
- **Gate the hard cases honestly.** Capabilities that need machinery not installed here return `verified=false` with a non-empty `detail` naming the blocker, never a fabricated result.
- **Provenance for every result**: the kernel script, raw output, assumptions snapshot, plan DAG, named convention block, and verification verdicts.

What it is deliberately not: a numerical relativity engine, a black box, a theorem prover, or a tool married to a single CAS.

## The two promises

Two rules define the product and are enforced mechanically, not by policy:

- **No unearned assertions.** Only kernel output can land in a result bundle. The model orchestrates; it cannot inject a computed expression.
- **No silent guessing.** A non-empty ambiguity ledger structurally blocks planning. Ambiguity is resolved by asking the human, never by the model picking.

## Quick links

- [Architecture](architecture.md) - the four layers and two hard boundaries
- [Getting started](getting-started.md) - install, build, test, run
- [Glossary](glossary.md) - NPR, the verification ladder, metric-affine terms
- [Surfaces (apps)](../apps/index.md) - CLI, HTTP server, MCP server, web client
- [Systems](../systems/index.md) - NPR, orchestrator, kernels, verification, provenance
- [Features](../features/index.md) - ingest, elicitation, EOM, perturbation, ADM

The authoritative design documents live in the repository: `NORTH_STAR.md` (the vision), `AGENTS.md` (the contributor contract), and `docs/02_TECH_SPEC.md` (the architecture).
