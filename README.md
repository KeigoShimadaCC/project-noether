# Noether

**Agentic symbolic-physics collaborator: write an action in LaTeX, answer a few
sharp questions, get back verified field equations with full provenance.**

A physicist pastes a LaTeX action, answers a short, targeted set of clarifying
questions, and gets back the equations of motion, the 3+1 (ADM) decomposition, and
the perturbed theory in clean, canonical, publication-ready LaTeX. An LLM
orchestrates; established computer-algebra kernels (Cadabra2, SymPy, and xAct on the
roadmap) compute; a verification ladder checks every result; the human decides what
the problem actually is.

The guiding sentence:

> A physicist writes an action in LaTeX, answers a few sharp questions, and gets
> back the equations of motion, the ADM decomposition, and the perturbed theory —
> clean, canonical, checkable, and reproducible — having spent their time on physics
> instead of syntax.

The full vision lives in [`NORTH_STAR.md`](NORTH_STAR.md); it wins every argument.

---

## Why

Symbolic tensor computation today forces a brutal trade: you can have **rigor** or
**ergonomics**, almost never both. xAct, Cadabra, and SymPy are correct and powerful,
but using them means hand-translating physics notation into an unforgiving CAS
dialect, declaring every symmetry and convention by hand, knowing in advance which
identities to apply, fighting term-expansion explosions, and reading the answer back
out of the dialect. A calculation that is conceptually an afternoon becomes weeks of
tooling.

The barrier is not the mathematics. It is the **translation-and-judgment layer**
between how a physicist thinks and what the machine demands — exactly where today's
tools are weakest and where modern LLMs are strongest. Noether is that layer.

---

## The core loop

A four-beat cycle, exactly as a good human collaboration would run:

1. **State the action.** Paste or type a LaTeX action `S = \int d^4x \sqrt{-g}\,(\dots)`.
   Messy, abbreviated, convention-light is fine.
2. **Noether interrogates, you answer.** Which symbols are dynamical? Is the metric
   symmetric? Is the connection Levi-Civita or independent with torsion and
   non-metricity? Dimension, signature, curvature sign? Vary with respect to what?
   Plain-language or menu answers turn an ambiguous string into a **well-posed
   problem**.
3. **The kernels compute.** With the problem well-defined, Noether plans the
   computation, generates audited kernel scripts, runs them in a sandbox, and drives
   the result toward *good form* — canonical indices, redundant terms removed,
   expressed in a basis the physicist recognizes.
4. **Noether returns the answer and its provenance.** LaTeX out, alongside the
   assumptions used, the identities applied, the checks that passed, and the exact
   kernel script that produced it. Accept, drill in, challenge a step, or change an
   assumption and re-run.

The loop is **resumable and stateful**: the problem definition, conventions, and
intermediate results persist across a session.

---

## What it can do today

Both Horizon 1 (equations of motion) and Horizon 2 (ADM, perturbation) evals pass,
and the general derivation path reaches arbitrary well-posed actions for the vary,
perturb, and adm tasks across the HTTP, MCP, and web surfaces, with persisted result
history.

- **Ingest** a LaTeX action into a backend-agnostic problem representation (the NPR).
- **Elicit** field roles, symmetries, geometry, and conventions through a
  propose-then-confirm dialogue that never guesses.
- **Derive equations of motion** (`vary`) for the metric, scalar, independent
  connection, and gauge-field classes — including full Palatini / metric-affine
  variation with torsion and non-metricity, the connection equation, Einstein-Cartan
  and hypermomentum decomposition, and a no-template compositional path for the
  nonminimal scalar-tensor and cubic Galileon sectors.
- **Perturb** to quadratic order (`perturb`) for scalar fields (incl. k-essence with
  the sound-speed kinetic mixing), the metric (the massless graviton), and abelian /
  non-abelian gauge potentials.
- **ADM 3+1 decomposition** (`adm`) of the gravitational sector — the Gauss-Codazzi
  split and the normal/tangential projections of the Einstein tensor, verified by the
  SymPy component kernel.
- **Teleparallel / symmetric-teleparallel routing** for f(T) and f(Q) families
  (curvature-free connection flag), with honest blockers where constrained-connection
  variation is not yet supported.
- **Provenance for every result**: the kernel script, raw output, assumptions
  snapshot, plan DAG, and verification verdicts, persisted and reloadable.

What it deliberately is **not**: a numerical relativity engine, a black box, a
theorem prover or discovery engine, or married to a single CAS. See
[`NORTH_STAR.md` §5](NORTH_STAR.md).

---

## Architecture

Four layers, with two hard boundaries (the NPR boundary and the provenance boundary):

```
Frontends          CLI chat  ·  Next.js web app  ·  MCP server  (all thin)
    │ HTTP / JSON / MCP — sessions, NPR diffs
Orchestrator       INGEST → ELICIT → PLAN → COMPUTE → VERIFY → PRESENT
(FastAPI, Python)  session store: NPR + transcript + results + artifacts
    │ NPR (the only language crossing here)   │ check requests
Kernel adapters                               Verification layer
  Cadabra2 (sandboxed subprocess)               check registry, ladder V0..V4
  SymPy (in-process)                            (component spot-checks today)
  xAct (wolframscript, roadmap)
```

- **The NPR (Noether Problem Representation)** is the backend-agnostic contract
  between "what the physicist meant" and "what any kernel executes" — a versioned,
  diffable pydantic schema. The orchestrator never emits kernel syntax into results,
  and kernels never see raw user LaTeX.
- **The model orchestrates; it cannot inject results.** Only `run_kernel` output can
  land in a result bundle. The "no unearned assertions" principle is mechanically
  enforced, not merely policy.
- **The verification ladder** (V0 well-formedness → V1 structural invariants → V2
  identity checks → V3 limiting cases → V4 independent recomputation) wraps every
  answer. "I checked it three ways and it holds" means V2 + V3 + V4 green.

Full detail in [`docs/02_TECH_SPEC.md`](docs/02_TECH_SPEC.md).

---

## Repository layout

```
NORTH_STAR.md        Vision document — the constitution
AGENTS.md            Working guide for agents and human contributors
README.md            This file
docs/                Design and research documents (00 INDEX through 04 EVALS)
noether/             Python package
  npr/               Problem representation: conventions, AST, schema, LaTeX,
                     validation, and the LaTeX action parser
  kernels/           Adapters: base contract, cadabra/ (subprocess; golden
                     templates, blocks.py compositional path, curvature.py
                     primitives, horndeski_g4g5.py), sympy_kernel/; versions.py
  llm/               LLM adapters behind one interface (ambient-auth CLI + stub)
  verify/            Check registry (V0..V3) and ladder runner
  provenance/        Result bundle writer/reader
  orchestrator/      Session state machine, planner, ingest, elicit, derive, store
  server/            HTTP session API (FastAPI, optional [server] extra)
  mcp/               MCP stdio server (optional [mcp] extra)
  cli/               noether chat / resume / sessions / kernels / ingest / elicit /
                     serve / mcp / eval{1..5}, eval1s, eval3s
evals/               Executable evals (1-5, 1s, 3s, 3p, 3g, 3a, 3y, 3k, 6, 7, 8,
                     f(T), f(Q)) + registry + pytest gates
tests/               Unit and adapter tests (cadabra golden tests skip if absent)
frontend/            Next.js + KaTeX web client over the HTTP session API
pyproject.toml       Package, deps, ruff, pytest config
```

---

## Installation

Requires Python 3.12+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"        # add [server] and/or [mcp]
brew tap kpeeters/repo && brew install cadabra2     # official macOS channel
```

Cadabra2 is driven as a sandboxed subprocess (`cadabra2` CLI). Set `NOETHER_CADABRA`
to point at a non-default binary. The cadabra-backed tests and evals skip cleanly when
the kernel is not installed; SymPy ships with the core package.

The LLM backend is **ambient-auth, no API key**: it auto-detects an installed agent
CLI (codex, claude, gemini, droid) and runs it one-shot as a sandboxed subprocess, so
credentials stay in that CLI's own login session. A deterministic stub backs the tests.

---

## Quickstart

```sh
# List kernel adapters and availability
.venv/bin/python -m noether.cli.main kernels

# Parse an action into a draft NPR and see the questions that block planning
.venv/bin/python -m noether.cli.main ingest "-\tfrac14 F_{\mu\nu} F^{\mu\nu}"

# Ingest, then let a detected agent CLI propose answers (unconfirmed by default)
.venv/bin/python -m noether.cli.main elicit "K(\phi,X) + F(\phi) R" --accept-llm

# Run an eval end to end (walking skeleton, with a provenance bundle)
.venv/bin/python -m noether.cli.main eval1        # likewise eval2..eval5, eval1s, eval3s

# Conversational loop: ingest, clarify, plan (resumable)
.venv/bin/python -m noether.cli.main chat
.venv/bin/python -m noether.cli.main sessions
.venv/bin/python -m noether.cli.main resume <session-id>
```

After installing the `noether` package, the `noether` console script is equivalent to
`python -m noether.cli.main`.

### HTTP session API

```sh
.venv/bin/python -m noether.cli.main serve         # FastAPI on 127.0.0.1:8754
```

`POST /sessions` ingests an action and returns the open question ledger;
`POST /sessions/{id}/elicit` returns unconfirmed model proposals;
`POST /sessions/{id}/resolve` (validated against the listed options) mutates the
session; `GET /sessions/{id}/plan` returns 409 until the problem is well posed;
`POST /sessions/{id}/derive` runs the general derivation (`kind` = eom | perturbation
| adm) and returns each field equation with the kernel's `verified` verdict;
`GET /sessions/{id}/results` reloads the full derivation history.

### MCP server

```sh
.venv/bin/python -m noether.cli.main mcp           # stdio; requires [mcp] extra
```

Exposes the same session surface as tools (`noether_ingest`, `noether_resolve`,
`noether_plan`, `noether_derive`, `noether_results`, …) so any MCP-capable agent can
delegate tensor calculus to Noether the way it delegates arithmetic to a sandbox.
Refusals are tool results, not exceptions: a host LLM cannot make Noether guess.

### Web client

```sh
cd frontend && npm install && npm run dev          # needs `noether serve` running
```

Next.js App Router + KaTeX. The browser talks only to Next; `/api/*` is proxied to the
FastAPI server (`NOETHER_API_URL`, default `http://127.0.0.1:8754`), so no physics
state lives client-side. The workspace ingests actions, runs the question flow, shows
the NPR side panel, renders each derivation as a provenance tree, and exports
kernel-verified results as a publication-LaTeX document.

---

## Default conventions

The repo defaults are named `noether-default-v1`; sessions may override any of them
through elicitation, and code threads the active convention block through every
computation. Metric-affine work additionally uses the `metric-affine-v1` block for
contortion/disformation signs.

- Dimension 4, signature mostly-plus `(-,+,+,+)`.
- `R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ}`,
  `R_{μν} = R^λ_{μλν}`, `R = g^{μν}R_{μν}`.
- `□ = g^{μν}∇_μ∇_ν`; `T^λ_{μν} = Γ^λ_{μν} - Γ^λ_{νμ}`.
- `A_{(μν)} = ½(A_{μν} + A_{νμ})`; `X = -½ ∇_μφ ∇^μφ`.
- `c = 1`; `κ = 8πG` kept symbolic unless fixed.

See [`AGENTS.md` §5](AGENTS.md).

---

## Development

```sh
.venv/bin/python -m pytest -q                       # full suite; cadabra tests skip if absent
.venv/bin/ruff format . && .venv/bin/ruff check .   # format + lint
```

Eval-driven development: a capability does not exist until its eval in
[`docs/04_EVALS.md`](docs/04_EVALS.md) passes end to end with checks green. Add the
eval before the capability. Every kernel adapter operation has a golden test pinned to
a kernel version; physics-bearing changes need a kernel-backed test. When you change
behavior, update the affected doc in the same change — stale docs are bugs. The full
contributor contract is in [`AGENTS.md`](AGENTS.md).

---

## Documentation

| Document | Purpose |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | Vision: the destination and the why. The constitution. |
| [`AGENTS.md`](AGENTS.md) | Working guide for agents and contributors. |
| [`docs/00_INDEX.md`](docs/00_INDEX.md) | Map of all documents and their status. |
| [`docs/01_RESEARCH.md`](docs/01_RESEARCH.md) | CAS landscape, prior art, kernel selection. |
| [`docs/02_TECH_SPEC.md`](docs/02_TECH_SPEC.md) | Architecture, stack, NPR schema, adapters, algorithms. |
| [`docs/03_METHODOLOGY.md`](docs/03_METHODOLOGY.md) | Elicitation, good form, verification ladder, dev process. |
| [`docs/04_EVALS.md`](docs/04_EVALS.md) | Acceptance evaluations with worked solutions. |

---

## Status and license

Research prototype (`v0.1.0`). Horizons 1 and 2 pass their evals; Horizon 3 (the xAct
cross-check kernel and full multi-backend cross-recomputation) is in progress.

License not yet chosen. Note that Cadabra2 is GPL-3.0 and is invoked as a subprocess,
not linked. See [`.github/REPO_METADATA.md`](.github/REPO_METADATA.md).
