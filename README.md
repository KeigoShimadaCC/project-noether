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

Both Horizon 1 (equations of motion) and Horizon 2 (ADM, perturbation) evals pass.
A later mission lifted the geometry to a general metric-affine setting, so the
independent affine connection with torsion `T^λ_{μν}` and non-metricity `Q_{λμν}` is
the geometry the tool reasons in by default; Levi-Civita is the `T = Q = 0` special
case. The general derivation path reaches arbitrary well-posed actions for the vary,
perturb, and adm tasks across the HTTP, MCP, and web surfaces, with persisted result
history.

- **Ingest** a LaTeX action into a backend-agnostic problem representation (the NPR).
- **Elicit** field roles, symmetries, geometry, and conventions through a
  propose-then-confirm dialogue that never guesses. Geometry inference reads the
  action's structural cues (`R(Γ)`, an explicit `T` or `Q`, the `f(Q)`/`f(T)` family)
  and proposes on-menu resolutions for a human to confirm. It never auto-applies them.
- **Reason in general metric-affine geometry**: curvature, torsion, and non-metricity
  primitives; the post-Riemannian decomposition `Γ = LC + K(T) + L(Q)` into the
  Levi-Civita part, the contortion, and the disformation; and the modified Bianchi
  identities. The Levi-Civita primitives are kept as the `T = Q = 0` case.
- **Derive equations of motion** (`vary`) for the metric, scalar, independent
  connection, and gauge-field classes. This covers the full Palatini / metric-affine
  connection variation with its projective mode (`Γ = LC(g) + δ^λ_ν A_μ` with `A_μ`
  arbitrary, so the connection is never uniquely fixed), the Einstein-Cartan case and
  the spin/dilation/shear hypermomentum decomposition, vector/gauge matter under the
  `F = dA` versus `F = ∇A` field-strength choice (the two differ by `T^λ_{μν}A_λ`
  under torsion), multifield Palatini scalar-tensor, and a no-template compositional
  path for the nonminimal scalar-tensor and cubic Galileon sectors.
- **Derive teleparallel field equations** (kernel-verified): `f(Q)` symmetric
  teleparallel in the coincident gauge (through the `Q = R + boundary` identity) and
  `f(T)` metric teleparallel on the tetrad/Weitzenböck connection. The linear
  `f(Q) = Q` and `f(T) = T` equations pass the Cadabra residue check, and the general
  `f(Q)`/`f(T)` field equation is cross-checked componentwise in SymPy.
- **Perturb** to quadratic order (`perturb`) for scalar fields (including k-essence
  with the sound-speed kinetic mixing), the metric (the massless graviton), and
  abelian or non-abelian gauge potentials, plus the metric-affine background (the
  metric fluctuation `h` together with the connection fluctuation `dG`) and the
  vector-affine perturbation for both field-strength choices.
- **ADM 3+1 decomposition** (`adm`) of the gravitational sector: the Gauss-Codazzi
  split and the normal/tangential projections of the Einstein tensor, verified by the
  SymPy component kernel (the ADM path writes no Cadabra script). On a metric-affine
  background it adds the connection-sector foliation split (the torsion and
  non-metricity pieces projected along the foliation), the primary/secondary
  constraint structure (gated when the Dirac chain cannot close, for example when
  `Q ≠ 0`), and the matter hypermomentum coupling.
- **Gate the hard cases honestly.** The higher Horndeski densities `G4(φ,X)R` and
  `G5` need covariant-derivative normal-ordering (xAct's `SortCovDs`, which is not
  installed here), so they return `verified=false` with a non-empty `detail` naming
  the blocker rather than a fabricated result.
- **Explain the geometry** through a separate teaching channel that narrates the
  tradeoffs of the geometric choices (torsion to spin coupling, projective freedom to
  a non-unique connection, and so on). The teaching is reasoned, not kernel-verified;
  it lives on its own field and never sets a result or mutates the NPR.
- **Provenance for every result**: the kernel script, raw output, assumptions
  snapshot, plan DAG, named convention block, and verification verdicts. The same
  history reloads identically across the HTTP API (`GET /sessions/{id}/results`), MCP
  (`noether_results`), and the bundle on disk, and renders in the web client as a
  provenance tree with kernel-verified / unverified / stale badges, a teaching panel,
  and a LaTeX export of the verified results only.

How a metric-affine result earns the word "verified": both the Cadabra in-script
residue check and the SymPy general-connection cross-check (run on explicit random
metric and connection backgrounds) must agree. This dual gate is what catches the
"torsion trap", where a Levi-Civita shortcut silently drops a torsion term and still
reports a zero residue.

What it deliberately is **not**: a numerical relativity engine, a black box, a
theorem prover or discovery engine, or married to a single CAS. See
[`NORTH_STAR.md` §5](NORTH_STAR.md).

---

## Architecture

Four layers, with two hard boundaries (the NPR boundary and the provenance boundary):

```
Frontends          CLI chat  ·  Next.js web app  ·  MCP server  (all thin)
    │ HTTP / JSON / MCP: sessions, NPR diffs
Orchestrator       INGEST → ELICIT → PLAN → COMPUTE → VERIFY → PRESENT
(FastAPI, Python)  session store: NPR + transcript + results + artifacts
    │ NPR (the only language crossing here)   │ check requests
Kernel adapters                               Verification layer
  Cadabra2 (sandboxed subprocess)               check registry, ladder V0..V4
  SymPy (in-process; general-connection oracle) (cross-checks + the dual gate)
  xAct (wolframscript, roadmap)
```

- **The NPR (Noether Problem Representation)** is the backend-agnostic contract
  between "what the physicist meant" and "what any kernel executes": a versioned,
  diffable pydantic schema. The orchestrator never emits kernel syntax into results,
  and kernels never see raw user LaTeX.
- **The model orchestrates; it cannot inject results.** Only `run_kernel` output can
  land in a result bundle. The "no unearned assertions" principle is mechanically
  enforced, not merely policy.
- **The verification ladder** (V0 well-formedness → V1 structural invariants → V2
  identity checks → V3 limiting cases → V4 independent recomputation) wraps every
  answer. "I checked it three ways and it holds" means V2 + V3 + V4 green. For a
  metric-affine result the SymPy general-connection oracle runs as an independent
  cross-check on explicit backgrounds, and the result is verified only when the
  Cadabra residue check and the SymPy cross-check agree (the dual gate).

Full detail in [`docs/02_TECH_SPEC.md`](docs/02_TECH_SPEC.md).

---

## Repository layout

```
NORTH_STAR.md        Vision document, the constitution
AGENTS.md            Working guide for agents and human contributors
README.md            This file
docs/                Design and research documents (00 INDEX through 04 EVALS)
noether/             Python package
  npr/               Problem representation: conventions (incl. metric-affine-v1),
                     AST, schema, LaTeX, validation, and the LaTeX action parser
  kernels/           Adapters: base contract, cadabra/ (subprocess; golden
                     templates, generate.py model-script path, blocks.py
                     compositional path, curvature.py metric-affine primitives,
                     horndeski_g4g5.py), sympy_kernel/ (geometry oracle, adm.py,
                     ft_tetrad.py, fq_coincident.py, linearized.py); versions.py
  llm/               LLM adapters behind one interface (ambient-auth CLI + stub)
  verify/            Check registry (V0..V3) and ladder runner
  provenance/        Result bundle writer/reader
  orchestrator/      Session state machine, planner, ingest, elicit, definitions,
                     derive, store
  server/            HTTP session API (FastAPI, optional [server] extra)
  mcp/               MCP stdio server (optional [mcp] extra)
  cli/               noether chat / resume / sessions / kernels / ingest / elicit /
                     serve / mcp / eval{1..5}, eval1s, eval3s, eval4ma, adm-affine,
                     vector-affine
evals/               Executable evals (1-5, 1s, 3s, 3p, 3g, 3a, 3y, 3k, 6, 7, 8,
                     4ma, adm-affine, vector-affine and its perturbation, f(T),
                     f(Q)) + registry + pytest gates
tests/               Unit and adapter tests (cadabra golden tests skip if absent;
                     dual-gate, metric-affine, teleparallel, hypermomentum suites)
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
.venv/bin/python -m noether.cli.main eval1        # also eval2..eval5, eval1s, eval3s,
                                                  # eval4ma, adm-affine, vector-affine

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
computation. Metric-affine work carries the `metric-affine-v1` block on top of the
base: torsion sign, non-metricity definition, contortion sign, disformation sign,
Ricci-contraction (an elicited choice once the connection is independent and Ricci
is non-symmetric), the field-strength definition (`F = dA` exterior derivative by
default, or `F = ∇A` covariant curl), the extrinsic-curvature `K`-sign, and the
foliation-normal direction. `noether/npr/conventions.py` is the authoritative source.

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

Research prototype (`v0.1.0`). Horizons 1 and 2 pass their evals, and the geometry now
spans the general metric-affine case: Palatini and Einstein-Cartan equations of
motion, verified `f(T)` and `f(Q)` teleparallel field equations, and metric-affine
perturbations and ADM, all under the Cadabra-plus-SymPy dual gate. The remaining
frontier stays honest about its limits: the higher Horndeski `G4(φ,X)R`/`G5` closures
are gated until covariant-derivative normal-ordering lands, which is the first job for
the planned xAct cross-check kernel and full multi-backend cross-recomputation.

License not yet chosen. Note that Cadabra2 is GPL-3.0 and is invoked as a subprocess,
not linked. See [`.github/REPO_METADATA.md`](.github/REPO_METADATA.md).
