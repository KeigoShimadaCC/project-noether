# AGENTS.md — working guide for Project Noether

This file tells any AI agent (and any human contributor) how to work in this
repository. Read it before touching anything.

---

## 1. What this project is

Noether is an agentic symbolic-physics collaborator. A physicist pastes a LaTeX
action, answers a few clarifying questions, and gets back equations of motion, the
ADM decomposition, and perturbed theory in clean, canonical, publication-ready LaTeX.
An LLM orchestrates; established computer-algebra kernels (Cadabra2, xAct, SymPy)
compute; a verification layer checks every result; the human decides what the problem
actually is.

The full vision is in `NORTH_STAR.md`. That document wins every argument. When a
design or implementation choice is unclear, re-read sections 8 (principles) and 18
(anti-goals) and pick the option that survives them.

## 2. Document map and read order

Read in this order when onboarding:

1. `NORTH_STAR.md` — vision, principles, anti-goals. The constitution.
2. `docs/00_INDEX.md` — map of all documents and their status.
3. `docs/01_RESEARCH.md` — the CAS landscape and prior art we build on.
4. `docs/02_TECH_SPEC.md` — architecture, stack, the problem representation (NPR),
   kernel adapters, algorithms.
5. `docs/03_METHODOLOGY.md` — elicitation protocol, good form, verification ladder,
   development process.
6. `docs/04_EVALS.md` — the five acceptance evaluations with worked solutions.
   These are the tests that define "done" for Horizons 1 and 2.

Keep these documents current. If an implementation decision contradicts a doc,
either fix the implementation or update the doc in the same change, with a note on
why. Stale docs are bugs.

## 3. Non-negotiable rules for agents working here

These mirror the North Star principles in operational form.

1. **Never assert a symbolic result you did not compute.** If a tensor identity,
   variation, or simplification appears in code, a test, a doc, or a chat reply, it
   must come from a kernel run or from a citable standard result, and must be marked
   as such. Do not "remember" field equations into existence inside product code.
2. **Conventions are always explicit.** Every expression that enters or leaves a
   kernel carries its convention block (dimension, signature, curvature sign,
   symmetrization weight). No file, function, or test may assume a convention
   silently. Repo-wide defaults exist (section 5 below) but must be referenced by
   name, never implied.
3. **Provenance is part of the result type.** Any function that returns a computed
   expression returns it together with the script, kernel version, and assumptions
   that produced it. A bare expression with no receipt is a type error in spirit,
   and eventually in fact.
4. **Ambiguity goes to the human.** Product code never silently guesses field roles,
   symmetries, or gauges. When you, as a developing agent, face an ambiguous design
   choice with physics consequences, ask the user rather than picking.
5. **Evals are acceptance tests.** A capability does not exist until the
   corresponding eval in `docs/04_EVALS.md` passes end to end, with verification
   checks green. Add the eval before adding the capability.
6. **No backend lock-in.** Nothing outside a kernel adapter may import or depend on
   a specific CAS. The NPR (Noether Problem Representation) is the only language the
   orchestrator speaks.
7. **Correctness over speed, everywhere.** Do not cache, approximate, truncate, or
   parallelize in a way that can change a symbolic answer. Slow and right beats fast
   and plausible.

## 4. Repository layout

Current:

```
NORTH_STAR.md        Vision document
AGENTS.md            This file
docs/                Design and research documents (00 through 04)
noether/             Python package
  npr/               Problem representation: conventions, AST, schema, LaTeX,
                     validation, and the LaTeX action parser (parse.py)
  kernels/           Adapters: base contract, cadabra/ (subprocess; runs both
                     frozen golden templates and inline LLM-generated scripts,
                     generate.py parameterizes an audited scaffold for arbitrary
                     actions), sympy_kernel/; versions.py pins kernel versions
  llm/               LLM adapters behind one interface: ambient-auth CLI
                     subprocess (auto-detects codex/claude/gemini/droid; no API
                     key) plus an in-process stub for tests
  verify/            Check registry (V0..V3 implemented) and ladder runner
  provenance/        Result bundle writer
  orchestrator/      Session state machine, planner with ambiguity gate,
                     ingest (LaTeX action -> draft NPR + open ambiguity ledger),
                     elicit (model proposes resolutions; only human-confirmed
                     answers mutate the NPR), definitions (propose readability
                     shorthands like F_phi for dF/dphi; human adopts),
                     derive (general EOM path: model writes a Cadabra script,
                     kernel's residue check decides verified vs unverified),
                     store (JSON session persistence)
  server/            HTTP session API (FastAPI, optional [server] extra):
                     ingest/elicit/resolve/plan/derive with the no-guessing
                     contract
  mcp/               MCP stdio server (optional [mcp] extra): same session
                     surface as tools (incl. noether_derive); refusals are tool
                     results, not guesses
  cli/               `noether chat` / `resume` / `sessions` (conversational
                     loop, chat.py), `noether kernels`, `noether ingest`,
                     `noether elicit`, `noether serve`, `noether mcp`,
                     `noether eval{1..5}`, `noether eval1s` (ADM of GR),
                     `noether eval3s` (Minkowski spectrum)
evals/               Executable evals 1-5, 1s, 3s, 3p + a general-path eval
                     (test_eval_general) + registry + pytest gates
tests/               Unit and adapter tests (cadabra golden test included)
frontend/            Web client (Next.js + KaTeX) over the HTTP session API;
                     /api/* proxied to `noether serve`, no client-side physics
pyproject.toml       Package, deps, ruff, pytest config
```

The general derivation path (model writes a Cadabra script, kernel verifies it
through an in-script residue check) now serves arbitrary well-posed actions for
the `vary` task across the metric, scalar, and gauge-field classes; see
`docs/02_TECH_SPEC.md` section 6, item 7. The `perturb` task has its first
kernel-verified scaffold too: the frozen `pert_scalar_quadratic` template
expands a scalar action to quadratic order and checks the linearized EOM two
ways (eval 3p). Planned next: wire `perturb` into the general path and the
product surfaces, widen it past the scalar sector, add an `adm` scaffold, then
the derivation tree and export views in the web client.

## 4.1 Development setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"        # add [server] for the HTTP API
brew tap kpeeters/repo && brew install cadabra2   # official macOS channel
.venv/bin/python -m pytest -q                      # full suite; cadabra tests
                                                   # skip if kernel missing
.venv/bin/python -m noether.cli.main eval1         # end-to-end walking skeleton
.venv/bin/python -m noether.cli.main eval2         # ... likewise eval3, eval4
cd frontend && npm install && npm run dev          # web client; needs
                                                   # `noether serve` running
```

Cadabra2 is driven as a sandboxed subprocess (`cadabra2` CLI); set
`NOETHER_CADABRA` to point at a non-default binary.

## 5. Default physics conventions

These are the repo defaults, named `noether-default-v1`. Sessions may override any
of them through elicitation; code must thread the active convention block through
every computation.

- Dimension: 4. Signature: mostly plus, `(-,+,+,+)`.
- Riemann: `R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ}`.
- Ricci: `R_{μν} = R^λ_{μλν}`. Scalar: `R = g^{μν}R_{μν}`.
- d'Alembertian: `□ = g^{μν}∇_μ∇_ν`.
- Torsion: `T^λ_{μν} = Γ^λ_{μν} - Γ^λ_{νμ}`.
- (Anti)symmetrization with weight: `A_{(μν)} = ½(A_{μν} + A_{νμ})`.
- Canonical kinetic shorthand: `X = -½ ∇_μφ ∇^μφ`.
- Units: `c = 1`; keep `κ = 8πG` symbolic unless the user fixes it.

`docs/04_EVALS.md` is written in these conventions.

## 6. Engineering conventions

- Language: Python 3.12+, full type annotations, `pydantic` for the NPR schema.
- Tests: `pytest`. Every kernel adapter has a golden-output test pinned to a kernel
  version. Every eval has an executable counterpart under `evals/`.
- Kernel runs are sandboxed subprocesses with timeouts; pinned versions live in one
  place (a lockfile or Docker image tags), never scattered.
- Formatting and linting: `ruff` (format + lint). Keep CI green.
- Commits: small, imperative subject lines, body explains the why.
- Secrets: API keys only via environment, never committed, never logged.

## 7. Documentation conventions

- Sentence-case headings. No emojis. Avoid em dashes in new prose (the original
  NORTH_STAR.md predates this rule and keeps its style).
- Physics in documents follows `noether-default-v1` unless a section says otherwise,
  and says so explicitly.
- Worked derivations in docs must state their convention block and show enough
  intermediate steps that a physicist can audit them.
- When you change behavior, update the affected doc in the same change.

## 8. How to work a task in this repo

1. Locate the task against the horizon plan (`NORTH_STAR.md` section 17) and the
   tech spec. If it expands scope, flag it instead of quietly building it.
2. If the task adds capability, write or extend the eval first.
3. Implement behind the NPR boundary: orchestrator logic stays kernel-agnostic,
   kernel specifics stay in adapters.
4. Run the relevant evals and tests. A physics-bearing change with no kernel-backed
   test does not merge.
5. Update docs touched by the change.
6. In your summary, separate "what the kernel verified" from "what I reasoned about".
   That boundary is the product's core promise; practice it in development too.
