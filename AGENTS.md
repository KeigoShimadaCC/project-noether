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

0. `README.md` — overview, install, quickstart, and the command/API surface.
   The fastest orientation; the documents below are authoritative.
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
README.md            Front door: overview, install, quickstart, surface map
NORTH_STAR.md        Vision document
AGENTS.md            This file
docs/                Design and research documents (00 through 04)
noether/             Python package
  npr/               Problem representation: conventions, AST, schema, LaTeX,
                     validation, and the LaTeX action parser (parse.py)
  kernels/           Adapters: base contract (including Capability.INDEPENDENT_CONNECTION),
                     cadabra/ (subprocess; runs frozen
                     golden templates, inline LLM-generated scripts via
                     generate.py, and blocks.py, which decomposes an additive
                     action into building blocks and assembles one script the
                     kernel residue-checks, no per-theory template; covers the
                     scalar and metric equations of motion of the nonminimal
                     scalar-tensor / scalar Horndeski sector),
                     sympy_kernel/; versions.py pins kernel versions
  llm/               LLM adapters behind one interface: ambient-auth CLI
                     subprocess (auto-detects codex/claude/gemini/droid; no API
                     key) plus an in-process stub for tests
  verify/            Check registry (V0..V3 implemented) and ladder runner
  provenance/        Result bundle writer and reader (write_bundle stores a
                     derivations.json per result; read_results reloads a
                     session's recorded derivations for history)
  orchestrator/      Session state machine, planner with ambiguity gate,
                     ingest (LaTeX action -> draft NPR + open ambiguity ledger;
                     adds a Gamma connection object when the action carries an
                     explicit connection, so the derive path can vary it),
                     elicit (model proposes resolutions; only human-confirmed
                     answers mutate the NPR, including geometry.connection,
                     the Ricci-contraction question that opens for an
                     independent connection, and the field-strength-definition
                     question that opens when a vector/gauge potential exists
                     on an independent-connection background; geometry
                     inference is exercised deterministically with
                     StubLLMAdapter: propose_resolutions returns one proposal
                     per open geometry and convention ambiguity with every
                     non-null choice in the menu, off-menu suggestions yield
                     choice=None, and the NPR is unchanged after proposing;
                     the inference prompt embeds the action's geometric cues
                     (R(Γ), T, Q, f(Q)/f(T) family) so the model's proposals
                     are grounded in the action, not a fixed default; a scalar
                     action carries no such cue; convention proposals
                     (Ricci-contraction, field-strength definition) are on-menu
                     with rationale and never auto-applied), definitions (propose readability
                     shorthands like F_phi for dF/dphi and, on metric-affine
                     NPRs, K(T), L(Q), and the f(Q) scalar Q; human adopts),
                     derive (general EOM / perturbation path: model writes a
                     Cadabra script, kernel's residue check decides verified vs
                     unverified; derive_eom includes connection fields in its
                     default list when geometry.connection.type is independent;
                     plus derive_adm, a SymPy-verified ADM split),
                     store (JSON session persistence; derive records each
                     result id into the session so history reloads)
  server/            HTTP session API (FastAPI, optional [server] extra):
                     ingest/elicit/resolve/plan/derive plus a results history
                     endpoint, all under the no-guessing contract
  mcp/               MCP stdio server (optional [mcp] extra): same session
                     surface as tools (incl. noether_derive and
                     noether_results); refusals are tool results, not guesses;
                     noether_derive with with_respect_to=['g','Gamma'] returns
                     both EOMs on a resolved Palatini session and a blocked
                     dict when the connection ambiguity is still open
  cli/               `noether chat` / `resume` / `sessions` (conversational
                     loop, chat.py; numbered geometry answers go through the
                     same menu-validation path as HTTP resolve before the
                     session store is updated), `noether kernels`, `noether ingest`,
                     `noether elicit`, `noether serve`, `noether mcp`,
                     `noether eval{1..5}`, `noether eval1s` (ADM of GR),
                     `noether eval3s` (Minkowski spectrum)
evals/               Executable evals 1-5, 1s, 3s, 3p, 3g, 3a, 3y, 3k, 6 (cubic Galileon),
                     7 (k-essence / general scalar Horndeski by composition),
                     8 (nonminimal scalar-tensor by composition, both EOMs)
                     + a general-path eval (test_eval_general) + registry +
                     pytest gates
tests/               Unit and adapter tests (cadabra golden test included;
                     test_cross_flows.py covers metric-affine cross-surface
                     consistency: HTTP/MCP/bundle round-trip, stale marking,
                     session resume, and the MCP blocked/refusal path;
                     test_geometry_inference.py covers the geometry inference
                     contract: propose_resolutions for geometry ambiguities,
                     off-menu nulled, never auto-applied, apply_resolutions
                     on-menu mutates, off-menu raises, HTTP /elicit and
                     /resolve surface enforcement)
frontend/            Web client (Next.js + KaTeX) over the HTTP session API;
                     /api/* proxied to `noether serve`, no client-side physics
pyproject.toml       Package, deps, ruff, pytest config
```

The general derivation path (model writes a Cadabra script, kernel verifies it
through an in-script residue check) now serves arbitrary well-posed actions for
the `vary` task across the metric, scalar, connection, and gauge-field classes;
see `docs/02_TECH_SPEC.md` section 6, item 7. Connection variation routes to
the `vary-connection` worked example (the audited `eval2_palatini_connection`
template) and carries `Capability.INDEPENDENT_CONNECTION`, so a connection field
is never silently routed to the metric worked example. For the pure Palatini
Einstein-Hilbert action (no matter fields other than the metric and the
independent connection), the connection EOM routes directly to the frozen
`eval2_palatini_connection` template, surfacing the verified projective-family
result (checks `solution_zero` and `ricci_shift_is_dA`) with a payload that
states the projective freedom (`Gamma = LC(g) + delta^lambda_nu A_mu`,
`A_mu` arbitrary) and never presents the connection as uniquely fixed
(VAL-EOM-004). Non-pure-EH connection variations (Palatini scalar-tensor,
Einstein-Cartan) still route through the general LLM-written script path. A scalar action with a `box`-coupling
(the Horndeski G3 term `K(phi) box phi`) routes to the audited
`eom_cubic_galileon_scalar` scaffold (eval 6), the first verified member past
scalar-tensor; see `docs/02_TECH_SPEC.md` section 6.1 for the representation
boundaries this exposed. The `vary` task also has a compositional path that
needs no model: when an additive Lagrangian decomposes fully into registered
building blocks (`noether/kernels/cadabra/blocks.py`), `derive_field` assembles
one script for the real action plus an independent candidate from the same
blocks, and the kernel residue-checks it. The scalar EOM blocks are canonical
kinetic, potential, cubic Galileon, k-essence `K(phi, X)`, and nonminimal
`F(phi) R` (eval 7 and 8); the metric EOM blocks are Einstein-Hilbert,
nonminimal `F(phi) R`, kinetic, potential, and cubic Galileon `G(phi) box phi`
(eval 8 and 6), so the full nonminimal scalar-tensor theory and the cubic
Galileon yield both equations of motion this way. It expands `X` to
its primitive in the kernel and collapses it back for display, and refuses
(falls back to the model path) on any term that matches no block, including the
held-out higher Horndeski densities (`G4(phi, X) R`, G5), which route to the
best-effort G4/G5 path (`attempt_g4g5_eom` in `derive.py`) instead of the
generic model-written script, returning `verified=False` with a non-empty
`detail` naming the SortCovDs blocker (VAL-EOM-013). The `perturb` task runs through the
same path for scalar fields, the metric, and rank-1 gauge potentials:
`derive_perturbation`, reachable as `kind="perturbation"` on the server, MCP,
and web clients, drives the frozen `pert_scalar_quadratic` scaffold (eval 3p)
for scalars, `pert_kessence_quadratic` (eval 3k) for an `X`-dependent scalar
`K(phi, X)` (it expands `X`, surfacing the sound-speed kinetic mixing),
`pert_metric_quadratic` (eval 3g) for the metric, `pert_gauge_quadratic`
(eval 3a) for an abelian gauge potential, and `pert_yang_mills_quadratic`
(eval 3y) for a non-abelian one (selected by the object's `gauge_group`
marker), expanding the action to quadratic order and
checking the linearized EOM two ways; all checks must pass before a result is
called verified. On a metric-affine (independent-connection) background,
a metric perturbation routes to `pert_metric_affine_quadratic` (including the
connection fluctuation dG), and a gauge-field (vector) perturbation routes to
`pert_vector_affine_dA_quadratic` for the F=dA field-strength choice
(both checks pass, no connection fluctuation in the result) or
`pert_vector_affine_covcurl_quadratic` for the F=nabla A choice
(residue gated due to the Kronecker-delta limitation with mixed-index dG
objects; SymPy cross-check provides independent verification); the two
choices differ by torsion-dependent terms (VAL-PERT-017) and the covcurl
action retains a*dG cross terms (VAL-PERT-018). The metric perturbation on a
metric-affine background (`pert_metric_affine_quadratic`, eval 4ma) includes
the connection fluctuation `dG` alongside the metric fluctuation `h`; the
connection is not perturbed independently, so it is excluded from the default
perturbation field list and requesting it raises `NotImplementedError` naming
the field (HTTP 422). The metric-affine perturbation eval is registered as the
CLI subcommand `noether eval4ma`, exercising the same path and checks. Other
field kinds (the rank-2 field strength, say) are refused
rather than guessed. The `adm` task is reachable the same way:
`derive_adm` (`kind="adm"` on the server, MCP, and web clients) returns the
ADM (3+1) decomposition of the gravitational sector, the Gauss-Codazzi split
and the normal/tangential projections of the Einstein tensor, verified by the
SymPy component kernel rather than Cadabra (it writes no model script); see
eval 1s. For a metric-affine NPR (independent connection with torsion and/or
non-metricity), `derive_adm` additionally produces the connection's foliation
decomposition (Gamma = LC + K(T) + L(Q) projected into normal and tangential
parts), surfaces torsion and non-metricity pieces explicitly, distinguishes
constraint pieces from evolution pieces, and identifies connection-sector
primary/secondary constraints (gated when the Dirac chain cannot close); see
eval adm-affine and section 6.5 of `docs/02_TECH_SPEC.md`. Any well-posed
action carrying a metric is accepted; one with no metric is refused (HTTP 422 /
MCP error naming the missing metric object). Each derivation carries its
convention block (signature, torsion sign, non-metricity definition,
Ricci-contraction, contortion sign, disformation sign, K-sign, foliation/normal
convention; for metric-affine NPRs also the field-strength definition) so no
convention is silently assumed; changing the elicited Ricci-contraction is
reflected in the result. The web client renders each derivation as a provenance tree
(action, plan, kernel script, every check the kernel reported, result) and
exports the kernel-verified results as a publication-LaTeX document; both are
presentation over data the server already returned, so no physics runs in the
browser. Derivations persist: each run records its result id into the session
and writes the presentation-shaped derivations into its provenance bundle, so
the same history reloads across the server (`GET /sessions/{id}/results`), MCP
(`noether_results`), and web. The cross-flow consistency is tested:
derivations returned by `POST /derive` equal those reloaded by `GET /results`,
MCP `noether_results`, and the bundle `derivations.json` field for field by
`result_id`; a gated result reads identically across HTTP and MCP; a late
resolution marks prior results stale on both surfaces; the full MCP tool chain
(ingest->resolve->plan->derive) returns blocked dicts while open and never
raises on the refusal path; and a metric-affine session resumes with geometry
resolutions, NPR version history, and result ids intact so a follow-up derive
needs no re-elicitation. A resolution that lands after results already
exist marks them stale rather than dropping or silently trusting them. Planned
next: the xAct cross-check kernel.

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
- Field-strength definition: `F_{μν} = 2∂_{[μ}A_{ν]}` (exterior derivative, `dA`); the
  covariant-curl definition `F = 2∇_{[μ}A_{ν]}` differs by `T^λ_{μν}A_λ` under torsion
  and is elicited as an alternative under an independent connection.
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
   kernel specifics stay in adapters. V0 validation stays structural, so do not
   treat raising or lowering across `\nabla` as free unless the active
   connection is explicitly metric compatible.
4. Run the relevant evals and tests. A physics-bearing change with no kernel-backed
   test does not merge.
5. Update docs touched by the change.
6. In your summary, separate "what the kernel verified" from "what I reasoned about".
   That boundary is the product's core promise; practice it in development too.
