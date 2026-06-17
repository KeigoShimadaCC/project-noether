# 02 — Technical specification

**Status:** draft.
**Scope:** the shape of the system in enough detail to start building Horizon 1.
Everything here serves `NORTH_STAR.md`; where they conflict, the North Star wins.

---

## 1. System overview

Four layers, matching the conceptual architecture in NORTH_STAR §13:

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend                                                    │
│   H1: CLI chat (LaTeX source out, optional rendered HTML)   │
│   H2+: web app (chat + provenance panel + session library)  │
└───────────────▲─────────────────────────────────────────────┘
                │ HTTPS / JSON (session events, NPR diffs)
┌───────────────┴─────────────────────────────────────────────┐
│ Orchestrator service (Python, FastAPI)                      │
│   LLM agent loop: INGEST → ELICIT → PLAN → COMPUTE →        │
│                   VERIFY → PRESENT (state machine)          │
│   Session store: NPR + transcript + artifacts (SQLite/PG)   │
└───────▲──────────────────────────────▲──────────────────────┘
        │ NPR (the only language       │ check requests
        │ crossing this boundary)      │
┌───────┴────────────────┐   ┌─────────┴────────────────────┐
│ Kernel adapters        │   │ Verification layer           │
│  cadabra2 (in-proc,    │   │  check registry, ladder      │
│   sandboxed worker)    │   │  V0..V4 (see 03_METHODOLOGY) │
│  xact (wolframscript)  │   │  uses the same adapters      │
│  sympy (in-proc)       │   │                              │
└────────────────────────┘   └──────────────────────────────┘
```

Two hard boundaries:

- **The NPR boundary.** The orchestrator never emits kernel syntax directly into
  results, and kernels never see raw user LaTeX. Everything crosses as NPR.
- **The provenance boundary.** Any expression returned to the user is wrapped in a
  result bundle (§7). There is no API to return a bare expression.

## 2. Frontend

### Horizon 1: CLI

A terminal chat client (`noether` command). Rationale: the target user lives in
terminals and TeX; a CLI proves the four-beat loop with zero frontend investment.

- Input: free text and LaTeX, multiline paste supported.
- Elicitation questions render as numbered options plus free-form answer.
- Results print as LaTeX source (copy-paste ready) and optionally write a rendered
  HTML/PDF artifact per result (`--render`).
- Every result prints its provenance pointer: path to the bundle directory
  containing scripts, assumptions, and checks.
- Sessions are named and resumable: `noether resume <session>`.

Status: `noether chat` runs the conversational loop (multiline LaTeX paste,
ingest, questions as numbered options plus free-form answers, plan once well
posed); `propose` inside the loop asks the detected agent CLI for suggestions
that take effect only when the human accepts them one by one. `noether
resume <id>` continues a stored session and `noether sessions` lists them;
the store is shared with the HTTP and MCP frontends. Arbitrary well-posed
sessions now derive through the general path (section 6, item 7) on the HTTP,
MCP, and web surfaces, which also record and reload result history; the
conversational `noether chat` loop itself stops at planning, and the eval
commands still carry their own provenance bundles. Tested in
`tests/test_chat.py` with scripted IO.

### Horizon 2+: web app

- Next.js + React, KaTeX for rendering, chat pane plus a structured side panel
  showing: current problem definition (fields, symmetries, conventions) as a live
  card; the derivation tree; verification status per result.
- The side panel is the UI expression of the NPR: the user can click any
  assumption and change it, which forks or invalidates downstream results
  explicitly (no silent recomputation).
- Export: `.tex` snippets, full provenance bundle as a zip, kernel scripts.

Status: implemented in `frontend/` (Next.js App Router, TypeScript, KaTeX;
no Tailwind, plain CSS). The home page ingests an action and lists stored
sessions; the session workspace shows the question flow on the left (options,
free-form answers, model proposals that require per-question confirmation)
and the NPR side panel on the right (problem card with the rendered action,
assumptions with a change control, event history). The browser talks only to
Next; `/api/*` is rewritten to the FastAPI server (`NOETHER_API_URL`,
default `http://127.0.0.1:8754`), so no physics state lives client-side.
The workspace also derives in place: eom, perturbation, and adm each render as
a provenance tree (action, plan, kernel script, every check the kernel
reported, then the result with its verdict), and the kernel-verified results
export as a publication-LaTeX document (copy or download). On load it reloads
any earlier derivations from the results history endpoint, flagging stale ones
and leaving them out of the export. All of this is pure presentation over data
the server already returned; the browser formats verified expressions but never
computes physics. CI builds the frontend with
type checking on every push.

### Horizon 2+: MCP server (implemented for the session surface)

Expose Noether as an MCP (Model Context Protocol) server so that Claude or any
MCP-capable agent can delegate tensor calculus to it the way it delegates
arithmetic to a Python sandbox. The host LLM does the conversation and planning;
Noether does the algorithmic, kernel-backed part and returns verified results.

- Tools map onto the existing orchestrator surface, not onto kernels directly:
  `ingest_action` (LaTeX in, NPR + open ambiguity ledger out), `resolve_ambiguity`,
  `derive` (vary/reduce/adm/perturb), `verify` (run the ladder on a claimed
  result), `render` (canonical LaTeX out).
- Provenance bundles are exposed as MCP resources, so the host agent can quote
  the receipt, not just the answer.
- The no-guessing contract survives the protocol: `derive` fails with the open
  ambiguity list until the host (or its human) resolves them. A host LLM cannot
  make Noether guess.
- This is a frontend in the §2 sense: a thin adapter over the same session API
  that drives the CLI and web app. No physics logic lives in it.

Status: `noether.mcp.create_mcp_server` (behind the optional `[mcp]` extra;
`noether mcp` runs it over stdio) exposes the session surface as tools:
`noether_ingest`, `noether_session(s)`, `noether_resolve`,
`noether_propose_definitions`, `noether_adopt_definitions`, `noether_plan`,
`noether_derive`, `noether_kernels`. Refusals are tool results, not exceptions:
`noether_plan` and `noether_derive` return `blocked=true` with the open
questions until the problem is well posed, off-menu resolutions are rejected
without mutating the session, and the tool instructions direct the host to
relay questions to its human. `noether_derive` runs the general derivation
(section 6, item 7): it returns each result with a `verified` flag the kernel
sets, never the host. `kind="eom"` (the default) varies the action; for the
scalar and metric sectors `kind="perturbation"` expands it to quadratic order
instead; and `kind="adm"` returns the ADM (3+1) decomposition of the
gravitational sector, verified by the SymPy component kernel.
The `verify`/`render` tools land as those compute surfaces are built out.
Tested in `tests/test_mcp.py` (skips without the extra).

The frontend is deliberately thin. All physics state lives server-side in the NPR
and session store; the same API drives CLI, web, and MCP.

### Status: the HTTP session API is implemented

`noether.server.create_app` (FastAPI, behind the optional `[server]` extra;
`noether serve` runs it) exposes the orchestrator loop over HTTP with the
no-guessing contract intact: `POST /sessions` ingests an action and returns the
open question ledger; `POST /sessions/{id}/elicit` returns UNCONFIRMED model
proposals (off-menu suggestions already discarded; 503 when no agent CLI is
detected); only `POST /sessions/{id}/resolve`, validated against the listed
options, mutates the session; `GET /sessions/{id}/plan` returns 409 with the
open questions until the problem is well posed. `GET /sessions/{id}/definitions`
proposes readability shorthands for the derivatives of function couplings
(notation, not results, see section 3.1) and `POST /sessions/{id}/definitions`
adopts the accepted ones. `POST /sessions/{id}/derive` runs the general
derivation (section 6, item 7) for a well-posed session and returns each field
equation with the kernel's `verified` verdict; it answers 409 with the open
questions while any remain, and 503 when the Cadabra kernel or an agent CLI is
missing on the server. Each derivation records its result id into the session
and stores the presentation-shaped derivations in its provenance bundle, so
`GET /sessions/{id}/results` reloads the full history (with `stale_result_ids`
naming any result a later resolution invalidated) without re-running a kernel;
the MCP `noether_results` tool returns the same shape. Sessions persist as JSON
through `noether.orchestrator.store.SessionStore` and are shared by CLI, web,
and MCP frontends. Tested in `tests/test_server.py` (skips without the extra).

## 3. Orchestrator

### 3.1 Agent loop

Built on a swappable model backend behind our own `LLMAdapter` interface
(`noether.llm`). The implemented backend is ambient-auth, no API key: it
auto-detects an installed agent CLI (codex, claude, gemini, droid) and runs it
one-shot as a sandboxed subprocess, mirroring the cadabra transport, so
credentials stay in that CLI's own login session. A `StubLLMAdapter` makes the
plumbing deterministic in tests. The LLM gets tools, not freedom:

- `parse_latex(action_tex) -> NPR draft + ambiguity list`. Implemented as two
  deterministic layers: `noether.npr.parse` (purely syntactic LaTeX -> NPR Expr,
  no physics inference) and `noether.orchestrator.ingest` (syntactic object
  discovery plus the ambiguity ledger). Ingest never assigns field roles,
  conventions, the fields to vary, or the curvature/connection/coupling meaning
  of a symbol; it emits each as an open question, so a freshly ingested action is
  structurally un-plannable until elicitation resolves it. The LLM narrates and
  may propose answers, but cannot make ingest guess. Validated against the five
  acceptance actions (`tests/test_parse.py`, `tests/test_ingest.py`); reachable
  from the CLI as `noether ingest "<lagrangian>"`.
- `ask_user(questions) -> answers` (elicitation; see 03_METHODOLOGY §1).
  Implemented in `noether.orchestrator.elicit` with a propose-then-confirm
  contract that makes AGENTS.md rule 4 structural: `propose_resolutions` asks the
  model to pick one listed option per open ambiguity, validates each suggestion
  against the allowed options (off-menu answers are discarded, never guessed),
  and returns suggestions plus model provenance without mutating the NPR. Only
  `apply_resolutions`, given human-confirmed choices, sets resolutions and
  unblocks planning. Reachable as `noether elicit "<lagrangian>"`; the explicit
  `--accept-llm` flag delegates confirmation to the model. Tested against all
  five acceptance actions (`tests/test_llm.py`, `tests/test_elicit.py`).
- `plan(task, npr) -> computation plan` (a DAG of kernel-task nodes)
- `run_kernel(kernel, task, npr) -> npr_expression + raw artifacts`
- `verify(result, checks) -> verdicts`
- `present(result_bundle) -> user-facing LaTeX + narrative`

The model plans, sequences, interprets, and narrates. It cannot inject an
expression into a result bundle; only `run_kernel` outputs can land there. This is
the mechanical enforcement of "no unearned assertions".

### 3.2 Session state machine

```
INGEST   user provides action; LLM-assisted parse to draft NPR
ELICIT   resolve ambiguity list via questions; NPR becomes well-posed
PLAN     choose kernels and step DAG for the requested task
COMPUTE  execute DAG; manage retries, term explosion strategy
VERIFY   run the check ladder appropriate to the result class
PRESENT  LaTeX + narrative + provenance; await next ask
```

Any user message can move the machine backwards (changing an assumption returns
to ELICIT and marks downstream results stale). State transitions are events,
persisted, so the session is replayable.

### 3.3 Statefulness

Session record = `{npr_versions[], transcript[], results[], artifacts[]}` in
SQLite (single user, H1) with a straight upgrade path to Postgres. NPR versions
are immutable; a change creates a new version with a diff, and results reference
the NPR version they were computed against. That gives resumability and an honest
answer to "what assumptions was this derived under?"

## 4. The NPR (Noether Problem Representation)

The backend-agnostic contract between "what the physicist meant" and "what any
kernel executes". JSON, versioned schema (`pydantic` models), designed for diffing
and human inspection.

Top-level shape:

```json
{
  "npr_version": "0.1",
  "conventions": {
    "id": "noether-default-v1",
    "dimension": 4,
    "signature": "mostly-plus",
    "riemann_sign": "+1",
    "ricci_contraction": "first-third",
    "symmetrization_weight": "1/n!"
  },
  "geometry": {
    "manifold": {"dim": 4, "coordinates": "abstract"},
    "metric": {"name": "g", "symmetry": "symmetric", "role": "dynamical"},
    "connection": {"type": "levi-civita"}
  },
  "objects": [
    {"name": "phi", "kind": "scalar-field", "role": "dynamical"},
    {"name": "K", "kind": "function", "args": ["phi", "X"], "role": "coupling"},
    {"name": "X", "kind": "shorthand",
     "definition_tex": "-\\tfrac12 \\nabla_\\mu\\phi\\nabla^\\mu\\phi"}
  ],
  "action": {
    "measure_tex": "d^4x \\sqrt{-g}",
    "lagrangian_ast": { "...expression tree, abstract indices..." },
    "lagrangian_tex": "K(\\phi,X) + G(\\phi,X)\\Box\\phi + F(\\phi)R"
  },
  "task": {
    "type": "vary",
    "with_respect_to": ["g", "phi"],
    "target_form": {"basis": "curvature-canonical", "collect_by": "tensor-structure"}
  },
  "ambiguities": []
}
```

Key design rules:

- **Expression AST, not strings.** The Lagrangian is stored as a tree of typed
  nodes (sum, product, tensor, derivative, function) with abstract indices and
  declared symmetries. `lagrangian_tex` is a cached rendering, never the source of
  truth after ELICIT completes.
- **`connection.type`** ∈ {`levi-civita`, `independent`} with flags
  `torsion: bool`, `nonmetricity: bool` for the independent case. This single field
  is what separates eval 1 from eval 2.
- **`ambiguities`** is a first-class list. INGEST fills it; ELICIT must empty it
  before PLAN may run. A non-empty ambiguity list is a type-level block on
  computation: guessing is structurally impossible, not just discouraged.
- **Round-trip law:** `render_tex(parse(npr)) ≡ npr` must hold (tested). The LLM
  may propose a parse; a deterministic validator checks index balance, symmetry
  consistency, and dimension homogeneity before the NPR is accepted.

The frozen schema gets its own doc (`05_NPR_SCHEMA.md`) once v0.1 stabilizes.

## 5. Kernel adapters

One interface, N implementations:

```python
class KernelAdapter(Protocol):
    name: str
    version: str
    def capabilities(self) -> set[Capability]: ...
    def compile(self, task: KernelTask, npr: NPR) -> KernelScript: ...
    def execute(self, script: KernelScript, timeout: int) -> KernelRawOutput: ...
    def parse_output(self, raw: KernelRawOutput) -> NPRExpression: ...
```

- `Capability` is an enum: `VARY`, `IBP`, `CANONICALIZE`, `SUBSTITUTE`,
  `PERTURB`, `ADM`, `COMPONENT_EVAL`, `INDEPENDENT_CONNECTION`, ... The planner
  selects kernels by capability, never by name, so adding a kernel is additive.
- `compile` is deterministic and template-driven per task type. The LLM does not
  write kernel scripts character by character in production; it selects and
  parameterizes audited templates. (During development, new templates are born
  from LLM drafts, then reviewed, golden-tested, and frozen.)
- `execute` runs in a sandboxed worker (separate process, resource limits, no
  network), pinned kernel version, captured stdout/stderr. Scripts and raw output
  are archived verbatim into the provenance bundle.
- `parse_output` lifts kernel output back into NPR expressions and re-validates
  (index balance, declared symmetries). A parse failure is a hard error, never a
  best-effort guess.

Initial adapters and their jobs (rationale in `01_RESEARCH.md`):

| Adapter | Transport | H1 jobs | H2+ jobs |
|---|---|---|---|
| cadabra2 | sandboxed subprocess (`cadabra2` CLI; in-process embedding optional later) | vary, IBP, canonicalize, substitute, independent connection | identity reduction depth |
| sympy | in-process | scalar algebra, dimension checks, component eval on explicit backgrounds | random-metric spot checks |
| xact | `wolframscript` subprocess | (off) | xPert perturbation, ADM support, cross-check canonical forms |

## 6. Core algorithms and strategies

What we rely on kernels for vs. what Noether itself implements:

**Kernel-owned (never reimplemented):** Butler-Portugal canonicalization,
variational derivatives with IBP, Bianchi/Ricci identity application,
perturbative expansion (xPert), Young projection.

**Noether-owned:**

1. **LaTeX → NPR parsing.** LLM proposes the parse (it is genuinely good at messy
   physicist LaTeX); a deterministic validator enforces well-formedness; failures
   and underdetermined choices become `ambiguities` entries for ELICIT. The LLM
   is a parser-assistant, not an authority.
2. **Plan construction.** Task → DAG of kernel steps. Example for `vary` w.r.t.
   metric: expand composite shorthands → distribute variation → IBP to strip
   derivatives off `δg` → collect surface terms → canonicalize → identity-reduce →
   project to target basis → collect. Each node names its kernel capability.
3. **Expansion management** (the anti-explosion strategy):
   - canonicalize and merge after every step, not at the end;
   - hash-cons terms by canonical form so duplicates collapse early;
   - collect by tensor structure (all coefficients of `R_{μν}∇^μφ∇^νφ` together)
     rather than holding flat term lists;
   - thresholds: when an intermediate exceeds N terms, the planner inserts an
     extra reduce step or splits the computation by structure sector, and reports
     progress honestly instead of hanging.
4. **Good-form pipeline.** Deterministic finishing pass: canonical index order,
   dummy renaming, sign normalization, chosen-basis projection, stable term
   ordering. Same NPR in, byte-identical LaTeX out. Negotiable targets (basis,
   collecting variable) come from the NPR `target_form`.
5. **Equality checking.** Two expressions are equal iff their canonical forms
   match after identity reduction; fallback falsifier: evaluate both on
   pseudo-random explicit backgrounds (sympy adapter) to catch canonicalization
   gaps. Used heavily by the verification layer.
6. **ADM and perturbation orchestration (H2).** Foliation/gauge data enters the
   NPR; the planner drives xPert/xCoba (or Cadabra rule sets) and the same
   good-form pipeline finishes the output. Algorithms are kernel-side; sequencing
   and presentation are ours.
7. **General derivation for arbitrary actions** (`noether.orchestrator.derive`,
   `noether.kernels.cadabra.generate`). The frozen golden templates only cover
   the eval actions. For any other well-posed action, the model parameterizes a
   Cadabra script instead of selecting a template: `generate_script` hands it
   the matching audited template as a worked example and a contract that the
   script must derive the equation of motion by `vary()` and then state an
   independent candidate equation, so the kernel can compute the residue and
   print `residue_zero`. `derive_field` runs that script and trusts the result
   only when the kernel reports `residue_zero=True`; anything else comes back
   marked unverified and is surfaced as such, never as truth. Every run, verified
   or not, writes a provenance bundle. The bright line holds: the model writes a
   script, the kernel decides whether the answer is trustworthy. This covers the
   `vary` task (equations of motion) for the metric, scalar, and gauge-field
   classes today. The general path is gated by `evals/test_eval_general.py`,
   which checks it reproduces eval 3's two kernel-verified equations of motion
   end to end. The `vary` task also has a compositional path that needs no
   model: when an additive Lagrangian decomposes fully into registered building
   blocks, `derive_field` assembles one Cadabra script for the actual action and
   an independent candidate from the same blocks, and the kernel residue-checks
   it (`noether.kernels.cadabra.blocks`). The scalar EOM blocks are canonical
   kinetic, potential, cubic Galileon, k-essence `K(phi, X)`, and the nonminimal
   `F(phi) R` term (eval 7 and 8); the metric EOM blocks are Einstein-Hilbert,
   nonminimal `F(phi) R`, kinetic, potential, and cubic Galileon `G(phi) box phi`
   (eval 8 and 6). So the full nonminimal scalar-tensor theory and the cubic
   Galileon yield both equations of motion compositionally. This is
   the non-tailored route to the general scalar Horndeski sector: any sum of
   registered blocks verifies without a new template, and an unrecognized term
   (an `X`-dependent `G4(phi, X) R`, say) leaves the decomposition partial so the
   model-written path runs instead. The `perturb` task now runs through the same model-written path:
   `derive_perturbation` (and `kind="perturbation"` on the server, MCP, and web
   clients) hands the model a scaffold chosen by field kind:
   `pert_scalar_quadratic` (eval 3p) expands a scalar action to quadratic order,
   and `pert_metric_quadratic` (eval 3g) expands the Einstein-Hilbert action
   about a flat background to recover the linearized vacuum Einstein equation,
   the massless graviton. A rank-1 gauge potential routes to the gauge
   scaffolds: `pert_gauge_quadratic` (eval 3a) for an abelian potential gives
   the source-free linearized Maxwell operator `nabla_mu f^{mu nu}` behind the
   photon's two transverse polarizations, and `pert_yang_mills_quadratic`
   (eval 3y) handles a non-abelian `gauge_group`, adding the
   background-covariant derivative and the gluon self-coupling
   `g f^{abc} v^b Fbar^c` to that operator. Every scaffold uses Cadabra weights
   to track fluctuation order and checks the linearized equation of motion
   twice, against the documented operator and by an independent route
   (linearizing the full nonlinear equation, or rebuilding the linearized
   Einstein tensor from the linearized Christoffels). All checks must pass
   before the result is called verified.
   The metric expansion has two extra wrinkles the scaffold handles: a second
   derivative comes off the test field through two `integrate_by_parts` passes
   with `\nabla` as a `::Derivative`, and equal terms written at different index
   heights only meld after everything is lowered to one explicit-`eta`
   convention and `\nabla` is rewritten as a commuting `::PartialDerivative`.
   The Yang-Mills scaffold carries its own: adjoint indices are a second index
   group with a Killing metric and `position=independent` so the totally
   antisymmetric structure constants contract and collapse, and the independent
   linearization route keeps weight `eps=2` (the test field carries `eps=1`)
   then expands `nabla(Abar v)` by `product_rule` before the cross-check. The
   scaffolds cover dynamical scalar fields, the metric, and rank-1 gauge
   potentials, so `derive_perturbation` refuses other field kinds (the rank-2
   field strength, say) rather than guessing. The `adm` task takes a different route: `derive_adm`
   (`kind="adm"` on the server, MCP, and web clients) writes no model script.
   Its deliverable is the ADM (3+1) decomposition of the gravitational sector,
   the Gauss-Codazzi split `sqrt(-g) R = N sqrt(h)(R3 + K_ij K^ij - K^2) -
   2 d_mu(sqrt(-g) v^mu)` and the normal/tangential projections of the Einstein
   tensor, which are universal foliation geometry independent of the action.
   The SymPy component kernel verifies all six identities (the split, both
   projections, the extrinsic-curvature identity K_ij = nabla_i n_j, and the
   lapse Euler-Lagrange equation) on a nondegenerate 1+2 background (eval 1s);
   `verified` is set from that suite. Any well-posed action carrying a metric is
   accepted; one with no metric is refused.

### 6.1 Representation boundaries (Horndeski as the worked stress case)

Running a Horndeski action through the pipeline exposed where ingest stops
guessing and where the verified path runs out of scaffold. Two fixes landed in
ingest and the parser, and one honesty fix in `derive_field`:

- The parser reads subscripted coupling names, so the standard `G_2(\phi,X)`
  through `G_5(\phi,X)` parse as functions with compound names rather than
  failing on the `_`. A subscripted name with no argument list is still
  refused, because nothing then says it is a function.
- Ingest treats a bare `X` as the convention-named kinetic shorthand
  `X = -1/2 nabla_mu phi nabla^mu phi` (section 5 of AGENTS.md) when the action
  carries exactly one dynamical scalar and `X` only ever appears as a plain
  scalar. So `X` is no longer offered as an independent field to vary. The
  reading is still put to the human as `amb-kinetic-X`; answering
  "independent-field" reclassifies `X` back to a dynamical scalar. With no
  scalar to anchor it, `X` is left alone rather than guessed.
- An unverified run now says which way it failed: a script that never reached
  the kernel's residue check (a script or kernel error, with the stderr tail)
  reads differently from one that ran and found a nonzero residue.

The first Horndeski member past scalar-tensor now verifies: the cubic Galileon
scalar sector `S = \int d^4x \sqrt{-g}(-1/2 (nabla phi)^2 - V + K(phi) box phi)`,
template `eom_cubic_galileon_scalar` (eval 6). The new mechanics over eval 3 are
the `box phi` coupling: variation splits `K box phi`, a two-pass
`integrate_by_parts` peels the second derivative off the test field, and the
coupling chain rule `nabla K -> K' nabla phi` reintroduces `phi`-derivatives when
`box` lands on `K`. `generate.py` routes a scalar action with a `box`-coupling to
this scaffold instead of the plain scalar example.

The general scalar sector then moved off per-theory scaffolds entirely
(`noether/kernels/cadabra/blocks.py`, eval 7). An additive scalar Lagrangian is
decomposed into building blocks (canonical kinetic, potential, cubic Galileon,
k-essence `K(phi, X)`), and one Cadabra script is assembled for the action the
user entered: the real integrand plus an independent candidate built from the
same blocks. The kernel's residue check then verifies that assembled action, so
trust still comes from the kernel, not from summing pre-blessed formulas. This
is the non-tailored path: any sum of registered blocks, with arbitrary coupling
names, verifies without a new template; adding a block extends coverage. It also
closes the `X` gap, because the k-essence block expands `X = -1/2 (nabla phi)^2`
to its primitive in the kernel through `nabla_mu X = -nabla_mu nabla_nu phi
nabla^nu phi`, then collapses `X` and `box` back to shorthand for display (the
operational-definition path). When a term matches no block, the decomposition is
left partial and `derive_field` falls back to the model-written script path
rather than guessing.

Composition then reached the metric sector and the first curvature coupling
(eval 8). The same module decomposes an additive Lagrangian into metric-sector
blocks (Einstein-Hilbert `R`, nonminimal `F(phi) R`, kinetic, potential) and
assembles one script for the metric equation of motion, reusing the eval-3
machinery: vary `g` and the Ricci tensor into `dGamma`, two `integrate_by_parts`
passes to peel the derivatives off `h`, and lower `h` to one explicit-`g`
convention before the residue check. So the full nonminimal scalar-tensor theory
now yields both equations of motion compositionally, with no per-theory
template, and a vacuum action (`R` alone) verifies as the Einstein tensor. Each
metric block was confirmed against the kernel before wiring (Einstein-Hilbert
alone returns `G_{mu nu}`; `F(phi) R` + kinetic + potential reproduces eval 3's
residue).

The cubic Galileon `G(phi) box phi` then joined the metric blocks (eval 6). Its
metric variation needs one step the others do not: the second covariant
derivative of the field varies as `delta(nabla_a nabla_b phi) = -dGamma^l_{ab}
nabla_l phi`. The assembler carries `nabla_a nabla_b phi` as a symmetric stand-in
`Hess_{ab}` so `vary` can take that variation, restores it afterward, and runs
the same two-pass IBP with the coupling chain rule `nabla G -> G' nabla phi`. The
`G nabla nabla phi` pieces cancel and the residue checks against the cubic stress
`-G' nabla_mu phi nabla_nu phi + 1/2 G' g_{mu nu}(nabla phi)^2`, the kinetic
stress with coupling `-G'` (since `G box phi = -G' (nabla phi)^2` up to a
boundary term). So the cubic Galileon coupled to Einstein gravity yields both
equations of motion compositionally.

What still does not verify: the higher Horndeski densities (an `X`-dependent
`G4(phi, X) R`, and G5) have no block and no audited template. Their covariant
equations of motion carry curvature derivatives and the no-Ostrogradski
counterterms, so they are held out until they residue-check cleanly rather than
added as a partial path; a term like `G(phi, X) R` matches no block and falls
back to the model path or is refused. The `perturb` scaffolds likewise do not
expand `X`, so a quadratic action for an `X`-dependent coupling is still
unverified. The ADM split verifies for any metric action, but it is the
universal foliation geometry, not
a Horndeski-specific Hamiltonian.

## 7. Provenance bundles

Every result is a directory (and a DB row pointing at it):

```
results/<session>/<result-id>/
  result.json        final NPR expression + rendered LaTeX
  assumptions.json   NPR version snapshot (conventions, roles, symmetries)
  plan.json          the executed DAG, per-node kernel + capability
  scripts/           exact kernel scripts, as executed
  raw/               kernel stdout/stderr, versions, timings
  checks.json        verification ladder verdicts (V0..V4, see 03_METHODOLOGY)
  derivations.json   presentation-shaped FieldDerivation records, so result
                     history reloads without re-running a kernel
  narrative.md       the human-readable derivation story shown to the user
```

Reproduction contract: `noether reproduce <result-id>` reruns `scripts/` against
pinned kernel versions and diffs canonical forms. CI runs this for the eval corpus.

## 8. Technology stack summary

| Concern | Choice | Notes |
|---|---|---|
| Service language | Python 3.12+ | matches Cadabra2/SymPy embedding |
| API | FastAPI | thin; sessions are event streams |
| Agent SDK | Claude Agent SDK (or equivalent) | behind our `Orchestrator` interface |
| Schema/validation | pydantic v2 | NPR models, versioned |
| Kernels | Cadabra2, SymPy (H1); Wolfram Engine + xAct (H2) | pinned, containerized |
| Storage | SQLite → Postgres | sessions, NPR versions, result index |
| Frontend H1 | CLI (Python, rich/textual) | LaTeX source out, optional HTML render |
| Frontend H2 | Next.js + React + KaTeX | thin client over the same API |
| Packaging | Docker images per kernel | reproducibility is a product feature |
| CI | pytest + eval corpus + `reproduce` runs | physics changes need kernel-backed tests |

## 9. Security and operational notes

- Kernel workers: no network, CPU/memory/time limits, throwaway filesystem except
  the artifact mount. Kernel scripts are generated from audited templates, which
  bounds the injection surface; raw user LaTeX never reaches a kernel.
- LLM calls carry no secrets: the implemented adapter shells out to an agent CLI
  whose credentials live in its own login session, so Noether holds no API key.
  Session content is the user's research and is treated as confidential (no
  training, no third-party logging). Caveat: agent CLIs are built for interactive
  use, so programmatic headless use may bump against their terms; fine for a
  personal research tool, revisit before any distribution.
- Determinism: pinned kernel versions, pinned model versions recorded per result,
  seeded randomness in spot checks. "Same bundle, same answer" is a test.

## 10. Open questions

1. Agent SDK choice and how much of the loop to hand it vs. own ourselves.
2. NPR expression AST: design our own minimal node set (likely) vs. adopting an
   existing tree format and constraining it.
3. Cadabra2 worker model: long-lived kernel process per session (fast, stateful,
   riskier) vs. fresh process per step (slow, clean). H1 starts fresh-per-step;
   revisit with profiling.
4. Wolfram licensing path for hosted use (blocker for H2 architecture freeze).
5. How `target_form` should express user-specific "good form" preferences beyond
   basis + collection (per-user convention profiles?).
