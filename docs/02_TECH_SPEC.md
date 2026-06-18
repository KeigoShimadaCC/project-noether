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
  Geometry questions stay menu-bound in the chat loop, so numbered picks and
  typed on-menu answers use the same confirmation path as the HTTP resolve
  surface before the session store is updated.
- Results print as LaTeX source (copy-paste ready) and optionally write a rendered
  HTML/PDF artifact per result (`--render`).
- Every result prints its provenance pointer: path to the bundle directory
  containing scripts, assumptions, and checks.
- Sessions are named and resumable: `noether resume <session>`.

Status: `noether chat` runs the conversational loop (multiline LaTeX paste,
ingest, questions as numbered options plus free-form answers, plan once well
posed); for the metric-affine questionnaire, connection type, torsion,
non-metricity, metric compatibility, the follow-up Ricci-contraction
choice, and the field-strength-definition question (when a vector/gauge
potential exists on an independent-connection background) all resolve
through the same on-menu confirmation path used by the HTTP
surface, so `chat` and `resume` persist the same geometry state before
planning. `propose` inside the loop asks the detected agent CLI for suggestions
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
proposes readability shorthands for the derivatives of function couplings and,
on a metric-affine NPR, the post-Riemannian notation `K(T)`, `L(Q)`, and the
`f(Q)` scalar `Q` (notation, not results, see section 3.1). `POST
/sessions/{id}/definitions` adopts the accepted ones. `POST
/sessions/{id}/derive` runs the general
derivation (section 6, item 7) for a well-posed session and returns each field
equation with the kernel's `verified` verdict; it answers 409 with the open
questions while any remain, and 503 when the Cadabra kernel or an agent CLI is
missing on the server. Each derivation records its result id into the session
and stores the presentation-shaped derivations in its provenance bundle, so
`GET /sessions/{id}/results` reloads the full history (with `stale_result_ids`
naming any result a later resolution invalidated) without re-running a kernel;
the MCP `noether_results` tool returns the same shape. The cross-surface
consistency is verified: derivations returned by `POST /derive` equal those
reloaded by `GET /results`, MCP `noether_results`, and the bundle
`derivations.json` field for field by `result_id`; every derivation carries a
non-empty `detail` (a confirmation reason when `verified=True`, a blocker when
`verified=False`), and a gated result is distinguishable from a verified one by
both `verified` and `detail` across all surfaces; a gated result reads
identically (same `verified`, `detail`, `checks`) across HTTP and MCP;
blocked/refused MCP calls return error/blocked dicts, never a fabricated
verified result; a late
resolution marks prior results stale on both surfaces; and a metric-affine
session resumes with geometry, NPR version history, and result ids intact.
Sessions persist as JSON
through `noether.orchestrator.store.SessionStore` and are shared by CLI, web,
and MCP frontends. Tested in `tests/test_server.py` and
`tests/test_cross_flows.py` (both skip without the extra).

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
  structurally un-plannable until elicitation resolves it. On the metric-affine
  path, the geometry questionnaire opens whenever the action carries curvature or
  other connection-dependent geometry, not just an explicit `(\Gamma)` suffix:
  ingest now asks on-menu questions for connection type, torsion,
  non-metricity, and metric compatibility, with no answer pre-selected, while a
  curvature-free scalar action such as k-essence keeps the default
  Levi-Civita draft and raises none of those geometry questions. When the
  action uses torsion or non-metricity but no curvature (e.g. `f(T)` or
  `f(Q)`), ingest also raises a curvature-free question: whether the
  connection is constrained to be curvature-free (teleparallel or symmetric
  teleparallel geometry) or whether curvature is still allowed. The
  `curvature_free` flag on `ConnectionSpec` distinguishes the teleparallel
  family (metric-compatible, torsionful, curvature-free) from Riemann-Cartan
  (same torsion flags but curvature not constrained), and similarly for
  symmetric teleparallel vs general non-metric connections. When the
  action carries an explicit connection (e.g. `R_{mu nu}(Gamma)`), ingest adds
  a `Gamma` connection object (kind=`connection`, role=`dynamical`) to the
  objects list so the derive path can vary it; the `amb-vary-wrt` options
  include connection-kind objects, and a compound `g and Gamma` option is
  prepended when both a metric and a connection are present. The LLM
  narrates and may propose answers, but cannot make ingest guess. Validated
  against the five acceptance actions (`tests/test_parse.py`,
  `tests/test_ingest.py`); reachable from the CLI as `noether ingest
  "<lagrangian>"`.
- `ask_user(questions) -> answers` (elicitation; see 03_METHODOLOGY §1).
  Implemented in `noether.orchestrator.elicit` with a propose-then-confirm
  contract that makes AGENTS.md rule 4 structural: `propose_resolutions` asks the
  model to pick one listed option per open ambiguity, validates each suggestion
  against the allowed options (off-menu answers are discarded, never guessed),
  and returns suggestions plus model provenance without mutating the NPR. The
  inference prompt (`build_elicitation_prompt`) embeds the action's geometric
  cues (presence of `R(\\Gamma)`, explicit `T`/`Q`, `f(Q)`/`f(T)` family) so
  the model's proposed geometry choices are grounded in the action, not a fixed
  default; a scalar action carries no such geometry cue (VAL-GUIDE-017). Only
  `apply_resolutions`, given human-confirmed choices, sets resolutions and
  mutates the dependent NPR fields. On the metric-affine path that means
  `geometry.connection` is updated from the confirmed menu answers, off-menu
  answers raise rather than slipping through, and choosing an independent
  connection opens a follow-up Ricci-contraction convention question and, if
  a vector/gauge potential is present, a field-strength-definition question
  (`F = dA` vs `F = nabla A`, differing by torsion per VAL-GEOM-020) before
  planning can continue. Convention proposals (Ricci-contraction, field-strength
  definition) are on-menu with rationale and never auto-applied; an off-menu
  convention proposal is nulled (VAL-GUIDE-020). Geometry inference is exercised
  deterministically with `StubLLMAdapter`: every non-null proposed choice is in
  the ambiguity's options, off-menu suggestions yield `choice is None` (rationale
  may survive), and after `propose_resolutions` the NPR is unchanged (not
  well-posed, geometry ambiguities unresolved, `geometry.connection` unchanged).
  Only a human-confirmed on-menu answer mutates geometry via `apply_resolutions`;
  off-menu and unknown-ambiguity-id confirmations raise `ValueError` and never
  mutate the NPR. On the HTTP surface, `POST /elicit` returns
  `confirmed: false` with proposals (off-menu nulled) leaving the ambiguity
  unresolved; mutation happens only after an on-menu `POST /resolve`; an
  off-menu resolve returns 400. Reachable as `noether elicit "<lagrangian>"`;
  the explicit `--accept-llm` flag delegates confirmation to the model. Tested
  against all five acceptance actions (`tests/test_llm.py`,
  `tests/test_elicit.py`) and the geometry-specific inference contract
  (`tests/test_geometry_inference.py`, VAL-GUIDE-001..007, VAL-GUIDE-017,
  VAL-GUIDE-020).
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
    "field_strength_definition": "exterior-derivative",
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
   and underdetermined choices become `ambiguities` entries for ELICIT. On the
   metric-affine path that validator stays structural: it checks index balance,
   but it does not commute the metric through `\nabla` unless metric
   compatibility was confirmed explicitly. The LLM is a parser-assistant, not an
   authority.
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
   `vary` task (equations of motion) for the metric, scalar, connection, and
   gauge-field classes today. Connection variation (`wrt` a `connection` object)
   routes to the `vary-connection` worked example (`eval2_palatini_connection`)
   and uses `Capability.INDEPENDENT_CONNECTION` rather than the generic `VARY`,
   so a connection field is never silently routed to the metric worked example.
   For the pure Palatini Einstein-Hilbert action (no matter fields other than
   the metric and the independent connection), the derive path routes directly
   to the frozen `eval2_palatini_connection` template, surfacing the verified
   projective-family result (checks `solution_zero` and `ricci_shift_is_dA`)
   with a payload that states the projective freedom (`Gamma = LC(g) +
   delta^lam_nu A_mu`, `A_mu` arbitrary) and never presents the connection as
   uniquely fixed (VAL-EOM-004). Non-pure-EH connection variations (Palatini
   scalar-tensor, Einstein-Cartan) still route through the general LLM-written
   script path.
   When the connection is independent and the metric is varied, the metric
   variation routes to the `vary-metric-palatini` worked example
   (`eval2_palatini_metric`) instead of the standard `vary-metric` one, because
   the curvature R_{mu nu}(Gamma) depends on the independent connection and
   must NOT be varied with the metric (no dGamma terms, no integrate_by_parts
   steps). The Palatini metric variation is algebraic: only g^{sigma nu} and
   sqrt(-g) vary, and the resulting field equation is the symmetrized
   R_{(mu nu)}(Gamma) - 1/2 g_{mu nu} Rtilde = 0, with both R_{mu nu} and
   R_{nu mu} appearing explicitly because the independent-connection Ricci
   carries no symmetry declaration. The projective mode Gamma = LC(g) +
   delta^lam_nu A_mu annihilates the connection equation identically, and the
   Ricci shift R(Gamma + proj) - R(Gamma) is exactly dA (the exterior
   derivative of A_mu), so the symmetric-part metric equation is
   projective-invariant for any starting connection. The Cadabra adapter
   advertises `INDEPENDENT_CONNECTION` in its capabilities. When the geometry
   connection is independent, `derive_eom` includes connection-kind objects in
   its default field list, so both the metric and connection equations are
   derived by default on a Palatini session; the MCP/HTTP `with_respect_to`
   parameter can override this to derive a subset of fields.

   The multi-field Palatini scalar-tensor action S = ∫√{-g} F(φ) R(Γ) extends
   this to three independent variations: metric, connection, and scalar. The
   metric variation gives the symmetrized Ricci equation (F × [R_{(μν)} - ½
   g_{μν} R̃] = 0), the connection variation yields the dF non-metricity
   source (∇_λ(√{-g} F g^{σν}) - δ^ν_λ ∇_ρ(√{-g} F g^{σρ}) = 0, where ∂F
   = F_φ ∂φ couples the scalar sector to the connection sector), and the
   scalar variation gives F_φ R̃(Γ) = 0. The dF source means the projective
   mode does NOT solve the connection equation when F is non-constant (unlike
   pure EH). All three templates compute an independent target and assert a
   genuine NOETHER_CHECK residue-zero sentinel: the metric and scalar targets
   are the expected EOM expressions; the connection target is the
   Euler-Lagrange equation in partial-derivative form (not the
   covariant-derivative form, because the covariant-divergence IBP theorem
   does not hold for an independent connection). The
   connection-variation IBP boundary-term assumption (δΓ vanishes on the
   boundary) is recorded explicitly in the Cadabra script and kernel output,
   not silently dropped. At T=Q=0, the metric-affine metric equation reduces
   to the Levi-Civita result F × G_{μν} = 0, residue-pinned and
   SymPy-confirmed; the connection equation at F=const, Γ=LC vanishes
   identically (metric compatibility), also SymPy-confirmed componentwise.
   The phi-dependent dF source term (`√{-g} F_φ [g^{γβ} ∂_α φ
   - δ^α_β g^{γρ} ∂_ρ φ]`) that couples the scalar and connection sectors
   is confirmed numerically by the SymPy oracle on explicit random
   metric and connection backgrounds (multiple seeds, both linear F = 1+cφ
   and nonlinear F = φ²), closing the gap where previously only the
   F=const sub-case had a SymPy cross-check.
   Three Cadabra templates (`palatini_st_metric`, `palatini_st_connection`,
   `palatini_st_scalar`) are registered in `templates.py` and verified in
   `tests/test_palatini_scalar_tensor_affine.py`.

   The general path is gated by `evals/test_eval_general.py`,
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
   `pert_kessence_quadratic` (eval 3k) does the same for an `X`-dependent scalar
   `K(phi, X)`, expanding `X` itself so the quadratic action carries the
   sound-speed kinetic mixing `K_XX (nabla phibar . nabla chi)^2`, and
   `pert_metric_quadratic` (eval 3g) expands the Einstein-Hilbert action
   about a flat background to recover the linearized vacuum Einstein equation,
   the massless graviton. A rank-1 gauge potential routes to the gauge
   scaffolds: `pert_gauge_quadratic` (eval 3a) for an abelian potential gives
   the source-free linearized Maxwell operator `nabla_mu f^{mu nu}` behind the
   photon's two transverse polarizations, and `pert_yang_mills_quadratic`
   (eval 3y) handles a non-abelian `gauge_group`, adding the
   background-covariant derivative and the gluon self-coupling
   `g f^{abc} v^b Fbar^c` to that operator. On a metric-affine
   (independent-connection) background, an abelian gauge potential routes to
   the torsion-aware vector-affine templates instead: `pert_vector_affine_dA_quadratic`
   when the field-strength definition is `F = dA` (exterior derivative, no Gamma
   dependence, standard Maxwell fluctuation with no connection terms), or
   `pert_vector_affine_covcurl_quadratic` when `F = nabla A` (covariant curl,
   T-dependent terms and a*dG cross-quadratic mixing). The choice follows the
   `field_strength_definition` convention resolved through elicitation; at
   `T=0` the two templates agree. Every scaffold uses Cadabra weights
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
   k-essence scaffold expands about a covariantly-constant-gradient background
   (`nabla nabla phibar = 0`, so `nabla Xbar = 0`), the standard setup for the
   sound speed; on it the coupling gradients close under the single chain rule
   `nabla K_X -> K_phiX nabla phibar` and its kin. When the connection is
   independent, the metric perturbation scaffold gains the connection fluctuation:
   `pert_metric_affine_quadratic` (eval 4ma) expands the Palatini
   Einstein-Hilbert action about a flat Minkowski background with `Gamma=0`,
   perturbing both `g_{mu nu} -> eta_{mu nu} + h_{mu nu}` and
   `Gamma^lambda_{mu nu} -> dG^lambda_{mu nu}`, so the quadratic action contains
   cross terms `h * dG` and `dG * dG` in addition to the graviton terms. The
   connection fluctuation `dG` (distinct from `h`) appears explicitly in the
   result, capturing the torsion and non-metricity modes. The linearized EOM
   check targets the linearized Palatini metric equation
   `R^{(1)}_{(alpha beta)}(dG) - 1/2 eta_{alpha beta} Rtilde^{(1)} = 0`, and
   the independent linearization route confirms it. The Ricci scalar is built
   as a fully contracted expression (not as `R_{sigma nu}` with free indices and
   then contracted) to avoid a Cadabra free-index clash where the derivative
   index in the second Palatini term conflicts with the contraction index.

   **Acceptance gating and the T=Q=0 LC limit.** The concrete acceptance
   case (Palatini EH around Minkowski) is verified (both `residue_zero` and
   `linearized_eom_match` True). The XOR condition holds: either verified with
   both checks True, or gated (`verified=False`) with a non-empty `detail`
   distinguishing the failure mode (no residue check / nonzero residue / residue
   zero but cross-check mismatch). The SymPy component cross-check on explicit
   random metric-affine backgrounds confirms the core physics claim (Ricci
   non-symmetric on asymmetric connections, linearized Palatini EOM has real
   metric-affine content, dG*dG part nonzero), and verified is gated behind it
   (the dual-gate invariant against the torsion trap). At `T=Q=0`, the
   metric-affine perturbation path reproduces the corresponding Levi-Civita
   result: the linearized Palatini metric equation equals the linearized
   Einstein tensor `G^{(1)}_{mu nu}` (matching the eval 3g operator), the Ricci
   of the LC connection is symmetric, and torsion is zero.

   The scaffolds cover dynamical scalar fields (plain and `X`-dependent), the
   metric, the metric-affine metric (with independent connection), and rank-1
   gauge potentials, so `derive_perturbation` refuses other field kinds (the
   rank-2 field strength, say) rather than guessing. On a metric-affine
   background the connection fluctuation `dG` is captured by the metric
   perturbation scaffold; the connection is not perturbed independently, so
   it is excluded from the default field list and a request to perturb it
   raises `NotImplementedError` naming the field (HTTP 422 with that message).
   The perturbation result persists: each run records its `result_id` into the
   session and writes a provenance bundle, reloading identically via
   `GET /results`, MCP `noether_results`, and web history with its `kind`,
   `verified` verdict, and `checks` intact. The metric-affine perturbation
   eval is registered as the CLI subcommand `noether eval4ma`, exercising the
   same path and checks. The `adm` task takes a different route: `derive_adm`
   (`kind="adm"` on the server, MCP, and web clients) writes no model script.
   Its deliverable is the ADM (3+1) decomposition of the gravitational sector,
   the Gauss-Codazzi split `sqrt(-g) R = N sqrt(h)(R3 + K_ij K^ij - K^2) -
   2 d_mu(sqrt(-g) v^mu)` and the normal/tangential projections of the Einstein
   tensor, which are universal foliation geometry independent of the action.
   The SymPy component kernel verifies all six identities (the split, both
   projections, the extrinsic-curvature identity K_ij = nabla_i n_j, and the
   lapse Euler-Lagrange equation) on a nondegenerate 1+2 background (eval 1s);
   `verified` is set from that suite. Any well-posed action carrying a metric is
   accepted; one with no metric is refused (HTTP 422 / MCP error naming the
   missing metric object). Each ADM derivation carries its active convention
   block (signature, torsion sign, non-metricity definition, Ricci-contraction,
   contortion sign, disformation sign, K-sign, foliation/normal convention;
   for metric-affine NPRs also the field-strength definition), so no convention
   is silently assumed; changing the elicited Ricci-contraction is reflected in
   the result. For a metric-affine NPR
   (independent connection), `derive_adm` additionally produces the connection's
   foliation decomposition (Gamma = LC + K(T) + L(Q) projected into normal and
   tangential parts), surfaces torsion and non-metricity pieces, separates
   constraint pieces from evolution, and identifies connection-sector primary
   and secondary constraints (gated when the Dirac chain cannot close); see
   section 6.5.

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

What still does not verify: the higher Horndeski densities, the `X`-dependent
quartic `G4(phi, X) R + G4_X[(box phi)^2 - (nabla_a nabla_b phi)^2]` and the
quintic G5. These are held out, not shipped partially. The reason is concrete.
Setting up the quartic scalar variation is straightforward (the term-by-term
expansion runs in Cadabra), but reducing it to a verified second-order equation
needs curvature machinery a generic `\nabla{#}::Derivative` does not supply: the
commutator `[nabla_a, nabla_b] nabla_c phi = -R^d{}_{cab} nabla_d phi`, the
metric-contracted Riemann-to-Ricci-to-scalar folds, the symmetry of the scalar
Hessian, and the contracted Bianchi identity. These are the identities through
which the apparent third derivatives are meant to cancel (the no-Ostrogradski
structure the `G4_X` counterterm exists to enforce).

The primitives now exist as a reusable layer, `noether/kernels/cadabra/curvature.py`,
each one pinned by a residue-checking test in `tests/test_curvature.py`: the
commutator (a difference form and a single-term form), the Ricci and scalar
folds, the Hessian symmetrization through a symmetric stand-in, the contracted
Bianchi identity (carried as a citable standard result, not re-derived), and the
targeted quartic box-commutator that takes the counterterm's own dangerous
combination `box^2 phi - nabla_a nabla_b nabla^b nabla^a phi` to a purely
second-order curvature coupling. That last one is the proof the approach works:
its test confirms the leading fourth-derivative term collapses to
`nabla R . nabla phi + R . nabla nabla phi` with no derivative of order three or
higher left.

The same module now also carries metric-affine (independent-connection) primitives
for curvature defined from an affine connection `Gamma^lambda_{mu nu}` with no
symmetry in the lower pair (torsion allowed).  These follow the eval2 Palatini
pattern: the connection is a `\partial`-Depends object, `R_{mu nu}` carries no
`::Symmetric` declaration, and the Ricci tensor is non-symmetric in general.
The primitives are `expand_riemann_affine` (the Riemann expansion in partial
derivatives of the connection), `expand_ricci_affine` (the traced form), and
`fold_ricci_affine` (the metric contraction, without symmetry).  They are pinned
by residue checks in `tests/test_curvature_affine.py` AND cross-checked against
the SymPy general-connection oracle (`riemann_of_connection`,
`ricci_of_connection`) on random backgrounds, per the dual-gate model
(architecture.md section 3.2): a green residue alone is insufficient because of
the torsion trap.  The existing Levi-Civita primitives are preserved as the
`T = Q = 0` special case; the `contracted_bianchi` primitive is marked
Levi-Civita-only and must not be reused under torsion.

The torsion primitive `T^lambda_{mu nu} = Gamma^lambda_{mu nu} -
Gamma^lambda_{nu mu}` is residue-pinned in the same module, with the
irreducible torsion decomposition into trace-vector, axial-vector, and
traceless-tensor parts.  The decomposition formulas are:

- Trace part: `(1/3)(delta^lambda_mu T_nu - delta^lambda_nu T_mu)` where
  `T_mu = T^rho_{rho mu}`.
- Axial part: `-(1/6) epsilon^lambda_{mu nu rho} A^rho` where
  `A^rho = (1/6) epsilon^{rho sigma kappa lambda} T_{sigma kappa lambda}`.
- Traceless tensor: `q = T - trace_part - axial_part`, satisfying
  `q^lambda_{lambda mu} = 0` and no totally antisymmetric component.

The three parts reassemble to T (residue 0 in the algebraic identities;
full componentwise verification by the SymPy oracle on random dim-4
backgrounds).  The decomposition is distinct from the contortion `K(T)` in
the post-Riemannian decomposition `Gamma = LC + K(T) + L(Q)`: the
contortion involves metric-based index raising while the irreducible trace
part uses only the Kronecker delta.  A Cadabra limitation prevents direct
tensor-level residue computation of `T - (t1 + t2 + q)` because
Kronecker-delta contractions produce free-index mismatches in sums; the
algebraic identities (trace extraction, traceless property) are verified in
Cadabra, and the full componentwise check is delegated to the SymPy
cross-check.  The torsion primitives are in `curvature.py`, the SymPy
helpers (`torsion_of_connection`, `torsion_trace_vector`,
`torsion_axial_vector`, `torsion_trace_part`, `torsion_axial_part`,
`torsion_traceless_tensor`) in `geometry.py`, and the pinned tests in
`tests/test_torsion_affine.py`.

The non-metricity primitive `Q_{lambda mu nu} = nabla_lambda g_{mu nu}` is
residue-pinned in the same module, together with the rewrite
`nabla_lambda g_{mu nu} -> Q_{lambda mu nu}` that replaces the baked-in
`nabla g -> 0` on the metric-affine path.  The rewrite for the inverse
metric is `nabla_lambda g^{mu nu} -> -g^{mu rho} g^{nu sigma} Q_{lambda rho sigma}`,
derived from the metric-compatibility condition for `g^{mu rho} g_{rho nu}`.
On the Levi-Civita path Q = 0 and these rewrites reduce to the old
zero substitutions.  The irreducible non-metricity decomposition splits Q
into three parts under the Lorentz group:

- Weyl-vector trace part (4 components in dim 4):
  `Q^(W)_{lambda mu nu} = (1/((n+2)(n-1)))
      [(n+1) omega_lambda g_{mu nu} - (omega_mu g_{lambda nu}
        + omega_nu g_{lambda mu})]`
  where `omega_lambda = Q_{lambda mu nu} g^{mu nu}`.  This carries the
  full Weyl trace and has zero second trace.
- Second-trace part (4 components):
  `Q^(2T)_{lambda mu nu} = (1/((n+2)(n-1)))
      [-2 qtilde_lambda g_{mu nu} + n(qtilde_mu g_{lambda nu}
        + qtilde_nu g_{lambda mu})]`
  where `qtilde_mu = Q_{lambda mu nu} g^{lambda nu}`.  This carries the
  full second trace and has zero Weyl trace.
- Traceless-tensor remainder (32 components):
  `Q^(TL)_{lambda mu nu} = Q_{lambda mu nu} - Q^(W) - Q^(2T)`.
  Doubly traceless.

As with the torsion decomposition, Kronecker-delta contractions produce
free-index mismatches in Cadabra sums, so the algebraic identities (trace
extraction, traceless properties) are verified in Cadabra and the full
componentwise check is done by the SymPy cross-check.  The decomposition
is distinct from the disformation L(Q) in the post-Riemannian split.
The non-metricity primitives are in `curvature.py`, the SymPy helpers
(`nonmetricity_of_connection`, `nonmetricity_weyl_trace`,
`nonmetricity_second_trace`, `nonmetricity_weyl_part`,
`nonmetricity_second_trace_part`, `nonmetricity_traceless_tensor`) in
`geometry.py`, and the pinned tests in `tests/test_nonmetricity_affine.py`.

The post-Riemannian decomposition `Gamma = LC(g) + K(T) + L(Q)` is the
representation backbone for metric-affine geometry.  Its primitives are in
`curvature.py` and `geometry.py`, pinned by residue checks in
`tests/test_post_riemannian.py` and cross-checked against the SymPy oracle.
The contortion and disformation signs are NOT asserted from memory; they
are derived and residue-pinned, then recorded as the named convention block
`metric-affine-v1`:

- Contortion: `K^lambda_{mu nu} = (1/2)(T^lambda_{mu nu}
  + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu}
  + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu})`.  Inversion:
  `K^lambda_{mu nu} - K^lambda_{nu mu} = T^lambda_{mu nu}` (residue 0).
- Disformation: `L^lambda_{mu nu} = (1/2) g^{lambda rho}(-Q_{mu nu rho}
  - Q_{nu rho mu} + Q_{rho mu nu})`.  Symmetric in lower pair.  Inversion:
  `Q_{lambda mu nu} = -(L^rho_{lambda mu} g_{rho nu}
  + L^rho_{lambda nu} g_{rho mu})` when T=0 (residue 0).
- Decomposition: `Gamma^lambda_{mu nu} = LC^lambda_{mu nu} + K^lambda_{mu nu}
  + L^lambda_{mu nu}` (SymPy componentwise on random backgrounds).

Cadabra verification strategy: the full expansion `Gamma - (LC + K + L)`
hits the Kronecker-delta limitation.  Instead, the decomposition is verified
through the inversion identities (K antisymmetry, L inversion) plus a
structural substitution-fires check, with the full componentwise equality
confirmed by the SymPy oracle.  SymPy helpers added:
`christoffel_of_metric`, `contortion_of_torsion`,
`disformation_of_nonmetricity` in `geometry.py`.  Cadabra primitives
added: `define_contortion`, `define_disformation`, `decompose_connection`,
`expand_lc` in `curvature.py`.

The torsionful commutator and non-symmetric scalar Hessian are also in
`curvature.py`, pinned by residue checks in `tests/test_commutator_affine.py`
and cross-checked against the SymPy oracle on random torsionful backgrounds.
These generalize the Levi-Civita commutator and symmetric Hessian to a
connection carrying torsion:

- `commute_third_derivative_affine`: the general commutator on a scalar
  field's covariant derivative,
  `[nabla_a, nabla_b] nabla_c phi = -R^d_{cab} nabla_d phi
  - T^d_{ab} nabla_d nabla_c phi`, carrying the torsion term that the
  Levi-Civita primitive omits.  When T=0 it reduces to the existing
  `commute_third_derivative`.
- `hessian_antisymmetry_affine`: the non-symmetric scalar Hessian under
  torsion, `nabla_mu nabla_nu phi - nabla_nu nabla_mu phi =
  -T^lambda_{mu nu} nabla_lambda phi`.  The antisymmetric part is nonzero
  on a torsionful background and zero at T=0, so the LC
  `hessian_to_symmetric` (which routes through a symmetric stand-in and
  silently drops the antisymmetric part) is invalid under torsion.

The dual gate is explicitly enforced and demonstrated in
`tests/test_dual_gate.py` (VAL-GEOM-014, VAL-GEOM-016).  The dual-gate
negative control uses the scalar Hessian: the LC `hessian_to_symmetric`
primitive gives `residue_zero=True` for the antisymmetric Hessian (because
`H_{mu nu}` is declared symmetric), but the SymPy oracle shows the actual
antisymmetric Hessian is nonzero under torsion.  The dual-gate verdict
(`verified = cadabra_residue_zero AND sympy_cross_check_agrees`) correctly
returns NOT verified.  The convention sign falsifier shows that flipping a
single sign in the `metric-affine-v1` convention block (contortion leading
sign 1/2 to -1/2, disformation leading sign 1/2 to -1/2, or Hessian torsion
sign - to +) changes the residue from 0 to nonzero or flips the SymPy
cross-check, proving no convention is silently baked in.

The SymPy cross-check uses a new `covariant_derivative_of_connection`
function in `geometry.py` that computes the covariant derivative of a
tensor using a general (possibly asymmetric) connection, plus
`riemann_down_of_connection` for the fully-lowered Riemann.  The key
correctness point: when `nabla_a` acts on the (0,2) tensor `nabla_b V_c`,
the connection term for the derivative index `b` produces the torsion
contribution `-T^d_{ab} nabla_d V_c` in the commutator.  This is the term
missing from the LC formula and the reason the LC primitive gives a wrong
answer under torsion (the torsion trap).

What is still open is the orchestration across the whole equation. Two lessons
came out of the attempts. First, a blind one-way commutator pass does not
converge: applied to every term it just trades the two contraction patterns'
identities back and forth. The reduction has to be targeted at the specific
contraction it means to fix, leaving an already-canonical term untouched.
Second, the variation does not hand you the canonical contraction. The leading
term comes out as `g^{a c} g^{b d} nabla_a nabla_b nabla_c nabla_d phi`, which
equals the box-commutator's input only after the inner Hessian is symmetrized,
and Cadabra's canonicalise does not symmetrize a nested second derivative. So the
missing piece is a normal-ordering pass: symmetrize inner Hessians, then drive
every third- and fourth-derivative contraction to a canonical order with the
matching curvature-emitting commutator, so the dangerous pieces cancel and only
curvature survives. This is exactly what xAct's xTras `SortCovDs` does, which
makes full G4 and G5 closure a natural first job for the planned xAct
cross-check kernel rather than a hand-rolled reimplementation in Cadabra.

A diagnostic caution worth recording: an inner-Hessian substitution that looks
like it proves second order (`nabla nabla phi -> H`, then `H -> 0`) silently
zeroes every higher-derivative term too, since they all contain an inner Hessian.
That gives a false "second order" pass. The honest check substitutes the bare
third derivative to zero and confirms the difference vanishes.

Then comes the harder metric equation (the full `delta R` with an `X`-dependent
coefficient), then G5. The gate is both equations of motion or neither: a quartic
term ships only when its scalar and metric equations both residue-check, so until
that normal-ordering pass is built and audited, `G4(phi, X) R` and G5 are
routed to the best-effort G4/G5 derivation path (not the generic model path),
which runs the hand-audited diagnostic scripts and returns `verified=False`
with a non-empty `detail` naming the SortCovDs blocker (VAL-EOM-013), rather
than added as a partial result. The detection function `has_g4g5_terms` in
`blocks.py` identifies Lagrangians containing these terms so the derive path
can intercept them before the model-written fallback. By contrast the
`perturb` path does expand `X`: the k-essence scaffold (eval 3k) carries the
quadratic action and sound speed of a general `K(phi, X)`, on a
covariantly-constant-gradient background. The ADM split verifies for any metric
action, but it is the universal foliation geometry, not a Horndeski-specific
Hamiltonian.

The best-effort G4(phi,X)R / G5 closure attempt now exists as a dedicated module,
`noether/kernels/cadabra/horndeski_g4g5.py`, exercised by
`tests/test_horndeski_g4g5.py` (VAL-GEOM-015, VAL-EOM-013). The module
constructs Cadabra scripts for both the scalar and metric EOM variations, applies
the available M2 primitives, and checks whether the result is second order. The
scalar EOM is second order (no third derivatives of phi survive the IBP); the
metric EOM carries third derivatives in wrapped `nabla_mu(G4_X nabla_nu
nabla_rho phi nabla^rho phi)` terms that, upon expansion via `product_rule`,
produce `G4_X nabla_mu nabla_nu nabla_rho phi nabla^rho phi`. Without the
normal-ordering pass (SortCovDs), these cannot be systematically driven through
the commutator, Ricci folds, and Bianchi, so the closure is gated
(`verified=False` with a non-empty `detail` naming the blocker). The result
satisfies the XOR condition: it is either fully verified (residue 0 and SymPy
agrees) or gated with a named blocker; never verified with a gate unmet.

The M3 integration wires this best-effort attempt into the general EOM
derivation path. When `derive_field` detects G4(phi,X)R terms in the Lagrangian
(via `has_g4g5_terms` in `blocks.py`), it routes to `attempt_g4g5_eom` in
`derive.py`, which runs the hand-audited scripts and produces
`FieldDerivation` objects for the scalar and metric EOMs. Each derivation
carries the diagnostic checks and the gated verdict, satisfying VAL-EOM-013:
if `verified==True` then `residue_zero=="True"`, else `verified==False` with
a non-empty `detail`. The G4/G5 derivations carry their convention block
(via `_convention_block(npr)`) just like every other derivation path (EOM,
perturbation, ADM), so even the gated result identifies the conventions that
produced the (unverified) expression.

### 6.2 Vector (Maxwell) EOM on a metric-affine background

The gauge kinetic term `S = -1/4 int sqrt(-g) F_{mu nu} F^{mu nu}` on an
independent-connection background highlights the field-strength choice
consequence (VAL-EOM-020, VAL-EOM-021). Two definitions of `F` are elicited:

- **Exterior derivative** `F = dA` (the default, `F_{mu nu} = 2 partial_{[mu}
  A_{nu]}`): the action has no Gamma dependence (F involves only partial
  derivatives of A), so the EOM is the LC divergence
  `nabla^{LC}_mu F^{mu nu} = 0`.  When this is expressed with the
  full-connection divergence `nabla^{aff}_mu`, T/Q correction terms appear:
  `nabla^{aff}_mu F^{mu nu} = (K^rho_{rho mu} + L^rho_{rho mu}) F^{mu nu}
  + (1/2) T^nu_{mu rho} F^{mu rho}`.  The hypermomentum is zero.

- **Covariant curl** `F = nabla A` (`F_{mu nu} = 2 nabla_{[mu} A_{nu]}`):
  the action depends on Gamma through the covariant derivative, so the
  EOM naturally involves the full-connection divergence with a torsion term:
  `(1/sqrt(-g)) partial_mu(sqrt(-g) F^{mu nu}) + (1/2) T^nu_{mu rho}
  F^{mu rho} = 0`.  The hypermomentum is nonzero, purely spin-type:
  `Delta^lambda_{mu nu} = -2 A_lambda F^{mu nu}` (antisymmetric in mu, nu).
  The connection equation of motion has this source term, coupling the gauge
  and connection sectors.

The two derivations differ exactly in the connection-equation source: zero vs
`-2AF`.  Both are verified:

1. **dA EOM**: Cadabra residue zero (nabla + LC-substitution approach, valid
   because F=dA has no connection dependence).  The T/Q correction identity
   (affine divergence = LC divergence + correction) is verified by SymPy on
   random affine backgrounds.
2. **dA hypermomentum = 0**: Cadabra check (no dG terms in the connection
   variation).  SymPy structural verification (dA has no Gamma dependence).
3. **Covcurl hypermomentum != 0**: Cadabra check (dG terms present).  SymPy
   exact symbolic derivative verification (connection variation gives
   `sqrt(-g) A_sigma F^{alpha beta}`, confirmed by perturbing one Gamma
   component with symbolic epsilon and extracting the linear coefficient).
4. **Covcurl EOM form**: Euler-Lagrange decomposition verified by SymPy:
   `partial L/partial A_sigma = 1/2 sqrt(-g) T^sigma_{mu nu} F^{mu nu}`
   (explicit A dependence from the Gamma*A terms in F) and
   `partial L/partial(partial_mu A_sigma) = -sqrt(-g) F^{mu sigma}`
   (partial-derivative dependence).  The Cadabra residue check for the
   covcurl EOM is gated: the expansion produces mixed-index G terms that
   `canonicalise` cannot resolve (known limitation per cadabra-gotchas.md).

Tests: `tests/test_vector_eom_affine.py` (31 tests: 3 Cadabra residue, 28
SymPy cross-checks on 3 random affine backgrounds).  Eval module:
`evals/eval_vector_affine.py`, `evals/test_eval_vector_affine.py`, registered
as `vector-affine` in `evals/registry.py`.

### 6.3 Einstein-Cartan connection equation (VAL-EOM-011)

The Einstein-Cartan connection equation is the Palatini connection EOM on a
metric-compatible (Q=0) torsionful background.  Its key property is that it
is algebraic in the contortion K: after substituting Gamma = LC + K, no
derivatives of K appear.  This means torsion is algebraically determined by
any spin source (hypermomentum) rather than propagating as an independent
degree of freedom.

Two verification gates enforce this claim (dual-gate requirement, section 3.2):

1. **Cadabra residue check**: the registered template
   `ec_connection_algebraic_in_K` in `templates.py` derives the Palatini
   connection equation, substitutes G = LC + K, and verifies that setting
   `partial_K -> 0` does not change the expression (`algebraic_in_K` check).
   It also verifies that G = LC + projective mode satisfies the equation
   (`solution_zero` check).  The template replaces the previously inline
   Cadabra script string in the test file.

2. **SymPy cross-check**: `palatini_connection_eom` in `geometry.py` computes
   the Palatini connection EOM coefficient for a general connection on an
   explicit coordinate-dependent background.  The Euler-Lagrange-derived
   formula is algebraic in Gamma (no derivative-of-Gamma terms after IBP),
   so the EOM difference E(Gamma=LC+K) - E(Gamma=LC) is purely algebraic in
   K.  The function `einstein_cartan_algebraic_in_K_residual` verifies this
   componentwise: it computes E(LC+K) - E(LC) and checks the difference
   equals the expected algebraic K expression (no derivative-of-K terms) on
   metric-compatible (Q=0) torsionful backgrounds.  A negative-control test
   confirms that K has nonzero coordinate derivatives on these backgrounds
   (the contortion involves g^{-1} and g, which are coordinate-dependent),
   making the absence of derivative-of-K terms a non-trivial cancellation.

   Additionally, `palatini_connection_eom` evaluates to zero at Gamma = LC
   (the Palatini solution up to projective mode), providing a SymPy analogue
   of the Cadabra `solution_zero` check.

The dead oracle function `einstein_cartan_connection_equation_residual` that
previously existed in `geometry.py` computed the EOM at Levi-Civita only,
with a wrong index structure in the delta term, and was never imported or
called.  It has been replaced by the correct `palatini_connection_eom` and
`einstein_cartan_algebraic_in_K_residual` functions, both wired into passing
tests.

Tests: `tests/test_einstein_cartan.py` (17 tests: 2 Cadabra residue checks
via the registered template, 15 SymPy cross-checks on 3 random
metric-compatible torsionful backgrounds).  The Cadabra template is
registered as `ec_connection_algebraic_in_K` in `templates.py`.

### 6.4 Vector perturbation on a metric-affine background (VAL-PERT-017/018)

The quadratic perturbation of a gauge field on a metric-affine background
exposes the field-strength choice consequence at quadratic order
(VAL-PERT-017) and the connection-matter cross-quadratic mixing
(VAL-PERT-018).  Two templates are registered:

1. **`pert_vector_affine_dA_quadratic`** (F = dA, exterior derivative):
   The quadratic action is the standard Maxwell fluctuation
   `S2 = -1/4 sqrt(-g) f_{mu nu} f^{mu nu}` where
   `f_{mu nu} = nabla_mu a_nu - nabla_nu a_mu`.  Because F = dA has
   no Gamma dependence, no connection fluctuation dG appears: the action
   is purely in the gauge sector.  Both Cadabra residue checks pass:
   `residue_zero=True` (the variation of S2 equals the linearized Maxwell
   operator) and `linearized_eom_match=True` (the same operator follows
   from independently linearizing the full nonlinear EOM).  Convention
   recorded: `field_strength_definition=exterior-derivative`.

2. **`pert_vector_affine_covcurl_quadratic`** (F = nabla A, covariant curl):
   The quadratic action on a Minkowski background with constant Abar is
   `S2 = -1/4 (f - T Abar)^2` where `T^lambda_{mu nu}` is the torsion
   of the connection fluctuation.  Expanding:
   - Part 1: `f_dA^2` (same as dA case, no T dependence)
   - Part 2: `f_dA * T * Abar` (a*dG cross term, VAL-PERT-018 mixing)
   - Part 3: `(T Abar)^2` (dG*dG term)

   The dA and covcurl quadratic actions differ by T-dependent terms
   (Parts 2 and 3), confirming VAL-PERT-017.  The a*dG cross term
   (Part 2) confirms VAL-PERT-018: the connection and matter fluctuations
   mix at quadratic order, not block-diagonalized away.

   The Cadabra residue check is gated: the dG*a cross terms produce
   mixed-index objects after `canonicalise` that Cadabra cannot resolve
   (the same Kronecker-delta limitation that blocks the covcurl EOM
   residue check; see cadabra-gotchas.md).  The template uses the
   torsion symbol `T^lambda_{mu nu}` (avoiding the Kronecker-delta
   limitation in the NOETHER_RESULT construction) and prints three parts
   separately.  The SymPy cross-check provides the independent
   verification, computing the dA and covcurl quadratic actions on
   explicit random affine backgrounds and confirming:
   - The two actions differ on torsionful backgrounds (VAL-PERT-017)
   - The covcurl action contains a*dG cross terms (VAL-PERT-018)
   - The two actions agree at T=0 (T=0 limit)
   - The covcurl linearized EOM contains a torsion source term absent in
     the dA case

   Convention recorded: `field_strength_definition=covariant-curl`.
   Detail: `"covariant-curl quadratic-action residue gated: dG*a cross
   terms produce mixed-index objects after canonicalise (Kronecker-delta
   limitation); SymPy cross-check provides independent verification"`.

The adapter now parses three additional sentinel markers:
`NOETHER_DETAIL:` (gating explanation) and `NOETHER_CONVENTION:`
(convention key=value pairs), in addition to the existing
`NOETHER_RESULT:` and `NOETHER_CHECK:`.  These are returned in the
`ComputedResult.value` dict as `detail` (string) and `conventions`
(dict[str, str]).

Tests: `tests/test_pert_vector_affine.py` (22 tests: 9 Cadabra
structural checks, 11 SymPy cross-checks on 3 random affine backgrounds,
2 XOR gate checks).  Routing: `perturb-vector-affine-dA` and
`perturb-vector-affine-covcurl` in `generate.py`.

### 6.5 Metric-affine ADM (3+1) decomposition (VAL-ADM-001 through VAL-ADM-009)

The existing `derive_adm` path (section 6, item 8) produces the GR ADM split
(lapse, shift, induced metric, extrinsic curvature K_{ij}, Hamiltonian and
momentum constraints) verified by the SymPy component kernel on a
nondegenerate 1+2 background. For a metric-affine NPR
(`geometry.connection.type == "independent"`), `derive_adm` now additionally
produces connection-sector decomposition pieces.

**Connection foliation decomposition.** The post-Riemannian decomposition
`Gamma^lambda_{mu nu} = LC^lambda_{mu nu}(g) + K^lambda_{mu nu}(T) +
L^lambda_{mu nu}(Q)` (pinned in M2) is projected along the foliation into
normal and tangential parts using the normal `n_mu = (-N, 0, ..., 0)` and the
tangential projector `h_{mu nu} = g_{mu nu} + n_mu n_nu`. The result exposes
the spatial, normal-upper, and mixed components of the contortion K(T) and the
disformation L(Q). Torsion and non-metricity are decomposed similarly:
spatial torsion `T^i_{jk}`, normal-upper torsion `T^n_{jk}`, mixed torsion
`T^i_{nk}`; spatial non-metricity `Q_{ijk}`, normal-first non-metricity
`Q_{nij}`, mixed non-metricity `Q_{inj}`. For a Levi-Civita connection
(T=Q=0), these pieces are absent (the decomposition reduces to the GR ADM
with no connection-sector output).

**Constraint/evolution separation.** The metric-sector constraints
(Hamiltonian `R^{(3)} + K^2 - K_{ab}K^{ab}`, momentum
`D_a(K^a_b - h^a_b K)`) are already labeled as constraints in the GR ADM
output (they are first-order in time derivatives and constrain initial data).
The connection-sector constraints are surfaced distinctly: the connection EOM
(variation of the action with respect to Gamma) is algebraic (no time
derivatives of Gamma), making all connection components non-dynamical and
generating primary constraints. For metric-compatible (Q=0) torsionful
theories, the Dirac constraint chain can be closed: the algebraic EOM
constrains the contortion K, and the projective gauge freedom (for pure
Palatini EH) generates first-class constraints. For non-metric-compatible
(Q!=0) theories, the Dirac chain cannot be closed in general and is gated
with a stated reason: the disformation L(Q) introduces additional structure
that requires action-specific analysis.

**Verification.** The `AffineADMGeometry` class in
`noether/kernels/sympy_kernel/adm.py` runs seven checks on a nondegenerate
1+2 background with a general affine connection carrying both torsion and
non-metricity: (F) the post-Riemannian decomposition `Gamma = LC + K(T) +
L(Q)` holds componentwise on the foliated background; (G) the torsion and
non-metricity foliation pieces are correctly extracted from the full tensors;
(H) the connection EOM is algebraic on a metric-compatible background; (I)
the connection-sector primary constraints are identified; plus
distortion-spatial-projections consistency, background nondegeneracy with
all distortion features switched on, and a distortion-nonzero falsifier
that explicitly asserts every distortion feature (T, Q, K, L in their
spatial, normal-upper, and mixed components) is nonzero on the background.
The falsifier is required by VAL-ADM-007: a verified ADM split is verified
because the SymPy kernel reduced the split and constraint projections to
zero on an explicit nondegenerate metric-affine background whose distortion
features are asserted nonzero, so a wrong tensor relation cannot survive
the check. The metric-sector GR checks (A-E) are
inherited from `ADMGeometry`. The adapter exposes the suite as the
`adm-affine-1p2` component-eval check. The derive path runs both
`adm-gr-1p2` (metric sector) and `adm-affine-1p2` (connection sector) when
the connection is independent, producing eight `FieldDerivation` objects:
three GR pieces plus five connection-sector pieces. The connection-sector
constraints piece carries `verified=False` with a detail when the Dirac chain
cannot close (Q!=0), and `verified=True` with a positive detail when the chain
closes (Q=0). The piece is never dropped: even a gated piece surfaces with
its `result_tex` present (VAL-ADM-008).
**Matter hypermomentum in the ADM constraint structure (VAL-ADM-015).**
When the action has matter that couples to the independent connection (has
nonzero hypermomentum `Delta^lambda_{mu nu} = -(2/sqrt(-g)) delta S_matter /
delta Gamma^lambda_{mu nu}`), the ADM result includes a "matter hypermomentum
contribution" piece naming the spin/dilation/shear decomposition entering
the connection-sector constraints. The spin part `tau^lambda_{mu nu}`
(antisymmetric in the first pair when lowered, traceless) sources the
torsion primary constraint; the dilation trace vector `Delta_nu` sources
the projective constraint; the shear part `sigma^lambda_{mu nu}` (symmetric
in the first pair when lowered, traceless) sources the non-metricity
constraint. The decomposition is verified by the SymPy kernel on the
foliated background (full (D+1)-dimensional reconstruction, spatial-sector
reconstruction, spin/shear symmetry/trace properties). The piece is
verified (Dirac chain closes) on a metric-compatible (Q=0) background and
gated (with a detail) when Q!=0. Pure-gravity sessions (no matter coupling
to the connection) carry no such matter piece. The detection uses the
NPR conventions (`field_strength_definition="covariant-curl"` for gauge
fields, scalar-field presence) and the `_action_has_hypermomentum` helper.
The adapter exposes the matter check as the `adm-affine-matter-1p2`
component-eval check.
**Verification model (VAL-ADM-006 through VAL-ADM-009).** The ADM result's
`kernel_name` is `"sympy"`: the SymPy component kernel verifies the split and
projections, and no LLM Cadabra script is written for the adm path (the
`derive_adm` function takes no `LLMAdapter`). The `verified` flag is set
solely by the kernel's component-eval checks, not by the model or by
explanatory prose. Each named check must `passed=True` on a background whose
distortion features are asserted nonzero by the falsifier. Any part that
cannot be reduced is returned `verified=False` with a `detail` naming the
blocker; the piece is still surfaced (`result_tex` present), never dropped
or reported true. An action with no metric is refused: `derive_adm` raises
`NotImplementedError` naming the missing metric object (e.g. "this action
declares no metric object 'g'"), which the server surface translates to HTTP
422 and the MCP surface returns as an error dict; no ADM derivation is
produced. Every derivation (EOM, perturbation, and ADM alike) carries its
full convention block on the `conventions` field of `FieldDerivation`:
signature, torsion sign, non-metricity definition, Ricci-contraction,
contortion sign, disformation sign, K-sign, foliation/normal convention, and
the convention id; for metric-affine NPRs the field-strength definition is
also included. A non-default convention chosen at elicitation (for example
`ricci_contraction=first-fourth`) appears identically in the convention blocks
of EOM, perturbation, and ADM results within the same session (VAL-CROSS-003).
Changing the elicited Ricci-contraction is reflected in all results. Teaching
narration about the geometry and its tradeoffs (torsion -> spin coupling,
non-metricity -> length non-conservation, projective freedom) is on the
`teaching` field of `FieldDerivation` (distinct from `detail` which carries
the verdict explanation, and from `result_tex` which carries the kernel output).
`detail` is always non-empty: it names the blocker when `verified=False` and
states the confirmation reason when `verified=True`, so a gated result is
distinguishable from a verified one by both fields across all surfaces.
The teaching field is pure prose that never sets a result expression, never
appears in `checks`, and never flips `verified`. Generating teaching mutates no
NPR and adds no NPR version. /derive and /results expose teaching as a
top-level per-derivation key. During elicitation (chat / HTTP elicit), the
proposal's rationale is surfaced to the user alongside the on-menu choice.
MCP `noether_derive`/`noether_results` return derivations whose `verified` and
`detail` match the HTTP surface; blocked/refused calls return error/blocked
dicts, never a fabricated verified result.

Tests: `tests/test_adm_affine.py` (34 tests: reachability, metric-sector
split, connection foliation decomposition, constraint/evolution separation,
connection-sector constraints, GR regression, adapter integration).
Eval: `evals/test_eval_adm_affine.py` (10 tests: elicitation gate, metric
and connection sector checks, adapter check). CLI: `noether adm-affine`.

### 6.6 Metric teleparallel f(T) gravity (VAL-EOM-023)

The metric teleparallel formulation derives the gravitational field equation
from the torsion scalar T rather than the Ricci scalar R. The connection is
the Weitzenbock connection, built from the tetrad (vierbein) e^a_mu:

  Gamma^rho_{mu nu} = E_a^rho partial_mu e^a_nu

where E_a^mu is the inverse tetrad. This connection is flat (R=0), metric
compatible (Q=0), and carries torsion (T!=0). The geometry is
`ConnectionSpec(family="teleparallel", curvature_free=True,
metric_compatible=True, torsion=True, nonmetricity=False)`. The
tetrad e^a_mu is a first-class NPR field (`ObjectDecl(kind="tetrad")`).

**Torsion scalar.** The Weitzenbock torsion scalar is:

  T = (1/4) T_{rho mu nu} T^{rho mu nu}
    + (1/2) T_{rho mu nu} T^{mu rho nu}
    - T_mu T^{mu}

where T_mu = T^rho_{rho mu} is the torsion trace vector. The modified
superpotential S^{rho mu nu} = (1/2)(K^{rho mu nu} + g^{rho mu} T^nu -
g^{rho nu} T^mu) contracts with f'(T) in the general EOM.

**Boundary-term identity.** The torsion scalar and the Levi-Civita Ricci
scalar are related by:

  T = -R(g) + 2 nabla_mu^{LC} T^mu

where nabla^{LC} is the Levi-Civita covariant derivative. The divergence
2 nabla_mu T^mu is a total boundary term. For the linear case f(T) = T,
this means the teleparallel action equals minus the Einstein-Hilbert action
plus a boundary term, so the metric EOM is G_{mu nu} = 0 (the same as GR,
up to sign conventions). The identity is verified componentwise by the SymPy
oracle (`ft_boundary_residual` in `ft_tetrad.py`) on explicit Weitzenbock
backgrounds.

**Cadabra verification.** The template `eom_ft_linear_tetrad` varies the
f(T) = T action via the boundary-term identity T = -R + 2 nabla_mu T^mu,
reducing the variation to the Einstein-Hilbert path. It passes the residue
check (residue_zero == True). The verified path detail records the
boundary-term identity and the tetrad/Weitzenbock formulation.

**SymPy cross-check.** The module `noether/kernels/sympy_kernel/ft_tetrad.py`
constructs Weitzenbock connections from tetrads on explicit coordinate-dependent
backgrounds and verifies:
- The connection is flat (R=0) by construction.
- The connection is metric compatible (Q=0).
- The torsion is nonzero for non-diagonal tetrads.
- The boundary-term identity holds (residual is zero).
- The linear f(T) = T EOM via boundary identity gives G_{mu nu} = 0.
- The general f(T) metric-form EOM formula is consistent on T!=0 backgrounds.
- The superpotential S^{rho mu nu} is computable.

The diagonal tetrad of a diagonal metric gives T=0; to get T!=0, the tests
use a rotated (non-diagonal) tetrad with an off-diagonal timelike entry
that mixes the -1 and +1 blocks of the Minkowski metric eta_{ab}. The metric
for the rotated tetrad is computed from the tetrad itself (g_{mu nu} = e^a_mu
e^b_nu eta_{ab}), rather than from the original diagonal metric.

**General f(T) field equation (metric form):**

  f'(T) [G_{mu nu} - (1/2) g_{mu nu} T]
  + S_{mu nu}^{rho} nabla_rho f'(T)
  + (1/2) g_{mu nu} [f(T) - T f'(T)] = 0

For f(T) = T: f'=1, nabla f'=0, f-Tf'=0, giving E_{mu nu} = G_{mu nu} -
(1/2) g_{mu nu} T. Via the boundary-term identity this is equivalent to
G_{mu nu} = 0.

**Convention block: tetrad-teleparallel-v1.**

  Torsion: T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
  Non-metricity: Q_{lambda mu nu} = 0 (metric compatible)
  Curvature: R^rho_{sigma mu nu}(Gamma) = 0 (curvature free)
  Minkowski: eta_{ab} = diag(-1, +1, +1, +1)
  Metric: g_{mu nu} = e^a_mu e^b_nu eta_{ab}

**NPR construction.** The eval module `evals/eval_ft_teleparallel.py` builds
an NPR with `ConnectionSpec(type="independent", family="teleparallel",
curvature_free=True, metric_compatible=True, torsion=True,
nonmetricity=False)`, a metric object, a tetrad object
(`ObjectDecl(kind="tetrad", name="e")`), a connection object, a torsion
scalar shorthand T, and the coupling function f(T). The elicitation path
asks the connection-type, torsion, non-metricity, metric-compatibility,
and curvature-free questions, with the action's torsion cue (the presence
of T) grounding the inference proposals.

Tests: `tests/test_teleparallel_fq_ft.py` (f(T) and f(Q) tests together).
Eval: `evals/test_eval_ft_teleparallel.py`. Cadabra template:
`eom_ft_linear_tetrad` in `templates.py`.

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
