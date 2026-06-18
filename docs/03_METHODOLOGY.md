# 03 — Methodology

**Status:** draft.
**Scope:** how Noether behaves (elicitation, good form, verification) and how we
build it (eval-driven development). This is the "how" that NORTH_STAR.md
deliberately deferred.

---

## 1. Elicitation protocol

The goal: turn an under-determined LaTeX action into a well-posed NPR with the
fewest, sharpest questions. The user should feel interrogated by a colleague, not
processed by a form.

### 1.1 Infer first, ask second

Before asking anything, INGEST runs an inference pass over the parsed action and
classifies every open item into one of three bins:

- **Inferable:** notation settles it. `\sqrt{-g}` implies a metric and Lorentzian
  signature; `F_{\mu\nu}` defined as `\partial_\mu A_\nu - \partial_\nu A_\mu` is
  antisymmetric by construction; an index that appears once up and once down is
  contracted. Inferred facts are stated back to the user for confirmation, not
  asked as questions.
- **Conventional:** a community default exists but reasonable people differ
  (curvature sign, signature, symmetrization weight, what `X` abbreviates).
  These are asked as confirm-or-correct items with the default shown:
  "I will take X = -½∇φ∇φ unless you say otherwise."
- **Undecidable:** only the physicist knows (which symbols are dynamical, whether
  the connection is a dynamical field, what to vary, which background and gauge).
  These are always asked, never defaulted.

For metric-affine geometry, the action's structure can force a narrower set of
questions. A term with curvature or an explicit connection opens the geometry
questionnaire. Connection type, torsion, non-metricity, and metric compatibility
are asked as menu-bound inferable questions, because the notation constrains the
space of sensible answers even though the physicist still has to confirm one.
On the CLI chat and resume loop, those geometry answers stay on the same
confirmation rail as the HTTP resolve surface: numbered picks and typed on-menu
answers are applied through the shared menu validator before they land in the
session store.
If the physicist confirms an independent connection, elicitation opens
conventional follow-up questions: which Ricci contraction to use now that
`R_{μν}` need not be symmetric, and (if the action declares a vector/gauge
potential) how the field strength should be defined (exterior derivative `dA`
vs connection-covariant curl `nabla A`, which differ by torsion per
VAL-GEOM-020). A curvature-free scalar action does not open that questionnaire
and stays on the default Levi-Civita draft.

The inference contract for geometry ambiguities mirrors the general
propose-then-confirm flow: the model proposes one option per open geometry
ambiguity, constrained to the menu; off-menu suggestions are nulled (choice is
`None`) while rationale may survive; and after `propose_resolutions` the NPR is
unchanged (not well-posed, geometry ambiguities unresolved, `geometry.connection`
unchanged). Only a human-confirmed on-menu answer, routed through
`apply_resolutions`, mutates `geometry.connection`. Off-menu and unknown-id
confirmations raise `ValueError` and never mutate the NPR.

The inference prompt (`build_elicitation_prompt`) embeds the action's geometric
cues (presence of `R(\Gamma)`, explicit `T`/`Q`, `f(Q)`/`f(T)` family) so the
model's proposed geometry choices are grounded in the action, not a fixed
default; a scalar action carries no such geometry cue (VAL-GUIDE-017).

Convention proposals (Ricci-contraction when the connection is independent,
field-strength definition when a gauge field is present on an
independent-connection background) follow the same propose-then-confirm
contract: the model proposes an on-menu choice with rationale, never
auto-applies it, and an off-menu convention proposal is nulled
(VAL-GUIDE-020). NPR conventions are unchanged until a human confirms
through `apply_resolutions`.

The HTTP surface enforces this identically: `POST /elicit` returns
`confirmed: false` with proposals (off-menu nulled), and `POST /resolve`
with an off-menu geometry answer returns 400. Inference is exercised
deterministically with `StubLLMAdapter` (VAL-GUIDE-001..007,
VAL-GUIDE-017, VAL-GUIDE-020, `tests/test_geometry_inference.py`).

### 1.2 Question discipline

- Batch related questions; one elicitation round should usually suffice for EOM
  tasks (the guiding scenario in NORTH_STAR §7 shows the target feel).
- Each question states why it matters when non-obvious ("If Γ is independent,
  you get a second field equation").
- Read-back: after elicitation, Noether restates the complete problem (fields,
  roles, symmetries, conventions, ask) in clean LaTeX and waits for confirmation.
  This read-back is stored as the assumption snapshot in the provenance bundle.
- Mid-session changes are first-class: changing an answer creates a new NPR
  version and explicitly marks dependent results stale, and this stale state
  is consistent across all surfaces (HTTP `GET /results`, MCP
  `noether_results`, and the frontend).

### 1.3 The ambiguity ledger

Mechanically, every open item is an entry in `npr.ambiguities`. PLAN refuses to
run while the ledger is non-empty (a structural guarantee, see tech spec §4).
This is how "ambiguity is resolved by asking, not by guessing" becomes code
rather than policy.

## 2. Computation planning

- Each user ask maps to a task type (`vary`, `reduce`, `adm`, `perturb`,
  `identity-check`, ...) with a standard plan template (DAG of capability-tagged
  steps; tech spec §6.2).
- Kernel selection is by capability and cross-check policy, not preference: the
  primary kernel computes; when the verification ladder demands it, a second
  kernel independently recomputes or spot-checks.
- Plans are visible. The user can ask "what are you about to do" and get the DAG
  in plain language before or after execution.

## 3. Good form, operationalized

NORTH_STAR §10 names four properties. Operationally:

- **Canonical:** kernel canonicalization (Butler-Portugal class) plus Noether's
  finishing pass: stable dummy-index names, stable term order, sign convention
  normalization. Determinism law: same NPR in, byte-identical LaTeX out.
- **Minimal:** an identity-reduction step runs until fixpoint: Bianchi and Ricci
  identities, derivative commutation to canonical order, declared symmetries,
  dimension-dependent identities for the session's dimension. "No term a known
  identity can remove" is the testable definition of minimal.
- **Transparent:** the result is projected onto a target basis named in
  `task.target_form`. Defaults per result class (EOM for metric theories default
  to curvature tensors + matter structures, collected by tensor structure), and
  the user can renegotiate ("write it with the Einstein tensor isolated",
  "collect in derivatives of φ"). Renegotiation re-runs only the finishing
  pipeline, not the derivation.
- **Faithful:** the chain from input action to final form is a recorded list of
  transformations, each either a kernel operation or a named identity. The
  equality `final ≡ initial` under stated assumptions is itself checked (V2/V3
  below) rather than assumed.

### 3.1 Readability shorthands (proposed, never imposed)

The derivatives of function couplings dominate the algebra: a coupling `F(φ)`
contributes `∂F/∂φ` and `∂²F/∂φ²` throughout the variation and the equations of
motion, and physicists routinely shorthand these as `F_φ`, `F_φφ`. Metric-affine
work adds the same pressure around the post-Riemannian decomposition, so the
definitions surface also offers `K(T)`, `L(Q)`, and the `f(Q)` scalar `Q` once
the NPR carries an independent connection. Noether proposes those shorthands
(`noether.orchestrator.definitions.propose_definitions`, exposed at `GET
/sessions/{id}/definitions`, the MCP tool `noether_propose_definitions`, and
the web client's "Suggested notation" card).

Two boundaries keep this honest:

- These are **definitions, not results.** `F_φ` is introduced as a name for the
  derivative `∂F/∂φ`, and `K(T)` or `L(Q)` are introduced as names for standard
  metric-affine objects; nothing here claims what any particular variation
  evaluates to, so AGENTS.md rule 1 is not engaged (there is no asserted result,
  only notation). Proposals are deterministic functions of the declared function
  couplings, their arguments, and the confirmed geometry family.
- Adoption is the **human's choice** (rule 4). Proposing never mutates the NPR;
  only an accepted proposal is applied, as a new immutable NPR version carrying a
  `shorthand` `ObjectDecl` whose `definition_tex` records the meaning. Couplings
  the human has pinned to a constant are not offered (their derivatives vanish),
  and already-declared symbols are never re-proposed, so the proposal set
  converges as notation is adopted. Adopting notation neither opens nor closes
  the ambiguity gate.

## 4. The verification ladder

Every result climbs as far up the ladder as its class allows. Verdicts ship in
`checks.json` and in the user-facing narrative. Teaching prose is carried on
the `teaching` field of `FieldDerivation`, separate from both `detail`
(diagnostic) and `checks` (kernel-verified). Failure at any rung blocks
PRESENT from labeling the result verified; Noether says plainly which checks ran,
which passed, and which were not applicable.

- **V0 — well-formedness.** Index balance, symmetry consistency, dimensional
  homogeneity of every term, round-trip parse of the output LaTeX. In the
  metric-affine path this stays structural: Noether does not treat
  raising/lowering across `\nabla` as free unless metric compatibility has been
  confirmed explicitly. Cheap, always on.
- **V1 — structural invariants.** Class-specific necessary conditions: a metric
  EOM must be symmetric; a U(1) gauge field EOM must be gauge covariant; an
  antisymmetrized symmetric pair must vanish. Computed, not asserted.
- **V2 — identity checks.** Noether identities from declared symmetries: the
  covariant divergence of a diffeomorphism-invariant metric EOM must vanish
  on-shell (and identically in vacuum); traces and contractions must match known
  constraints. These are full kernel computations on the result.
- **V3 — limiting cases.** Parameter and coupling limits that must reproduce
  known theories (set G(φ,X)=0 and recover the textbook result; set F = const and
  recover minimally coupled scalar; D=4 must annihilate the Gauss-Bonnet EOM).
  Targets come from the regression corpus.
- **V4 — independent recomputation.** A second kernel (or the component
  spot-check: evaluate both sides on pseudo-random explicit backgrounds via
  SymPy/SageManifolds) reproduces the canonical form. The SymPy component
  spot-check variant is in use today (it is what verifies the ADM split and
  anchors the perturbation linearizations); the second symbolic kernel (xAct)
  that makes full cross-kernel recomputation mandatory for novel-theory results
  is Horizon 3.

The ladder is the product's trust story. "I checked it three ways and it holds"
(NORTH_STAR §16) means, concretely: V2 + V3 + V4 green.

### 4.1 The general derivation path and its residue check

For the eval actions, the derivation runs a frozen, golden-tested template. For
any other well-posed action there is no pre-written script, so the model writes
one (`noether.kernels.cadabra.generate`, `noether.orchestrator.derive`). The
trust story does not loosen: the generated script must derive the equation of
motion by `vary()` and then state an independent candidate equation, and the
kernel canonicalizes the difference and prints whether the residue is zero. That
residue check is a V3-style equality verified by computation, so `derive_field`
labels a result verified only when the kernel reports `residue_zero=True`. A
script that cannot make the residue vanish yields an unverified result, shown as
such and never as truth, and the detail says which way it failed: a script that
never reached the residue check (a script or kernel error, reported with the
stderr tail) reads differently from one that ran and found a nonzero residue.
The model writes a script; the kernel decides whether the answer holds. This covers the `vary` task for the metric, scalar, connection,
and gauge-field classes. On a Palatini (independent-connection) session the
MCP/HTTP `noether_derive` with `with_respect_to=['g','Gamma']` returns both EOMs
(metric and connection); on a session with the connection ambiguity still open
it returns a blocked dict with the open questions, never a guess. The planner
raises `AmbiguityBlocked` (HTTP 409) until every ambiguity is resolved, and
resolving the connection to `independent` enables the `INDEPENDENT_CONNECTION`
plan step. The `perturb` task now runs through the same model-written
path for scalar fields and the metric: `derive_perturbation` (reachable as
`kind="perturbation"` on the server, MCP, and web clients) hands the model the
`pert_scalar_quadratic` scaffold for scalars (eval 3p) or `pert_metric_quadratic`
for the metric (eval 3g), expanding the action to quadratic order (tracking
fluctuation order through Cadabra weights) and confirming the linearized equation
of motion both against the documented operator and by an independent route. All
checks must pass for the result to count as verified. The scaffolds cover
dynamical scalars and the metric, so the path refuses other field kinds.
When the connection is independent, the metric perturbation scaffold
`pert_metric_affine_quadratic` (eval 4ma) includes the connection fluctuation
`dG` alongside `h`; the connection is not perturbed independently. The
metric-affine perturbation result persists across surfaces (HTTP, MCP, store,
and web), with its `kind`, `verified` verdict, and `checks` intact, and the
eval is registered as the CLI subcommand `noether eval4ma`. The
`adm` task (reachable as `kind="adm"`) returns the ADM (3+1) decomposition of
the gravitational sector, the Gauss-Codazzi split and the Einstein-tensor
projections, verified by the SymPy component kernel on an explicit background
(eval 1s) rather than by a model-written Cadabra script. Any action carrying a
metric is accepted; one without a metric is refused (HTTP 422 / MCP error
naming the missing metric object) rather than guessed. Every derivation (EOM,
perturbation, and ADM alike) carries its full convention block (signature,
torsion sign, non-metricity definition, Ricci-contraction, contortion sign,
disformation sign, K-sign, foliation/normal convention) so no convention is
silently assumed.

## 5. Honesty and failure policy

- If a computation exceeds resource bounds, Noether reports where it stopped,
  what partial structure it found, and what would make it tractable (e.g.
  "restrict to quadratic order", "fix this gauge first"). It never returns a
  truncated expression silently.
- If a task needs a capability no installed kernel has, it says so and names the
  missing capability. No emulation by LLM algebra, ever.
- If verification fails, the user sees the failure first, prominently, with the
  failing check's script available. A wrong-looking verified result outranks a
  right-looking unverified one.

## 6. Development methodology

- **Eval-driven development.** `docs/04_EVALS.md` (and its executable mirror in
  `evals/`) is the definition of done. New capability work starts by writing or
  extending an eval, including its verification expectations. Horizon 1 ships
  when evals 1 to 4 pass end to end with their checks green; eval 5 and the
  stretch tasks gate Horizon 2.
- **Walking skeleton first.** The first milestone is the thinnest full loop:
  one action class (eval 1), one kernel (Cadabra2), CLI front, real provenance
  bundle, V0 to V2 checks. Breadth comes after the loop is trustworthy,
  matching "breadth follows trust" (NORTH_STAR §17).
- **Golden tests at the adapter boundary.** Every kernel adapter operation has
  pinned input/output pairs per kernel version. Kernel upgrades run the full
  golden suite plus eval reproduction before adoption.
- **Provenance from day one.** The bundle format exists in the walking skeleton,
  not retrofitted. `noether reproduce` works from the first shipped result.
  Cross-surface consistency is tested: the derivations returned by `POST /derive`
  equal those reloaded by `GET /results`, MCP `noether_results`, and the bundle
  `derivations.json` field for field by `result_id`.
- **Convention discipline.** All development, tests, and docs use
  `noether-default-v1` (AGENTS.md §5) unless explicitly testing convention
  handling, and convention-handling tests must cover at least one non-default
  block (opposite curvature sign) end to end. The guiding scenario's "she wants
  the opposite sign convention" is a test case, not an anecdote.
- **Bets get revisited.** NORTH_STAR §15 names five bets. Each horizon review
  checks them against evidence (e.g. do users actually welcome elicitation?
  does collect-by-structure tame the explosion?). Falsified bets trigger doc
  revisions, not quiet drift.
