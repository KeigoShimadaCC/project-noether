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
  the connection is independent, what to vary, which background and gauge).
  These are always asked, never defaulted.

### 1.2 Question discipline

- Batch related questions; one elicitation round should usually suffice for EOM
  tasks (the guiding scenario in NORTH_STAR §7 shows the target feel).
- Each question states why it matters when non-obvious ("If Γ is independent,
  you get a second field equation").
- Read-back: after elicitation, Noether restates the complete problem (fields,
  roles, symmetries, conventions, ask) in clean LaTeX and waits for confirmation.
  This read-back is stored as the assumption snapshot in the provenance bundle.
- Mid-session changes are first-class: changing an answer creates a new NPR
  version and explicitly marks dependent results stale.

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
motion, and physicists routinely shorthand these as `F_φ`, `F_φφ`. Noether
proposes those shorthands (`noether.orchestrator.definitions.propose_definitions`,
exposed at `GET /sessions/{id}/definitions`, the MCP tool
`noether_propose_definitions`, and the web client's "Suggested notation" card).

Two boundaries keep this honest:

- These are **definitions, not results.** `F_φ` is introduced as a name for the
  derivative `∂F/∂φ`; nothing here claims what any particular variation
  evaluates to, so AGENTS.md rule 1 is not engaged (there is no asserted result,
  only notation). Proposals are deterministic functions of the declared function
  couplings and their arguments.
- Adoption is the **human's choice** (rule 4). Proposing never mutates the NPR;
  only an accepted proposal is applied, as a new immutable NPR version carrying a
  `shorthand` `ObjectDecl` whose `definition_tex` records the meaning. Couplings
  the human has pinned to a constant are not offered (their derivatives vanish),
  and already-declared symbols are never re-proposed, so the proposal set
  converges as notation is adopted. Adopting notation neither opens nor closes
  the ambiguity gate.

## 4. The verification ladder

Every result climbs as far up the ladder as its class allows. Verdicts ship in
`checks.json` and in the user-facing narrative. Failure at any rung blocks
PRESENT from labeling the result verified; Noether says plainly which checks ran,
which passed, and which were not applicable.

- **V0 — well-formedness.** Index balance, symmetry consistency, dimensional
  homogeneity of every term, round-trip parse of the output LaTeX. Cheap,
  always on.
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
  SymPy/SageManifolds) reproduces the canonical form. Mandatory for novel-theory
  results once the second kernel lands (H2); spot-check variant available in H1.

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
such and never as truth. The model writes a script; the kernel decides whether
the answer holds. This covers the `vary` task for the metric, scalar, and
gauge-field classes; `adm` and `perturb` wait on their own audited scaffolds
rather than being guessed.

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
- **Convention discipline.** All development, tests, and docs use
  `noether-default-v1` (AGENTS.md §5) unless explicitly testing convention
  handling, and convention-handling tests must cover at least one non-default
  block (opposite curvature sign) end to end. The guiding scenario's "she wants
  the opposite sign convention" is a test case, not an anecdote.
- **Bets get revisited.** NORTH_STAR §15 names five bets. Each horizon review
  checks them against evidence (e.g. do users actually welcome elicitation?
  does collect-by-structure tame the explosion?). Falsified bets trigger doc
  revisions, not quiet drift.
