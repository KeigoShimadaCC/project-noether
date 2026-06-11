# Project Noether — North Star

> **Working codename:** *Noether* — after Emmy Noether, because everything this tool
> does begins with an action, a set of symmetries, and a variation. Rename freely.

**Document type:** Vision / North Star.
**What this is:** the destination and the *why*.
**What this is not:** a spec, an architecture diagram, an API surface, or a methodology.
Those come later, and they serve this document — not the other way around.

**Companion documents** (each serves this one):

| Document | Role |
|---|---|
| `AGENTS.md` | Working guide for AI agents and contributors in this repo |
| `docs/00_INDEX.md` | Map of all project documents |
| `docs/01_RESEARCH.md` | Landscape: existing CAS kernels, prior art, what we build on |
| `docs/02_TECH_SPEC.md` | Architecture, stack, problem representation, kernel adapters |
| `docs/03_METHODOLOGY.md` | Elicitation protocol, good form, verification ladder, dev process |
| `docs/04_EVALS.md` | Acceptance evaluations with worked solutions |

---

## 1. The one sentence

**A physicist writes an action in LaTeX, answers a few questions, and gets back the
field equations, the 3+1 (ADM) decomposition, and the perturbed theory — in clean,
canonical, publication-ready form — without ever hand-translating their physics into a
computer-algebra dialect.**

If we only ever nail that one sentence, we have won.

---

## 2. The problem we are attacking

Symbolic tensor computation today forces a brutal trade. You can have **rigor** or you
can have **ergonomics**, almost never both.

- **The rigor exists.** xAct/xTensor, Cadabra, SymPy and friends can, in principle,
  vary an action, canonicalize indices, apply Bianchi identities, do a 3+1 split, and
  perturb around a background. These tools are correct and powerful.

- **The ergonomics are punishing.** To use them you must:
  - hand-translate physics notation into a specific, unforgiving CAS syntax;
  - declare, by hand, every symmetry, every index range, every convention;
  - know *in advance* which identities to apply and in which order;
  - fight expansion explosions, where a three-line Lagrangian becomes a
    ten-thousand-term mess that hides the answer instead of revealing it;
  - and read the output back out of the CAS dialect into something a human recognizes.

The result: a calculation that is conceptually an afternoon of work becomes weeks of
tooling. The barrier is not the mathematics. The barrier is the **translation layer**
between how a physicist thinks and what the machine demands — and the **judgment**
about which manipulations lead to insight versus noise.

That translation-and-judgment layer is exactly where today's tools are weakest and
where modern AI is strongest. That gap is the whole opportunity.

---

## 3. Who this is for, and the moment it serves

**Primary user:** a working theorist — graduate student, postdoc, or faculty — in
gravitation, cosmology, field theory, or high-energy theory. Someone fluent in the
physics and in LaTeX, who is *not* necessarily fluent in any particular CAS, and who
does not want to become a software engineer to get a derivation done.

**The moment:** they have an action. Maybe it is new. Maybe it is a known theory they
want to push one order further. They want to know its consequences — the equations of
motion, the constraint structure, the spectrum around a background — and they want the
result in a form they can check, trust, and paste into a paper.

**Secondary users we welcome but do not optimize for first:** advanced undergraduates
learning variational methods; researchers in adjacent fields (continuum mechanics,
condensed matter field theory) whose problems share the same shape; and the theorist's
*future self*, six months later, trying to reproduce what they did.

**A note on scope.** Noether is **general purpose**. It is for *any* local action — a
scalar field, a gauge theory, a higher-derivative gravity, an effective field theory,
a metric-affine model with torsion and non-metricity. The hard reference theories
(Horndeski-class scalar-tensor, Lovelock gravity, Palatini / metric-affine systems)
are our **proof of capability**, not the boundary of the product. If Noether can
handle those, it can handle most of what a theorist throws at it.

---

## 4. What it is

Noether is an **agentic symbolic-physics collaborator**. It pairs the judgment and
language fluency of a large language model with the rigor of established
computer-algebra kernels, and it keeps a human in the loop for exactly the decisions
that are irreducibly the physicist's to make.

The model **orchestrates**. The kernels **compute**. A verification layer **keeps the
model honest**. The human **decides what the problem actually is**.

The interface is conversational and notation-native: LaTeX in, LaTeX out, with a
running dialogue in between. The experience should feel less like operating software
and more like working with a meticulous, tireless collaborator who never makes an
arithmetic slip, never forgets a convention, and always shows their work.

---

## 5. What it is NOT

Stating the anti-scope is part of the vision.

- **Not a numerical relativity engine.** Noether is symbolic. It hands a clean,
  decomposed system to numerics; it does not evolve initial data.
- **Not a black box.** It never returns an answer it cannot justify, reproduce, and
  break down into checkable steps.
- **Not a replacement for the physicist's judgment.** It will not silently guess which
  symbols are dynamical, what symmetries hold, or what "the right gauge" is. It asks.
- **Not married to one CAS.** No single backend is the product. The product is the
  experience and the correctness; the kernels are interchangeable means.
- **Not a theorem prover or a discovery engine.** It computes consequences of a stated
  theory faithfully. It does not invent physics or claim results it did not derive.
- **Not, in this document, a methodology.** *How* we canonicalize, *how* we elicit,
  *how* we verify — all deferred. Here we only commit to *that* we do these things.

---

## 6. The core loop

The experience is a four-beat cycle, exactly as a good human collaboration would run.

1. **The user states the action.**
   They paste or type a LaTeX action — `S = \int d^4x \sqrt{-g}\,(\dots)`. Messy,
   abbreviated, convention-light: that is fine. The raw expression is the starting
   point, not a contract.

2. **Noether interrogates, the user answers.**
   Before computing anything, Noether resolves ambiguity by asking. *Which symbols are
   dynamical fields? Is this metric symmetric? Is the connection Levi-Civita, or
   independent with torsion and non-metricity? What dimension, what signature, what
   curvature sign convention? Vary with respect to what? Decompose along which
   foliation? Perturb around which background, to which order, in which gauge?* The
   user answers in plain language or by selecting options. This dialogue turns an
   ambiguous string into a **well-posed symbolic problem**.

3. **The symbolic engine computes.**
   With the problem now well-defined, Noether plans the computation, generates the
   appropriate kernel instructions, runs them, and drives the result toward *good form*
   — canonical indices, redundant terms removed, expressed in a basis the physicist
   recognizes. The expansion explosion is managed for the user, not surfaced at them.

4. **Noether returns the answer — and its provenance.**
   The result comes back in LaTeX, alongside the reasoning trail: the assumptions used,
   the identities applied, the checks that passed, and the exact kernel script that
   produced it. The user can accept, drill in, challenge a step, or change an
   assumption and re-run.

The loop is **resumable and stateful**. The problem definition, the conventions, and
the intermediate results persist across the session, so a long derivation is a
conversation with memory — not a series of disconnected one-shot prompts.

---

## 7. A guiding scenario

*(Illustrative, deliberately generic — not tied to any specific paper.)*

A postdoc is studying a higher-derivative scalar-tensor model. She pastes:

```
S = \int d^4x \sqrt{-g}\, [ K(\phi, X) + G(\phi, X)\,\Box\phi + F(\phi)\,R ]
```

Noether reads it back to her in clean LaTeX and asks a short, targeted set of
questions: *The dynamical fields are the metric and the scalar — confirm? The metric is
symmetric and the connection is its Levi-Civita connection — confirm? `X` is the
canonical kinetic term `-\tfrac12 \nabla_\mu\phi\nabla^\mu\phi` — confirm? Four
dimensions, mostly-plus signature?* She confirms, and corrects one thing: she wants the
opposite curvature sign convention. Noether notes it and carries it through everything
that follows.

She asks for the equations of motion. Noether varies with respect to both fields,
integrates by parts, collects, canonicalizes, and returns two clean tensor equations —
plus a note that it verified the metric equation is symmetric and divergence-free in
the appropriate limit, and that setting `G = 0` reproduces the textbook result. It
offers the kernel script that generated them.

Next she wants the 3+1 picture. Noether asks which foliation and gauge, performs the
ADM decomposition, and returns the lapse/shift/spatial-metric form with the constraint
and evolution pieces separated.

Finally she asks for the spectrum. Noether asks: *around which background, to what
order, in which gauge?* She says Minkowski, quadratic order, a convenient gauge. It
expands, isolates the propagating degrees of freedom, and reports the kinetic structure
— flagging, in passing, a term that vanishes on-shell and a would-be ghost mode that
the structure of the action removes.

Elapsed time: an afternoon, mostly spent thinking about physics. Not three weeks spent
fighting syntax. Every result carries its assumptions and its receipt.

---

## 8. Principles (the non-negotiables)

These are the commitments that, if violated, mean we have drifted off the North Star.

1. **Correctness is sacred; speed is negotiable.** A fast wrong answer is worse than
   useless in research. We never trade rigor for latency.

2. **No unearned assertions.** The model never states a symbolic result it did not
   obtain from a kernel. If it didn't compute it, it doesn't claim it.

3. **Show your work, always.** Every answer is accompanied by its assumptions, its
   identity chain, and a reproducible script. Provenance is a feature, not an add-on.

4. **Meet the physicist in their notation.** LaTeX in, LaTeX out. The human should
   never have to think in a CAS dialect. The dialect is our problem, not theirs.

5. **Ambiguity is resolved by asking, not by guessing.** What is a variable, what
   symmetries hold, what "good form" means — these belong to the human. We surface the
   choice; we do not silently make it.

6. **The human stays in the loop where judgment lives.** Noether is a collaborator,
   not an oracle. It is opinionated and proactive, but the physicist holds the wheel.

7. **Verifiability over magic.** A result the user cannot check is a liability. We
   prefer transparent, checkable derivations to impressive-looking opaque ones.

8. **Good form is the deliverable.** Producing *an* answer is the floor. Producing the
   *clean, canonical, recognizable* answer is the job.

9. **General first, special-cased never.** We build for the structure of variational
   field theory, not for one family of papers. Special cases are tests, not scope.

10. **Honest about limits.** When a computation is intractable, ill-posed, or beyond
    the current kernels, Noether says so plainly — and says why.

---

## 9. What Noether can do at the North Star

The end-state capability surface. When Noether is "done" in the North Star sense, a
physicist can:

- **Ingest an action from LaTeX**, in whatever abbreviated, convention-light form they
  naturally write, and have it parsed into a faithful internal representation.

- **Declare the problem interactively** — which symbols are dynamical fields, which are
  fixed backgrounds, which are coupling functions; the symmetries of every object
  (metric symmetric, field strengths antisymmetric, a connection with or without
  torsion/non-metricity); the dimension, signature, and conventions.

- **Derive the equations of motion** by varying with respect to one or several fields,
  including independent connections — with integration by parts and surface terms
  handled correctly.

- **Reduce to good form** — canonical index ordering, removal of redundant terms via
  Bianchi, symmetry, and dimension-dependent identities, and expression in a chosen
  basis of invariants or an irreducible decomposition.

- **Perform the 3+1 / ADM decomposition** along a stated foliation, separating
  constraints from evolution.

- **Perturb around a background** to a requested order, in a requested gauge, isolating
  the propagating degrees of freedom and their kinetic structure.

- **Inspect symmetries and identities** — read off Noether identities and conserved
  currents implied by the stated invariances of the action.

- **Receive everything in LaTeX**, with provenance, and **export the reproducible
  kernel script** for archival or independent re-check.

The reference bar: Noether should be able to walk a Horndeski-class, Lovelock-class, or
metric-affine derivation from action to spectrum. If it can do those, it can do most of
the field. The concrete, checkable version of this bar lives in `docs/04_EVALS.md`:
five worked action-to-result pairs that double as the acceptance tests for Horizon 1
and 2.

---

## 10. The idea of "good form"

"Good form" is subjective, and naming it precisely is part of the vision because it is
what separates a CAS dump from a usable result.

Good form means the expression is:

- **Canonical** — indices in a deterministic order, dummy indices normalized, so two
  equal expressions are visibly equal.
- **Minimal** — no redundant terms; everything removable by an identity (Bianchi,
  symmetry, dimension-dependent) has been removed.
- **Transparent** — written in a basis the physicist recognizes and reasons about,
  whether that is an irreducible decomposition, a preferred set of scalar invariants,
  or a conventional tensor structure.
- **Faithful** — provably equal to the input under the stated assumptions, with the
  transformation chain available for inspection.

Crucially, *good form is partly the user's call.* Two physicists may want the same
result in different bases. Noether's job is to converge to **the form this user
recognizes**, and to make the target negotiable rather than imposed.

---

## 11. What the AI must learn before it computes

The elicitation step is the heart of the human-in-the-loop value, so the vision names
the *kinds* of things Noether must establish (the specifics are methodology, deferred):

- **Roles of symbols** — dynamical field, fixed background, coupling function, constant.
- **Symmetries** — which tensors are symmetric, antisymmetric, or have mixed symmetry;
  whether a connection carries torsion and/or non-metricity, or is Levi-Civita.
- **Geometric setting** — dimension, metric signature, and whether a preferred
  foliation or frame is in play.
- **Conventions** — curvature sign, index placement, (anti)symmetrization weight, and
  the definitions of any composite shorthands the user employed.
- **The ask** — vary with respect to which fields; decompose along which foliation and
  gauge; perturb around which background, to which order, in which gauge.

The deeper principle: **a LaTeX action under-determines a symbolic problem.** The
dialogue is what closes the gap. Noether should ask the *fewest, sharpest* questions
needed — and never the ones it can already infer — so the interrogation feels like a
sharp colleague, not a form to fill out.

---

## 12. Trust and verification

For a research tool doing research-grade mathematics, trust is the entire game. A
result the user cannot believe is worthless, and a confidently-wrong result is
dangerous. So verification is not a phase — it is a property of every answer.

The North Star commitments:

- **Reproducibility.** Every result emits the exact kernel script that produced it.
  Anyone can re-run it.
- **Independent checking.** Where feasible, results are cross-checked — limiting cases
  that recover known theories, dimensional consistency, symmetry constraints the answer
  must satisfy, and regression against established results.
- **No hallucinated mathematics.** The boundary between "the model is reasoning about
  the problem" and "the kernel computed this" is bright and visible. Symbolic claims
  come from kernels, full stop.
- **Legible derivations.** The identity chain is shown, not hidden, so a skeptical
  physicist can audit each step rather than trusting a final box.

If a physicist cannot, in principle, verify what Noether returns, we have failed —
regardless of whether the answer happens to be correct.

---

## 13. Architecture vision (conceptual only)

Held deliberately at the level of *shape*, not design. Four conceptual layers:

- **The conversational orchestrator** — an LLM agent (built on a coding/agent SDK or
  CLI such as the Codex or Claude tooling) that parses intent, runs the elicitation
  dialogue, plans the computation, generates kernel instructions, interprets results,
  and decides when to retry. It holds the running state of the problem.

- **The problem representation** — a backend-agnostic internal description of the
  action, fields, symmetries, conventions, and the requested computation. This is the
  contract between "what the physicist meant" and "what any kernel will execute."

- **The symbolic kernels** — established CAS engines, used as interchangeable compute
  backends, chosen per task to fit the job (for instance, a kernel strong at
  metric-affine variation for the connection equations; one strong at perturbation for
  the spectrum). No kernel is *the product*.

- **The verification layer** — the checks, limiting cases, and reproducibility
  artifacts that wrap every result.

The bet embedded here: **a clean, backend-agnostic problem representation is what lets
the LLM orchestrate without being trusted to do the math, and lets us swap or combine
kernels as the ecosystem evolves.** We commit to the layering; we defer everything
about how each layer is built.

---

## 14. Why agentic, and why now

The shape of this problem is a near-perfect fit for an agentic LLM:

- The **hard, human-facing parts** — understanding an under-specified LaTeX action,
  knowing which clarifying questions matter, choosing a sensible computational route,
  translating intent into kernel code, and reading raw output back into physics — are
  exactly what language models now do well.

- The **rigorous parts** — index canonicalization, variation, identity application,
  decomposition — are exactly what CAS kernels already do well and what LLMs do badly.

The agentic framing lets each side do what it is good at: the model handles language,
judgment, planning, and orchestration; the kernel handles symbolic truth; and the loop
between them (plan → compute → check → revise) recovers from missteps the way a careful
human would. Three years ago the orchestration layer was not good enough to be trusted
with the messy front of this pipeline. Now it is. That is the "why now."

---

## 15. The hard parts — our bets

The risks we are knowingly taking on, stated as beliefs:

- **Trust is the make-or-break.** We bet that *verifiable, transparent* results — not
  raw capability — are what earn a researcher's confidence. If we are wrong about how
  much provenance physicists demand, we have mis-built the product.

- **"Good form" can be operationalized.** We bet that the fuzzy notion of a "clean,
  recognizable" result can be made concrete enough to target and negotiable enough to
  satisfy different users. This is genuinely hard and partly subjective.

- **The expansion explosion is tameable as a UX problem.** We bet that the
  combinatorial blow-up that makes naive CAS use miserable can be managed *for* the
  user — surfaced as insight, not as a ten-thousand-term wall.

- **Backend pluralism beats backend lock-in.** We bet that a clean problem
  representation over interchangeable kernels ages better than committing to one CAS.

- **Elicitation is a feature, not friction.** We bet that physicists will *welcome*
  sharp clarifying questions, because the alternative — silent wrong assumptions — is
  what they fear most.

Naming these as bets, not certainties, is itself a North Star commitment: we will
revisit them honestly as we learn.

---

## 16. What success looks like

Qualitative first, because that is what the North Star is for:

- A theorist reproduces a derivation that *used* to take weeks of tooling in an
  afternoon of thinking — and trusts the result enough to put it in a paper.
- The bottleneck in a calculation shifts back to *the physics* — what to compute and
  what it means — instead of *the tooling*.
- A researcher reaches for Noether before reaching for a blank CAS notebook, the way
  one reaches for a calculator before long division.
- The phrase "I checked it three ways and it holds" becomes the normal experience, not
  the lucky one.

If we later attach numbers, they should measure *that* — time-to-first-trusted-result,
fraction of results the user accepts without manual rework, breadth of theories handled
end to end — rather than vanity metrics. But the destination is the experience above,
not any single number.

---

## 17. Horizons (vision-level, not a roadmap)

Deliberately coarse. These are *altitudes*, not dated milestones or scoped work.

- **Horizon 1 — the walking skeleton.** The full four-beat loop works end to end on a
  meaningful class of actions: paste LaTeX, get interrogated, get equations of motion
  back in good form, with provenance. Narrow but *whole*. The point is to prove the
  loop, not to cover every theory. Concretely: evals 1 through 4 in
  `docs/04_EVALS.md` pass end to end.

- **Horizon 2 — the full pipeline.** ADM decomposition and perturbation join the
  equations of motion. Verification deepens (limiting cases, regression against known
  results). The set of supported theories and symmetries broadens toward the hard
  reference cases.

- **Horizon 3 — the trusted collaborator.** Multi-backend orchestration, robust
  cross-checking, and a track record across diverse theories. Noether becomes something
  a researcher relies on without double-checking it by hand — because it already
  checked itself, and showed the work. It also becomes available as infrastructure:
  an MCP server exposing the same session API, so any capable agent can delegate
  tensor calculus to Noether the way it delegates arithmetic to a sandbox
  (`docs/02_TECH_SPEC.md` §2).

The ordering principle across all horizons: **breadth follows trust.** We earn the
right to handle harder theories by being unimpeachably correct on easier ones first.

---

## 18. Anti-goals, restated

So we do not drift:

- We do not build a numerical evolution engine.
- We do not return results without provenance.
- We do not guess what only the physicist can decide.
- We do not lock ourselves to a single CAS.
- We do not optimize for one family of papers.
- We do not chase capability ahead of trust.
- We do not, in this document, descend into methodology or spec. That is the next
  document's job — and it will serve this one.

---

## 19. The North Star, once more

**A physicist writes an action in LaTeX, answers a few sharp questions, and gets back
the equations of motion, the ADM decomposition, and the perturbed theory — clean,
canonical, checkable, and reproducible — having spent their time on physics instead of
syntax.**

Everything we build is in service of that sentence. When a decision is unclear, we ask:
*does this get a physicist from an action to a trusted result faster, more correctly,
and more transparently?* If yes, it is on the path. If no, it is not.
