# 01 — Research: the landscape we build on

**Status:** draft.
**Question this answers:** what already exists, what is good at what, and which
kernels Noether should orchestrate first.

The conclusion up front: the rigor Noether needs already exists, split across
two or three mature kernels with disjoint strengths and brutal ergonomics. Nobody
has built the translation-and-judgment layer on top of them. That is consistent
with the North Star's core claim, and it tells us to build adapters, not algebra.

---

## 1. Symbolic tensor kernels

### 1.1 xAct suite (Wolfram Language)

- **What it is:** the de facto research standard for abstract tensor algebra in
  gravity. A family of packages: xTensor (abstract tensors, covariant derivatives,
  Lie derivatives), xPerm (canonicalization of index permutations, C backend),
  xPert (perturbation theory to arbitrary order around arbitrary backgrounds),
  xCoba (component/basis computations), xTras (field-theory utilities: variational
  derivatives via `VarD`, `AllContractions`, Young projectors), Invar (Riemann
  invariant simplification database).
- **Strengths:** the strongest canonicalizer in the field (Butler-Portugal via
  xPerm); xPert makes high-order perturbation theory practical; xTras covers
  variation and basis construction; large body of published work depends on it,
  which gives us regression targets.
- **Weaknesses:** requires Wolfram Engine or Mathematica (licensing; the free
  Wolfram Engine for developers covers some uses but redistribution is
  constrained); steep, idiosyncratic API; error states can be silent; no native
  Python story (drive it via `wolframscript` or WSTP).
- **Role in Noether:** second kernel. Primary engine for perturbation theory
  (xPert) and as the cross-check kernel for canonicalization and variation.

### 1.2 Cadabra2 (open source, Python-embedded)

- **What it is:** a CAS written specifically for field theory problems (Peeters,
  2007; Cadabra2 rewrite 2018). Expressions are LaTeX-like by design. Ships as a
  Python module (`cadabra2`), so it embeds directly in a Python service.
- **Strengths:** field-theory-native operations out of the box: `vary`,
  `integrate_by_parts`, `canonicalise`, `substitute`, `meld`, Young-tableau-aware
  simplification, anticommuting objects, implicit dependence declarations. Input
  and output are close to physicist LaTeX, which shortens our translation layer.
  GPL, no license wall, easy to pin and containerize.
- **Weaknesses:** smaller ecosystem than xAct; ADM decomposition and perturbation
  theory are not turnkey (they are scriptable, and published worked examples exist,
  including GR and cosmology algorithm papers built on Cadabra + Python); fewer
  prebuilt identity databases than Invar.
- **Role in Noether:** primary kernel for Horizon 1. Variation, integration by
  parts, canonicalization, identity application. Its Python embedding makes the
  adapter the cheapest to build and the easiest to sandbox.

### 1.3 SymPy (Python)

- **What it is:** general-purpose Python CAS. Relevant parts: `sympy.tensor.tensor`
  (abstract index tensors with a Butler-Portugal canonicalizer), `sympy.diffgeom`,
  plus the broader scalar engine.
- **Strengths:** zero-friction dependency; fine for scalar manipulation, coupling
  functions, component checks on explicit backgrounds (FLRW, Minkowski), and as an
  independent canonicalization cross-check.
- **Weaknesses:** abstract tensor field theory support is thin compared to the two
  above; no practical variational calculus on curved-space actions; performance
  limits on large expressions.
- **Role in Noether:** utility kernel. Scalar algebra, dimensional analysis,
  component-level verification on explicit backgrounds, third opinion on
  canonical forms.

### 1.4 Others surveyed (not selected initially)

- **SageManifolds (SageMath):** excellent for component computations on explicit
  charts; not the abstract-index engine we need first. Candidate for the
  verification layer (evaluate both sides of a claimed identity on a random
  explicit metric).
- **Maxima itensor/ctensor:** venerable, but weaker canonicalization and a small
  maintenance community.
- **Redberry:** Java abstract-tensor CAS with good canonicalization; development
  dormant.
- **FieldsX, OGRe, MathGR, EinsteinPy, GraviPy:** useful single-purpose tools;
  none covers variation + canonicalization + perturbation at research grade.
- **Lean / mathlib and proof assistants:** wrong tool for production calculation
  today, but the "kernel verifies, model orchestrates" pattern from LLM+Lean work
  is exactly the trust architecture Noether commits to. Worth tracking.

### 1.5 Kernel comparison summary

| Capability | Cadabra2 | xAct | SymPy |
|---|---|---|---|
| Abstract index tensors | yes | yes | partial |
| Canonicalization (Butler-Portugal class) | yes | yes (xPerm, fastest) | yes (slower) |
| Variational derivative w/ IBP | yes (native) | yes (xTras `VarD`) | no |
| Independent connection / torsion | yes (scriptable) | yes (xTensor supports general connections) | no |
| Perturbation to order n | scriptable | yes (xPert, native) | no |
| ADM / foliation | scriptable | partial (xCoba + literature scripts) | no |
| Dimension-dependent identities | partial | yes (xTras, Invar) | no |
| License | GPL | Wolfram-dependent | BSD |
| Python embedding | native | subprocess (`wolframscript`) | native |

The split confirms the multi-kernel bet: no single kernel covers the North Star
capability surface, and two of the three best engines do not even share a host
language.

---

## 2. Prior art on the orchestration layer

What exists today between LLMs and CAS kernels:

- **Wolfram's LLM tooling** (LLM function kits, the 2026 "foundation tool" /
  computation-augmented generation push): proves the "model plans, kernel computes"
  pattern at the scalar/general-math level. Not tensor-aware, not elicitation-driven,
  no provenance contract.
- **MCP servers wrapping CAS engines** (community SymPy/Mathematica MCP servers,
  2025 onward): thin tool exposure. They hand the model a calculator; they do not
  manage problem representation, conventions, or verification. Useful precedent for
  our adapter transport, nothing more.
- **Language-agents-for-theoretical-physics literature** (e.g. arXiv 2506.06214 and
  follow-ups, 2025-2026): establishes both the promise and the documented failure
  mode: models confidently produce unverified symbolic claims. Supports the bright
  "kernel computed this" boundary as a hard requirement, not a nice-to-have.
- **Physics-guided equation discovery agents** (2026 preprints): adjacent but a
  different product. They search for equations from data; Noether derives
  consequences of a stated theory. Our anti-goal ("not a discovery engine") keeps
  us distinct.

Gap analysis: no existing system combines (a) notation-native LaTeX ingestion, (b)
structured elicitation that turns an under-determined action into a well-posed
problem, (c) multi-kernel orchestration behind a backend-agnostic representation,
and (d) provenance and cross-checking as a property of every answer. That
combination is the product.

---

## 3. Algorithmic foundations to reuse (not reinvent)

- **Index canonicalization:** Butler-Portugal algorithm (and xPerm's optimized
  implementation) for bringing monomials to canonical form under slot and dummy
  symmetries. We consume it through kernels; we never reimplement it.
- **Variational calculus on actions:** Euler-Lagrange with integration by parts and
  surface-term tracking; both Cadabra (`vary` + `integrate_by_parts`) and xTras
  (`VarD`) implement it. Palatini-type variation (independent connection, torsion,
  non-metricity) is standard in the metric-affine literature and scriptable in both.
- **Perturbation theory:** xPert's order-n expansion machinery around arbitrary
  backgrounds.
- **Identity reduction:** Bianchi and Ricci identities (kernel-native), Invar's
  database of Riemann invariant relations, dimension-dependent (Lovelock/Schouten)
  identities via xTras.
- **3+1 decomposition:** Gauss-Codazzi-Ricci relations as rewrite rules; published
  Cadabra and xAct worked examples exist for the ADM split of GR, which become our
  regression baselines.

---

## 4. Decisions this research implies

1. **Horizon 1 kernel: Cadabra2.** No license wall; covers variation, IBP,
   canonicalization, substitutions. Implementation note (2026-06-12): driven
   via the `cadabra2` CLI in a sandboxed subprocess; installed on macOS from
   the author's official Homebrew tap (`kpeeters/repo`), since neither
   homebrew-core nor conda-forge carries it. The in-process Python embedding
   remains an option later.
2. **Horizon 2 adds xAct** via `wolframscript` subprocess for perturbation theory
   and as the independent cross-check kernel. License handling becomes a real
   workstream at that point (Wolfram Engine free tier vs. user-supplied licenses).
3. **SymPy ships from day one** as the utility and component-check engine.
4. **The NPR (problem representation) is ours to design** (see `02_TECH_SPEC.md`).
   Nothing off the shelf plays this role; the MCP-style wrappers confirm that a
   bare tool interface without a shared representation is not enough.
5. **Regression corpus:** seed it from textbook results (vacuum GR, Maxwell,
   minimally coupled scalar) plus published xAct/Cadabra worked examples, so every
   capability lands with known-good targets. The five evals in `04_EVALS.md` are
   the first entries.

## 5. Open research questions

- How far does the free Wolfram Engine license actually stretch for a hosted
  service? (Legal reading required before Horizon 2 architecture freezes.)
- Cadabra2 behavior on very large expression trees (10^4+ terms): where does it
  degrade, and does `meld` suffice for our collect-by-structure needs?
- Best mechanical encoding of metric-affine variation in Cadabra (independent
  connection with torsion and non-metricity) at research grade: validate against
  the metric-affine literature before trusting the adapter (eval 2 is the gate).
- Random-component spot checks as a verification tier: how strong is evaluating a
  claimed tensor identity on pseudo-random explicit metrics as a falsifier? (Very
  strong in practice; quantify false-pass risk for our use.)
