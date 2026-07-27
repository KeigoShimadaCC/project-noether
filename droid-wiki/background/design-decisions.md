# Design decisions

These are the load-bearing choices in Project Noether and the failure mode each one exists to prevent. Every choice is enforced mechanically, in code, not by a guideline. Where a North Star principle or AGENTS.md rule backs a decision, it is cited.

## The NPR boundary

**What.** The Noether Problem Representation (NPR) is the only language the orchestrator speaks to a kernel. It is a versioned, backend-agnostic pydantic schema (`noether/npr/schema.py`) describing the action, fields, symmetries, conventions, and the requested task. Kernels are reached through `KernelAdapter` (`noether/kernels/base.py`), selected by capability, never by name. Nothing outside a kernel adapter may import or depend on a specific CAS.

**Why.** The system must not marry a single CAS. Cadabra2, SymPy, and (planned) xAct each have strengths; a clean problem representation lets the orchestrator swap or cross-check them without rewriting orchestration. This is North Star principle 4 in architectural form and AGENTS.md rule 6 ("no backend lock-in"). The bet, named in NORTH_STAR.md section 15, is that backend pluralism ages better than commitment to one engine.

**Enforced by.** AGENTS.md rule 6. The `Capability` enum and `KernelAdapter` protocol in `noether/kernels/base.py` are the contract surface. See [../systems/npr.md](../systems/npr.md).

## The provenance boundary and no unearned assertions

**What.** Only a `ComputedResult` (`noether/kernels/base.py`) can carry a computed expression into a result. The model can write a kernel script, but the script carries no authority: a result is trusted only after a kernel's own in-script check confirms it. The `verified` flag is set by the kernel, never by the model or the orchestrator.

**Why.** An LLM hallucinating a tensor identity is the central failure mode of this product. NORTH_STAR.md section 12 names it: "the boundary between 'the model is reasoning about the problem' and 'the kernel computed this' is bright and visible. Symbolic claims come from kernels, full stop." A result with no receipt is, in spirit and eventually in fact, a type error. Mechanical enforcement, not policy, is what makes principle 2 ("no unearned assertions") survive a hurried contributor.

**Enforced by.** AGENTS.md rule 3 ("provenance is part of the result type") and the `ComputedResult` model: a result that did not come through it carries no provenance and must never reach the user. The LLM adapter docstring in `noether/llm/base.py` states the adapter "carries no authority over physics: it cannot inject a computed expression into a result (only kernels can)." See [../systems/provenance.md](../systems/provenance.md).

## The ambiguity gate

**What.** `build_plan` in `noether/orchestrator/planner.py` raises `AmbiguityBlocked` while `npr.unresolved_ambiguities()` is non-empty. Only a human-confirmed, on-menu answer mutates the NPR, via `apply_resolutions` in `noether/orchestrator/elicit.py`. The model may propose resolutions with rationale, but proposing never mutates the NPR; off-menu suggestions yield `choice=None` and leave the NPR unchanged.

**Why.** A guessed field role, symmetry, or gauge produces a confident wrong answer that is far more expensive to debug than a question is to ask. NORTH_STAR.md principle 5 is "ambiguity is resolved by asking, not by guessing." The deeper principle in section 11 is that a LaTeX action under-determines a symbolic problem, so the dialogue is what closes the gap. AGENTS.md rule 4 sends ambiguity to the human.

**Enforced by.** AGENTS.md rule 4. `AmbiguityBlocked` in `planner.py` is the structural block; `apply_resolutions` is the single mutation path; the geometry-inference test suite pins the propose-never-mutates contract. See [../systems/orchestrator.md](../systems/orchestrator.md).

## The dual gate for metric-affine results

**What.** A metric-affine result is called verified only when the Cadabra residue check and the SymPy general-connection cross-check agree. The SymPy oracle recomputes on explicit random metric and connection backgrounds, independently of the Cadabra script.

**Why.** This is the defense against the torsion trap: a Levi-Civita shortcut silently drops a torsion term and still reports a zero residue, because the dropped term was never there to leave a residue. Independent recomputation on explicit random backgrounds catches the drop. NORTH_STAR.md section 12 commits to independent checking, and principle 1 makes correctness sacred over speed. The dual gate makes that commitment mechanical for the hardest case in the system.

**Enforced by.** The dual-gate verdict logic in `noether/orchestrator/derive.py` and the SymPy cross-check path documented in `docs/02_TECH_SPEC.md` section 6. See [../systems/verification.md](../systems/verification.md).

## Ambient-auth LLM transport

**What.** The implemented LLM adapter shells out to an ambient agent CLI (codex, claude, gemini, or droid, auto-detected) as a one-shot subprocess. Noether holds no API key; credentials live in the agent CLI's own login session.

**Why.** Credentials stay out of the repository and out of process memory that Noether controls, which keeps AGENTS.md's secrets rule honest. The model is treated as just another sandboxed tool behind a clean adapter, mirroring the Cadabra subprocess transport. `docs/02_TECH_SPEC.md` section 9 records the caveat that agent CLIs are built for interactive use, so headless programmatic use may bump their terms and should be revisited before any distribution.

**Enforced by.** `noether/llm/base.py` (the `LLMAdapter` protocol and the no-authority docstring) and the ambient-auth adapter. See [../systems/llm.md](../systems/llm.md).

## Immutable NPR versions

**What.** NPR versions are append-only. A late resolution does not edit the version a result was computed against; it creates a new version and marks prior results stale.

**Why.** Results reference the NPR version they were verified on. If a late answer could rewrite that version, the provenance receipt would describe a problem that no longer exists, and auditability breaks. Marking stale keeps the history honest: the old result stands as computed, and the user is told it predates the current problem. This is North Star principle 3 ("show your work, always") in persistence form.

**Enforced by.** The NPR version history in `noether/npr/schema.py` and the stale-marking path in the session store. See [../systems/npr.md](../systems/npr.md).

## Frozen golden templates and compositional blocks

**What.** The audited eval derivations run from frozen kernel templates (`noether/kernels/cadabra/templates.py`) that the model cannot edit. For additive Lagrangians that decompose fully into registered building blocks, `noether/kernels/cadabra/blocks.py` assembles one script for the real action plus an independent candidate, and the kernel residue-checks it, with no model in the loop and no per-theory template.

**Why.** A frozen script the model cannot edit is the trusted offline core: the evals' correctness does not depend on model behavior. The compositional path extends coverage past the frozen set without writing a new template per theory, while still being kernel-checked rather than model-asserted. This keeps principle 2 intact as the supported theory set grows.

**Enforced by.** AGENTS.md rule 5 ("evals are acceptance tests") and the residue-check verdict in `derive.py`. See [../systems/kernels/cadabra.md](../systems/kernels/cadabra.md).

## The teaching channel as a separate field

**What.** `FieldDerivation.teaching` in `noether/orchestrator/derive.py` is pure prose that narrates the geometry tradeoffs of a metric-affine derivation (what torsion implies for spin coupling, what projective freedom means for the connection equation). It is distinct from `detail` (the verdict reason or blocker) and from `result_tex` (the kernel output). Generating teaching mutates no NPR and sets no result.

**Why.** The verified/reasoned boundary must stay visible to the user. Folding reasoned explanation into the result expression would blur the bright line from principle 2; folding it into `detail` would make a gated result look explained rather than blocked. Keeping teaching as its own field lets the model be useful about geometry without ever claiming a computed result.

**Enforced by.** The `FieldDerivation` model and the `_geometry_teaching` function in `derive.py`; the teaching-channel test suite pins the "teaching mutates no NPR, sets no result" contract. See [../systems/orchestrator.md](../systems/orchestrator.md).

## Honest gating

**What.** When a computation is beyond the current kernels, the system returns `verified=false` with a non-empty `detail` naming the blocker, rather than a fabricated answer. The worked case is higher Horndeski G4/G5: `attempt_g4g5_eom` in `derive.py` returns `verified=false` with a `detail` naming the SortCovDs blocker. `FieldDerivation._detail_must_be_nonempty` enforces that a gated result is distinguishable from a verified one by both `verified` and `detail`.

**Why.** A fabricated answer is worse than an honest "not yet". NORTH_STAR.md principle 10 is "honest about limits," and section 12 says a confidently-wrong result is dangerous. The validator makes the honesty mechanical: there is no `verified=false` with an empty reason, so a gate is never silent.

**Enforced by.** The `_detail_must_be_nonempty` validator and the G4/G5 path in `derive.py`; AGENTS.md rule 1 ("never assert a symbolic result you did not compute").
