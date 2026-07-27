# Patterns and conventions

The rules here are the operational form of the North Star principles. They are enforced by structure and tests wherever possible, not just by review. `AGENTS.md` is the authoritative contributor contract; this page summarizes the patterns a contributor meets in the code.

## The non-negotiable rules

1. **Never assert a symbolic result you did not compute.** Any tensor identity, variation, or simplification in code, tests, docs, or chat must come from a kernel run or a citable standard result, and must be marked as such.
2. **Conventions are always explicit.** Every expression crossing a kernel boundary carries its convention block. No file, function, or test assumes a convention silently. Repo defaults exist but are referenced by name.
3. **Provenance is part of the result type.** A function that returns a computed expression returns it with the script, kernel version, and assumptions that produced it. A bare expression with no receipt is a type error in spirit.
4. **Ambiguity goes to the human.** Product code never silently guesses field roles, symmetries, or gauges.
5. **Evals are acceptance tests.** A capability does not exist until its eval in `docs/04_EVALS.md` passes end to end with checks green. Add the eval before the capability.
6. **No backend lock-in.** Nothing outside a kernel adapter imports or depends on a specific CAS. The NPR is the only language the orchestrator speaks.
7. **Correctness over speed, everywhere.** Do not cache, approximate, truncate, or parallelize in a way that can change a symbolic answer.

## How the rules show up in code

### The model has no authority

The LLM adapter's only job is `complete(system, prompt) -> text` (`noether/llm/base.py`). It cannot inject a computed expression into a result (only kernels can) and it cannot resolve an ambiguity (only a confirmed human answer can, via `noether.orchestrator.elicit.apply_resolutions`). When the model writes a Cadabra script, that script is tagged "generated (parameterized; unverified until the ladder confirms it)" in `noether/kernels/cadabra/adapter.py`, and the `verified` flag is set by the kernel's own check, never by the model or the orchestrator.

### The ambiguity gate is structural

`build_plan` in `noether/orchestrator/planner.py` raises `AmbiguityBlocked` while `npr.unresolved_ambiguities()` is non-empty. There is no code path that plans an under-specified problem. The HTTP `GET /plan` returns 409 in that state, and MCP tools return a blocked dict rather than raising.

### Immutable NPR versions

NPR versions are append-only (`Session.npr_versions`). Resolving an ambiguity, confirming a menu answer, or adopting a shorthand each produces a new immutable version (`noether/orchestrator/session.py`). Results reference the version they were computed against; a later resolution marks prior results stale rather than editing them.

### Sentinel-parsed kernel output

The Cadabra adapter trusts only sentinel-marked lines from stdout (`NOETHER_RESULT:`, `NOETHER_CHECK:`, `NOETHER_DETAIL:`, `NOETHER_CONVENTION:`). Everything else is treated as noise. This keeps parsing robust against a chatty kernel.

### `detail` is always non-empty

`FieldDerivation` has a pydantic `model_validator` that rejects an empty `detail`. Every derivation path must populate it: a confirmation reason when verified, a blocker when gated. This makes the verified-vs-gated distinction visible on every surface.

### The AST node / geometric-cue coupling

`noether/npr/ast.py` defines the `Expr` node types. `noether/orchestrator/elicit.py` `_detect_geometric_cues` walks the AST with a `match` statement covering every node type. Adding a new node type without adding a corresponding case raises `UnhandledASTNodeError` at runtime. This is fail-loud by design; the fix is to add the case.

## Engineering conventions

- **Language**: Python 3.12+, full type annotations, `pydantic` v2 for the schema. Frontend is TypeScript with React 19 and Next.js.
- **Formatting and linting**: `ruff` (format plus lint), line length 100, rule set `E, F, I, UP, B`. Frozen kernel-script files (`templates.py`, `blocks.py`, `curvature.py`, `horndeski_g4g5.py`) are exempt from `E501` because their lines are verbatim artifacts.
- **Kernel versions** live in one place, `noether/kernels/versions.py` (SymPy `1.14`, Cadabra `2.5.15`). Bumping a pin is a deliberate act: re-run the full suite and update the file in the same commit.
- **Kernel runs** are sandboxed subprocesses with timeouts.
- **Commits**: small, imperative subject lines; the body explains the why.
- **Secrets**: API keys only via environment, never committed or logged. The ambient-auth LLM design keeps credentials inside the agent CLI's own login session.

## Working a task

1. Locate the task against the horizon plan and the tech spec. If it expands scope, flag it.
2. If it adds capability, write or extend the eval first.
3. Implement behind the NPR boundary: orchestrator logic stays kernel-agnostic, kernel specifics stay in adapters. V0 validation stays structural.
4. Run the relevant evals and tests. A physics-bearing change with no kernel-backed test does not merge.
5. Update the docs the change touches, in the same change.
6. In the summary, separate "what the kernel verified" from "what I reasoned about". That boundary is the product's core promise.

See [Development workflow](development-workflow.md), [Testing](testing.md), and [Tooling](tooling.md) for the mechanics.
