# Security

Project Noether is a research prototype, not a hardened production system. This page documents the trust boundaries that the design does enforce and the properties it structurally guarantees. Nothing here is a claim of production readiness.

## Trust boundaries

### Ambient-auth LLM, no API key

The LLM adapter (`noether/llm/cli.py`) does not hold, transmit, or store any credential. It auto-detects an installed agent CLI on `PATH` (`codex`, `claude`, `gemini`, `droid`, in that order) and runs it as a one-shot subprocess in headless mode. Authentication lives entirely in that CLI's own login session. Noether never sees an API key, never writes one to the environment, and never logs one. If no agent CLI is found, `complete()` raises `LLMError` naming the executables it looked for; it does not fall back to anything.

See `../systems/llm.md` for the adapter contract and `../background/design-decisions.md` for the rationale.

### Sandboxed kernel subprocesses

Cadabra2 runs as a sandboxed subprocess in a `TemporaryDirectory` with a timeout (default 300 seconds; configurable per adapter). The adapter (`noether/kernels/cadabra/adapter.py`) honors the `NOETHER_CADABRA` environment variable to point at a non-default binary, then writes the script, runs `cadabra2`, captures stdout/stderr verbatim, and parses results. Kernel scripts are either frozen golden templates (`noether/kernels/cadabra/templates.py`) or model-generated scripts assembled by the orchestrator. Model-generated scripts carry no authority of any kind until the kernel's own residue check confirms them; a script that fails the check is recorded as `verified=false` with a non-empty `detail` naming the blocker.

### Sentinel-parsed kernel output

Only four sentinel-prefixed lines count as kernel output:

- `NOETHER_RESULT:` the result expression
- `NOETHER_CHECK:` a check verdict
- `NOETHER_DETAIL:` a detail string
- `NOETHER_CONVENTION:` a convention field

Everything else the kernel prints is treated as noise and discarded. This makes the parser robust against a chatty kernel, benign warnings, or a malicious stdout stream that tries to smuggle content past the boundary. A kernel cannot inject a result by printing prose; it must print the sentinel.

## Structural safety properties

### The no-guessing contract

The system structurally refuses to produce an answer for an under-specified problem. `build_plan` (`noether/orchestrator/planner.py`) raises `AmbiguityBlocked` while the NPR's ambiguity ledger is non-empty, so planning cannot complete and no derivation runs. On the HTTP surface this surfaces as `409 Conflict`; on the MCP surface it surfaces as a `blocked` dict. A host LLM cannot make Noether guess, because the gate is in the planner, not in the model's output. See `../systems/orchestrator.md` and `../systems/npr.md`.

### The provenance boundary

Only a `ComputedResult` (`noether/kernels/base.py`) enters a derivation result. The model writes a script; the kernel runs it; the kernel's residue check sets `verified`. The model cannot inject a hallucinated tensor identity into `result_tex`, because `result_tex` is populated from the kernel's sentinel output, not from the model's prose. The `FieldDerivation.detail` field is always non-empty (a pydantic validator rejects an empty string), so a gated result is distinguishable from a verified one across every surface. See `../primitives/computed-result.md` and `../systems/provenance.md`.

## Input validation

- `/resolve` (`noether/server/app.py`) validates every answer against the listed options for the matching ambiguity; an off-menu answer is rejected, never silently coerced.
- The NPR schema (`noether/npr/schema.py`) is a pydantic v2 model; structural validation runs on every ingest and every resolution.
- V0 structural validation (`noether/npr/validate.py`) checks index balance and expected free indices before any kernel runs.
- The MCP surface returns `error` or `blocked` dicts on refusal paths, never a fabricated `verified` result.

## Network and deployment surface

- No remote services, no database, no network egress except the local agent CLI subprocess.
- The HTTP server (`noether serve`) binds `127.0.0.1:8754` (loopback only).
- The MCP server (`noether mcp`) speaks stdio.
- The web client (`frontend/`) proxies `/api/*` to the loopback server via `next.config.mjs` rewrites; the browser holds no physics state.

## Licensing

Cadabra2 is GPL-3.0 and is invoked as a subprocess, not linked into Noether. The repository metadata (`.github/REPO_METADATA.md`) records that a license for Noether itself has not yet been chosen; that decision is pending before the repo goes public. See `../background/design-decisions.md` for the licensing rationale.
