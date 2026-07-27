# Debugging

A troubleshooting runbook for the failures a contributor is most likely to hit. It is organized by symptom. When in doubt, run `.venv/bin/python -m noether.cli.main kernels` to see which kernel adapters are available.

## Kernel unavailable

### cadabra2 not on PATH

The cadabra-backed golden tests and evals skip cleanly when `cadabra2` is missing, but a `kernel_cadabra`-marked test that unexpectedly runs (or a manual `noether evalN` that needs Cadabra) will report the kernel as unavailable.

Fix:

```sh
brew tap kpeeters/repo && brew install cadabra2     # official macOS channel
```

Or point Noether at a non-default binary:

```sh
export NOETHER_CADABRA=/path/to/cadabra2
```

Confirm with:

```sh
.venv/bin/python -m noether.cli.main kernels
```

### SymPy version mismatch

`tests/test_versions.py` fails with `installed sympy X.Y is not the pinned 1.14.x series`. A SymPy minor-version bump can change canonicalisation, so the component checks are only meaningful against the pinned series.

Fix: check `noether/kernels/versions.py` (`sympy_matches_pin()`) and install the pinned series in your venv. If you genuinely intend to bump the pin, re-run the full eval suite and the cadabra golden tests, confirm every check is still green, and update `versions.py` in the same commit.

## AmbiguityBlocked from build_plan

`build_plan` in `noether/orchestrator/planner.py` raises `AmbiguityBlocked` while `npr.unresolved_ambiguities()` is non-empty. There is no code path that plans an under-specified problem. On HTTP this surfaces as `409` on `GET /plan`; on MCP it surfaces as a blocked dict rather than an exception.

Fix: resolve the open ambiguities before planning.

- HTTP: `POST /sessions/{id}/elicit` to get unconfirmed model proposals, then `POST /sessions/{id}/resolve` with validated answers (the resolve endpoint validates against the listed options; off-menu answers are rejected).
- CLI: `noether chat` walks the question flow interactively; `noether elicit "<action>" --accept-llm` accepts the model's proposals (still unconfirmed until you resolve).

A problem that is structurally not well-posed (no metric object, say) returns `409` on `GET /plan` even after elicitation. That is a different cause from an open ambiguity ledger; read the response body.

## Gated result (verified=false with non-empty detail)

A derivation can come back with `verified=false` and a non-empty `detail` field. This is the gated state, not a crash. `detail` always carries a blocker when gated (a confirmation reason when verified), enforced by a pydantic `model_validator` on `FieldDerivation`.

Read `detail` for the blocker. Examples:

- **SortCovDs for higher Horndeski.** The G4(phi, X) R and G5 densities route to `attempt_g4g5_eom` in `noether/orchestrator/derive.py`, which returns `verified=false` with a `detail` naming the SortCovDs blocker. This is expected; the higher Horndeski sector is held out and gated on purpose (VAL-EOM-013).
- **Covariant-curl vector perturbation on a metric-affine background.** The `F = nabla A` choice routes to `pert_vector_affine_covcurl_quadratic`; the residue is gated due to the Kronecker-delta limitation with mixed-index `dG` objects, and a SymPy cross-check provides independent verification (VAL-PERT-017, VAL-PERT-018).

The gated state is visible across HTTP, MCP, and the bundle. Do not paper over it; if a result is gated, the blocker is real and the detail tells you which one.

## HTTP 409 on GET /plan

Two causes, both in the response body:

1. **Open ambiguity ledger.** Resolve via `/elicit` and `/resolve` (see above).
2. **Problem not well-posed.** For example, an action with no metric object is refused for the ADM task. Re-visit the ingest step or the action.

## MCP blocked dict

MCP tools (`noether_derive`, `noether_plan`, etc.) return a blocked dict rather than raising when the session is not ready. The cause is the same as the HTTP equivalent: an open ambiguity ledger (resolve first), a not-well-posed problem, or a gated derivation. The blocked dict carries the same `detail` the HTTP surface would. Refusals are tool results, not exceptions, so a host LLM cannot make Noether guess.

## Sentinel parsing

The Cadabra adapter trusts only sentinel-marked lines from stdout. Only these count as signal:

- `NOETHER_RESULT:` - a computed expression.
- `NOETHER_CHECK:` - a boolean check outcome.
- `NOETHER_DETAIL:` - detail prose for the derivation.
- `NOETHER_CONVENTION:` - the convention block.

Everything else printed by the kernel is treated as noise. If a check you expected is missing, the script likely did not emit the sentinel line; do not grep the kernel's chatty output for the answer. Inspect the script in the provenance bundle and confirm the sentinel was emitted.

## LLM unavailable

The LLM backend is ambient-auth: it auto-detects an installed agent CLI (`codex`, `claude`, `gemini`, `droid`) and runs it one-shot as a sandboxed subprocess. If none is detected, the orchestrator cannot propose resolutions or write model scripts.

The error names which CLIs were looked for. Fix by installing and logging in to one of the supported agent CLIs. The deterministic tests use the `StubLLMAdapter` and do not need a real model, so this only affects interactive `noether chat` and `noether elicit` runs, plus the general model-written-script derivation path.

## The dual gate failing on a metric-affine result

For a metric-affine (independent-connection) result, a derivation is verified only when the Cadabra in-script residue check and the SymPy general-connection cross-check agree. If the dual gate fails, the most common cause is the **torsion trap**: a Levi-Civita shortcut silently dropped a torsion term and still reported a zero residue, but the SymPy oracle on explicit random metric and connection backgrounds caught it.

Fix:

1. Read the SymPy oracle stderr in the provenance bundle. The oracle runs on explicit backgrounds, so a mismatch points at the exact term.
2. Check the Cadabra script for an implicit metric-compatibility assumption (raising/lowering across `\nabla` as free, or a `# ::` declaration that pins the connection to Levi-Civita).
3. Confirm the convention block threaded into the script matches the elicited one (torsion sign, non-metricity definition, contortion sign, disformation sign, Ricci-contraction).

The dual gate is described in [`docs/02_TECH_SPEC.md`](../../../docs/02_TECH_SPEC.md) and the [Verification](../systems/verification.md) page. It is what makes "verified" mean something on a metric-affine background.

## Session state

NPR versions are immutable and append-only (`Session.npr_versions`). Resolving an ambiguity, confirming a menu answer, or adopting a shorthand each produces a new immutable version. Results reference the version they were computed against.

Two consequences to keep in mind when debugging session behavior:

- A late resolution does not edit or drop prior results; it marks them stale. If a result you reloaded looks stale after a resolve, that is correct, not a bug.
- A resumed session (`noether resume <id>`) carries geometry resolutions, NPR version history, and result ids intact, so a follow-up derive needs no re-elicitation. If a resume asks you to re-resolve, the session was not persisted or the id is wrong.

## Provenance bundle

Every derivation writes a provenance bundle: the action, the plan, the kernel script, every check the kernel reported, and the result. When a derivation misbehaves, read the bundle first. The bundle is also what the web client renders as a provenance tree, so the data you see in the browser is the same data you debug here. See [Verification](../systems/verification.md) and [Kernels](../systems/kernels/index.md).
