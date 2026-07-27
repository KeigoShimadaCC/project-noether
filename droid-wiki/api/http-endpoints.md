# HTTP endpoints

## Purpose

The FastAPI HTTP session API in `noether/server/app.py` exposes the orchestrator loop as JSON endpoints. This page is the contract reference: method, path, request body, response shape, and the no-guessing behavior of each endpoint. For how to run the server, see `../apps/http-server.md`.

The app is built by `create_app(store=None, llm=None, results_root=None)`. With `llm=None` each request defers to an auto-detected agent CLI (`CliLLMAdapter`); tests inject a stub. The default host and port are `127.0.0.1:8754`, set by the `noether serve` CLI subcommand in `noether/cli/main.py`.

## Endpoints

| Method | Path | Status | Contract note |
| --- | --- | --- | --- |
| `GET` | `/health` | `200` | Reports kernel availability; no session state. |
| `GET` | `/sessions` | `200` | Lists stored session ids. |
| `POST` | `/sessions` | `201` / `422` | Ingests a LaTeX action; `422` on a parse error. |
| `GET` | `/sessions/{id}` | `200` / `404` | Returns the session payload; `404` if unknown. |
| `POST` | `/sessions/{id}/elicit` | `200` / `503` / `502` | Unconfirmed model proposals only; never mutates the session. |
| `POST` | `/sessions/{id}/resolve` | `200` / `404` / `400` | Applies human-confirmed on-menu choices; off-menu is `400`, unknown id is `404`. |
| `GET` | `/sessions/{id}/definitions` | `200` | Proposes readability shorthands; `confirmed: false`. |
| `POST` | `/sessions/{id}/definitions` | `200` / `404` | Adopts selected proposals; `404` for an unknown proposal id. |
| `GET` | `/sessions/{id}/plan` | `200` / `409` | `409` with the open question list while the ledger is non-empty. |
| `POST` | `/sessions/{id}/derive` | `200` / `409` / `422` / `503` / `502` / `400` | Runs a kernel-checked derivation; see derive contract below. |
| `GET` | `/sessions/{id}/results` | `200` / `404` | Reloads the full derivation history from provenance bundles. |

### GET /health

No request body. Returns:

```json
{
  "status": "ok",
  "kernels": {
    "sympy": {"available": true, "version": "..."},
    "cadabra": {"available": true, "version": "..."}
  }
}
```

`available` is `false` and `version` is `null` when a kernel binary is not on the server.

### GET /sessions

No request body. Returns `{"sessions": ["s-...", ...]}`, the list of stored session ids.

### POST /sessions

Request body: `CreateSessionRequest`.

```python
class CreateSessionRequest(BaseModel):
    lagrangian: str
    measure: str = r"d^4x \sqrt{-g}"
```

Parses the LaTeX Lagrangian density into a new session, runs `ingest_action`, and stores it. Returns a `session_payload` with `state: "elicit"`, `well_posed: false`, and the open `questions` (each with `id`, `question`, `kind`, `options`, `resolution: null`). A metric-curvature action such as `R` opens the full geometry questionnaire (`amb-connection`, `amb-torsion`, `amb-nonmetricity`, `amb-metric-compatibility`, `amb-curvature-free`).

A `ParseError` becomes `422` with `{"detail": "parse error: ..."}`.

### GET /sessions/{id}

No request body. Returns the `session_payload` for the session. `404` if the id is unknown.

### POST /sessions/{id}/elicit

No request body. Calls `propose_resolutions` with the configured LLM adapter and returns unconfirmed proposals:

```json
{
  "confirmed": false,
  "note": "model proposals only; confirm through POST /sessions/{id}/resolve",
  "llm": {"name": "...", "version": "..."},
  "proposals": [
    {"ambiguity_id": "amb-connection", "choice": "levi-civita", "rationale": "..."}
  ]
}
```

This endpoint never mutates the session. Off-menu suggestions are already discarded by `propose_resolutions` (they come back with `choice: null`). `503` when no LLM backend is available; `502` on an `LLMError`.

### POST /sessions/{id}/resolve

Request body: `ResolveRequest`.

```python
class ResolveRequest(BaseModel):
    resolutions: dict[str, str] = Field(min_length=1)
```

The map keys are ambiguity ids; the values are the human-confirmed choices. `Session.confirm_resolutions` validates each choice against the ambiguity's listed `options`. An unknown ambiguity id is `404`; an off-menu choice is `400` with the `ValueError` message, and the session is left unchanged. On success the session is saved and the `session_payload` is returned with `well_posed: true` once the ledger closes.

If results already exist when a resolution lands, `mark_results_stale` is called with the reason `"assumption resolved after results existed"`, so prior derivations are flagged stale rather than silently trusted.

### GET /sessions/{id}/definitions

No request body. Returns proposed readability shorthands:

```json
{
  "confirmed": false,
  "note": "readability notation only; these are definitions, not results. ...",
  "proposals": [
    {"id": "def-F-phi", "symbol": "F_phi", "symbol_tex": "...",
     "meaning_tex": "...", "definition_tex": "...", "rationale": "..."}
  ]
}
```

These are notation, not results. Adopting them never reopens a question.

### POST /sessions/{id}/definitions

Request body: `AdoptDefinitionsRequest`.

```python
class AdoptDefinitionsRequest(BaseModel):
    accept: list[str] = Field(min_length=1)
```

Each entry is a proposal id from `GET /definitions`. An unknown id is `404`. On success the shorthands are added to the session and the `session_payload` is returned.

### GET /sessions/{id}/plan

No request body. Calls `Session.plan()`. While the ambiguity ledger is open this raises `AmbiguityBlocked`, which the endpoint turns into:

```json
{"detail": {"blocked": true, "questions": [...]}}
```

with status `409`. Once well posed it returns:

```json
{
  "task_type": "vary",
  "steps": [{"capability": "...", "description": "..."}],
  "verification": "..."
}
```

The plan is never a guess; the `409` carries the questions so the client can relay them.

### POST /sessions/{id}/derive

Request body: `DeriveRequest` (optional; defaults to `kind: "eom"`, no `with_respect_to`).

```python
class DeriveRequest(BaseModel):
    with_respect_to: list[str] | None = None
    kind: str = "eom"  # "eom" | "perturbation"
```

`kind` is `"eom"`, `"perturbation"`, or `"adm"`. An unknown kind is `422`. For `eom` and `perturbation`, cadabra must be installed (`503` if not) and an LLM backend must be available (`503` if not, `502` on `LLMError`). For `adm`, only the SymPy component kernel is used and no model script is written.

`with_respect_to` restricts the variation to specific declared objects. A name that is not a declared object is `400`. For `kind: "eom"`, the list is written into `npr.task.with_respect_to` on a deep copy; for `perturbation` it is passed as the `fields` argument.

While the ledger is open, `AmbiguityBlocked` becomes `409` with the question list. A `NotImplementedError` (for example, perturbing the connection field on a metric-affine background) becomes `422`.

On success the endpoint records each result id into the session (deduped, so a repeat derive does not grow history) and returns:

```json
{
  "session_id": "s-...",
  "derivations": [FieldDerivation.model_dump(), ...]
}
```

### GET /sessions/{id}/results

No request body. Returns the `results_payload`:

```json
{
  "session_id": "s-...",
  "results": [FieldDerivation.model_dump(), ...],
  "stale_result_ids": ["r-...", ...]
}
```

The derivations are reloaded from their provenance bundles on disk, so this endpoint runs no physics. `stale_result_ids` names any result an assumption change has since invalidated. `404` if the session id is unknown.

## Response shapes

### session_payload

Returned by `POST /sessions`, `GET /sessions/{id}`, `POST /sessions/{id}/resolve`, and `POST /sessions/{id}/definitions`. Constructed by `session_payload` in `noether/orchestrator/view.py`:

```json
{
  "session_id": "s-...",
  "state": "elicit",
  "well_posed": false,
  "action": {"measure_tex": "...", "lagrangian_tex": "..."},
  "objects": [{"name": "...", "kind": "...", "role": "...", "definition_tex": "..."}],
  "questions": [
    {"id": "amb-...", "question": "...", "kind": "inferable",
     "options": ["levi-civita", "independent"], "resolution": null}
  ],
  "events": [{"state": "INGEST", "detail": "..."}]
}
```

### FieldDerivation

Each entry in the `derivations` array of a `derive` or `results` response is a `FieldDerivation` (`noether/orchestrator/derive.py`) serialized with `model_dump()`:

| Field | Type | Meaning |
| --- | --- | --- |
| `wrt` | `str` | The field the result varies or perturbs. |
| `kind` | `str` | `"eom"`, `"perturbation"`, or `"adm"`. |
| `capability` | `str` | The kernel capability used. |
| `result_id` | `str` | Stable id; deduped across re-runs. |
| `result_tex` | `str \| null` | The kernel-produced LaTeX, or `null` when gated. |
| `verified` | `bool` | `true` only when the kernel confirmed the result. |
| `checks` | `dict[str, str]` | Per-check name to outcome. |
| `kernel_name` | `str` | Which kernel produced the result. |
| `kernel_version` | `str` | Pinned kernel version. |
| `llm_name` | `str` | Which LLM parameterized the script. |
| `llm_version` | `str` | LLM version. |
| `script` | `str` | The Cadabra script that was run. |
| `bundle_path` | `str \| null` | Path to the provenance bundle on disk. |
| `detail` | `str` | Always non-empty: a confirmation reason when `verified`, a blocker when gated. |
| `teaching` | `str` | Prose narrating geometry tradeoffs for metric-affine derivations; reasoned, not kernel-verified; never mutates the NPR. |
| `conventions` | `dict[str, str]` | The active convention block at derivation time. |

The `detail` validator in `FieldDerivation` rejects an empty string, so a gated result (`verified: false`, `detail` naming the blocker) is always distinguishable from a verified one (`verified: true`, `detail` confirming the check).

## Related pages

- `index.md` - overview of both API surfaces and the endpoint-to-tool mapping.
- `mcp-tools.md` - the MCP tool counterparts.
- `../apps/http-server.md` - how to run the server.
- `../systems/orchestrator.md` - the session state machine these endpoints drive.
- `../features/index.md` - the `vary`, `perturb`, and `adm` tasks reachable through `/derive`.
- `../how-to-contribute/debugging.md` - common `409` and blocked-dict causes.
