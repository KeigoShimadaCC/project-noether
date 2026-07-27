# MCP tools

## Purpose

The stdio MCP server in `noether/mcp/server.py` exposes the orchestrator session surface as tools a host LLM can call. The host converses and plans; Noether does the kernel-backed part. This page is the contract reference: tool name, args, return shape, and refusal behavior. For how to run the server, see `../apps/mcp-server.md`.

Tool logic lives in `NoetherTools` as plain methods, unit-testable without the MCP runtime. `create_mcp_server(store=None, llm=None)` wraps a `NoetherTools` instance with `FastMCP` for stdio transport. The server ships behind the `[mcp]` extra; tests in `tests/test_mcp.py` skip cleanly when it is missing.

The server is registered with `FastMCP("noether", instructions=...)`. The instructions tell the host to ingest, relay the returned questions to its human, confirm answers with `noether_resolve` using on-menu choices only, then plan and derive. While questions are open, `noether_plan` returns `blocked=true` with the question list; the host relays it and never guesses.

## No-guessing contract over MCP

Refusals are tool results, not exceptions. A host LLM cannot make Noether guess, and the host can relay the open question list to its human verbatim.

| Situation | Tool result |
| --- | --- |
| Plan while the ledger is open | `{"blocked": true, "questions": [...]}` |
| Derive while the ledger is open | `{"blocked": true, "questions": [...]}` |
| Off-menu resolution choice | `{"error": "... is not a listed option for ...; options: [...]"}`; session unchanged |
| Unknown ambiguity id in a resolution | `{"error": "no ambiguity ..."}` |
| Empty resolutions map | `{"error": "resolutions must not be empty"}` |
| Unknown session id | `{"error": ...}` |
| Parse error on ingest | `{"error": "parse error: ..."}` |
| Unknown derivation kind | `{"error": "unknown derivation kind ..."}` |
| Undeclared `with_respect_to` field | `{"error": "... is not a declared object"}` |
| Missing cadabra or LLM backend | `{"error": "cadabra kernel not installed; cannot derive"}` or `{"error": "no agent CLI / LLM backend available ..."}` |
| Gated derivation | `{"session_id": ..., "derivations": [{..., "verified": false, "detail": "<blocker>"}]}` |

Every error or blocked dict leaves the session unmutated. A gated derivation is returned as a normal result with `verified: false` and a non-empty `detail`, not as an error.

## Tools

| Tool | Args | Returns |
| --- | --- | --- |
| `noether_kernels` | (none) | Per-kernel availability and version. |
| `noether_ingest` | `lagrangian: str`, `measure: str = r"d^4x \sqrt{-g}"` | `session_payload` for a new session, or `{"error": "parse error: ..."}`. |
| `noether_sessions` | (none) | `{"sessions": [id, ...]}`. |
| `noether_session` | `session_id: str` | `session_payload`, or `{"error": ...}` for an unknown id. |
| `noether_resolve` | `session_id: str`, `resolutions: dict[str, str]` | `session_payload` after applying on-menu choices, or `{"error": ...}`. |
| `noether_propose_definitions` | `session_id: str` | Proposed readability shorthands; `confirmed: false`. |
| `noether_adopt_definitions` | `session_id: str`, `accept: list[str]` | `session_payload` after adopting the named proposals, or `{"error": ...}`. |
| `noether_plan` | `session_id: str` | `{"blocked": false, "task_type", "steps", "verification"}` when well posed, or `{"blocked": true, "questions": [...]}`. |
| `noether_derive` | `session_id: str`, `with_respect_to: list[str] \| None = None`, `kind: str = "eom"` | `{"session_id", "derivations": [...]}`, or `{"blocked": true, ...}`, or `{"error": ...}`. |
| `noether_results` | `session_id: str` | `results_payload` with `results` and `stale_result_ids`, or `{"error": ...}`. |

There is no `noether_elicit` tool. On HTTP, `/elicit` returns unconfirmed model proposals for a human to review. On MCP the host LLM is itself the proposer: it calls `noether_session` to read the open questions and relays them.

### noether_kernels

No args. Returns:

```json
{
  "sympy": {"available": true, "version": "..."},
  "cadabra": {"available": true, "version": "..."}
}
```

`available` is `false` and `version` is `null` when a kernel binary is not installed.

### noether_ingest

Args: `lagrangian` (required), `measure` (optional, default `d^4x \sqrt{-g}`). Parses the LaTeX action into a new session and returns a `session_payload` with `state: "elicit"`, `well_posed: false`, and the open `questions`. A parse error returns `{"error": "parse error: ..."}` as a tool result, never an exception.

### noether_sessions

No args. Returns `{"sessions": ["s-...", ...]}`. The list is the same one HTTP `GET /sessions` returns, because both surfaces share one `SessionStore`.

### noether_session

Args: `session_id`. Returns the `session_payload`, or `{"error": ...}` for an unknown id.

### noether_resolve

Args: `session_id`, `resolutions` (a non-empty map of ambiguity id to choice). The tool validates each choice against the ambiguity's listed options before applying anything. An empty map, an unknown ambiguity id, or an off-menu choice returns `{"error": ...}` and does not mutate the session. On success the session is saved and the `session_payload` is returned. If results already exist, `mark_results_stale` flags them, matching the HTTP behavior.

### noether_propose_definitions

Args: `session_id`. Returns proposed readability shorthands with `confirmed: false`. Each proposal carries `id`, `symbol`, `symbol_tex`, `meaning_tex`, `definition_tex`, and `rationale`. These are notation, not results.

### noether_adopt_definitions

Args: `session_id`, `accept` (a non-empty list of proposal ids). An empty list or an unknown id returns `{"error": ...}`. On success the shorthands are added and the `session_payload` is returned. Adopting a definition never reopens a question.

### noether_plan

Args: `session_id`. Calls `Session.plan()`. While the ledger is open it returns:

```json
{"blocked": true, "questions": [...]}
```

Once well posed it returns:

```json
{
  "blocked": false,
  "task_type": "vary",
  "steps": [{"capability": "...", "description": "..."}],
  "verification": "..."
}
```

The `blocked` key is always present, so the host can branch on it directly.

### noether_derive

Args: `session_id`, `with_respect_to` (optional list of declared object names), `kind` (default `"eom"`; one of `"eom"`, `"perturbation"`, `"adm"`).

For `eom` and `perturbation`, cadabra must be installed and an LLM backend must be available; otherwise the tool returns `{"error": ...}`. For `adm`, only the SymPy component kernel is used and no model script is written. `with_respect_to` restricts the variation; an undeclared name returns `{"error": "... is not a declared object"}`.

While the ledger is open the tool returns `{"blocked": true, "questions": [...]}`. A `NotImplementedError` returns `{"error": ...}`. On success it records each result id into the session (deduped) and returns:

```json
{"session_id": "s-...", "derivations": [FieldDerivation.model_dump(), ...]}
```

Example: on a resolved Palatini session, `noether_derive` with `with_respect_to=["g", "Gamma"]` returns both the metric and the connection equations of motion. On the same session before the connection ambiguity is resolved, it returns `{"blocked": true, "questions": [...]}` naming the open connection question.

### noether_results

Args: `session_id`. Returns the `results_payload`:

```json
{
  "session_id": "s-...",
  "results": [FieldDerivation.model_dump(), ...],
  "stale_result_ids": ["r-...", ...]
}
```

The derivations are reloaded from their provenance bundles on disk, so this tool runs no physics. `stale_result_ids` names any result an assumption change has since invalidated. An unknown id returns `{"error": ...}`.

## FieldDerivation shape

Each entry in a `derivations` array is a `FieldDerivation` (`noether/orchestrator/derive.py`) serialized with `model_dump()`. The shape is identical to the HTTP surface.

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
| `teaching` | `str` | Prose narrating geometry tradeoffs for metric-affine derivations; reasoned, not kernel-verified. |
| `conventions` | `dict[str, str]` | The active convention block at derivation time. |

## Related pages

- `index.md` - overview of both API surfaces and the endpoint-to-tool mapping.
- `http-endpoints.md` - the HTTP endpoint counterparts.
- `../apps/mcp-server.md` - how to run the server with `noether mcp`.
- `../systems/orchestrator.md` - the session state machine these tools drive.
- `../features/index.md` - the `vary`, `perturb`, and `adm` tasks reachable through `noether_derive`.
- `../how-to-contribute/debugging.md` - common blocked-dict causes.
