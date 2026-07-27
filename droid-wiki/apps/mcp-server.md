# MCP server

Active contributors: KeigoShimadaCC

## Purpose

The MCP server exposes the orchestrator session surface as a set of stdio tools that a host LLM can call. The host converses and plans; Noether does the kernel-backed part. The no-guessing contract survives the protocol: refusals are tool results, not exceptions, so a host LLM cannot make Noether guess and can relay the open question list to its human. Tool logic lives in `NoetherTools` as plain methods, unit-testable without the MCP runtime; `create_mcp_server` wraps it for stdio transport.

## Directory layout

```
noether/mcp/
  server.py   NoetherTools + create_mcp_server (FastMCP wrapper)
  __init__.py exports NoetherTools, create_mcp_server
```

The server lives behind the `[mcp]` extra. Tests in `tests/test_mcp.py` skip cleanly when the extra is missing; they exercise `NoetherTools` directly and check the FastMCP wrapper for the expected tool registry.

## Tools

| Tool | NoetherTools method | Returns |
|---|---|---|
| `noether_kernels` | `kernels` | Per-kernel availability and version. |
| `noether_ingest` | `ingest(lagrangian, measure)` | `session_payload` for a new session, or `{"error": "parse error: ..."}`. |
| `noether_sessions` | `sessions` | `{"sessions": [id, ...]}`. |
| `noether_session` | `session(session_id)` | `session_payload`, or `{"error": ...}` for an unknown id. |
| `noether_resolve` | `resolve(session_id, resolutions)` | `session_payload` after applying on-menu choices. `{"error": ...}` for an unknown id, an empty resolutions map, an unknown ambiguity id, or an off-menu choice. Marks prior results stale. |
| `noether_propose_definitions` | `propose_definitions(session_id)` | Proposed readability shorthands; `confirmed: false`. |
| `noether_adopt_definitions` | `adopt_definitions(session_id, accept)` | `session_payload` after adopting the named proposals. |
| `noether_plan` | `plan(session_id)` | `{"blocked": false, "task_type", "steps", "verification"}` when well posed, or `{"blocked": true, "questions": [...]}` while the ledger is open. |
| `noether_derive` | `derive(session_id, with_respect_to, kind)` | `{"session_id", "derivations": [...]}`. `kind="eom"\|"perturbation"\|"adm"`. `{"blocked": true, "questions": [...]}` while blocked; `{"error": ...}` for an unknown kind, unknown id, undeclared `with_respect_to` field, missing cadabra/LLM, or `NotImplementedError`. |
| `noether_results` | `results(session_id)` | `results_payload`: derivations reloaded from their provenance bundles plus `stale_result_ids`. |

## How it works

```mermaid
flowchart TD
    Host["Host LLM"] --> Ingest["noether_ingest"]
    Ingest --> Questions["session_payload\nwith open questions"]
    Questions --> Relay["Host relays questions\nto its human"]
    Relay --> Resolve["noether_resolve\n(on-menu only)"]
    Resolve --> Check{"off-menu?"}
    Check -- yes --> ErrData['{"error": ...}\nsession unchanged']
    Check -- no --> Save["store.save"]
    Save --> Plan["noether_plan"]
    Plan --> Blocked{"blocked?"}
    Blocked -- yes --> BlockedData['{"blocked": true,\n "questions": ...}']
    BlockedData --> Relay
    Blocked -- no --> Derive["noether_derive\nkind=eom|perturbation|adm"]
    Derive --> AdmPath{kind}
    AdmPath -- adm --> SymPy["derive_adm\n(SymPy only)"]
    AdmPath -- eom/perturbation --> Cadabra["derive_eom /\nderive_perturbation\n(cadabra + LLM)"]
    SymPy --> Results["noether_results"]
    Cadabra --> Results
    Results --> Host
```

`NoetherTools(store, llm, results_root)` holds one store, the same `SessionStore` the HTTP API and CLI use. `_llm()` returns the injected adapter or falls back to `CliLLMAdapter`. `_record` dedups result ids into the session and saves, matching the HTTP `_record` helper. The ADM path skips cadabra and the LLM because `derive_adm` writes no model script; it uses only the SymPy component kernel. The eom and perturbation paths require both cadabra and an LLM backend; either missing returns `{"error": ...}`.

`create_mcp_server(store, llm)` constructs a `FastMCP` instance named `noether` with instructions that tell the host to ingest, show the questions to its human, confirm with `noether_resolve`, then `noether_plan`, and to relay the blocked list rather than guess. Each `@server.tool()` wrapper delegates to the corresponding `NoetherTools` method.

## The no-guessing contract over MCP

Refusals are data. A host LLM receives a tool result it can read and relay, never an exception it might swallow.

| Situation | Tool result |
|---|---|
| Open ledger on `noether_plan` | `{"blocked": true, "questions": ["amb-...", ...]}` |
| Open ledger on `noether_derive` | `{"blocked": true, "questions": [...]}` |
| Off-menu choice on `noether_resolve` | `{"error": "<choice> is not a listed option for <id>; options: [...]"}`; session unchanged |
| Unknown session id | `{"error": ...}` from `session`, `plan`, `resolve`, `derive`, `results` |
| Unknown derivation kind | `{"error": "unknown derivation kind ..."}` |
| Undeclared `with_respect_to` field | `{"error": "<field> is not a declared object"}` |
| Missing cadabra or LLM | `{"error": "cadabra kernel not installed; cannot derive"}` or `{"error": "no agent CLI / LLM backend available ..."}` |

Derivations carry `verified` and a non-empty `detail` matching the HTTP surface. A verified result has `verified: true` and a `detail` that confirms the check; a gated result has `verified: false` and a `detail` naming the blocker. The two are distinguishable through both fields on both surfaces.

## The Palatini dual-derivation example

`noether_derive` with `with_respect_to=['g', 'Gamma']` returns both equations of motion on a resolved Palatini session: the metric EOM and the connection EOM. When the connection ambiguity is still open, the same call returns `{"blocked": true, "questions": [...]}` naming the open question, not a partial or guessed result. This is the MCP analogue of the HTTP `409` and the CLI's refusal to print a plan.

## Integration points

- **Session store.** `NoetherTools.store` is a `SessionStore`; the same store the HTTP API and CLI use. A session created over MCP is visible to `noether resume <id>` and to `GET /sessions`.
- **View layer.** `session_payload` and `results_payload` are imported from `noether/orchestrator/view.py`, so the JSON shape is identical to the HTTP server.
- **Derive functions.** `derive_eom`, `derive_perturbation`, `derive_adm` are the same functions the HTTP server calls; the only difference is how a refusal is shaped (`{"error": ...}` vs `HTTPException`).
- **Host LLM.** The host drives elicitation; `noether_elicit` is not a tool because the host is itself the proposer. The host must relay the question list from `noether_ingest` and `noether_plan` to its human and call `noether_resolve` only with confirmed on-menu answers.

## Running it

```sh
pip install -e ".[mcp]"
noether mcp                      # stdio; default store
noether mcp --store /path/to/sessions
```

`cmd_mcp` in `noether/cli/main.py` constructs the server with `create_mcp_server(store=store)` and calls `.run()`. Point an MCP-capable host at the `noether mcp` command.

## Entry points for modification

- **Add a tool.** Add a method to `NoetherTools` in `noether/mcp/server.py` and a `@server.tool()` wrapper in `create_mcp_server`. Return dicts; never raise for expected refusals.
- **Change a refusal shape.** Keep the HTTP and MCP shapes consistent: the blocked plan payload (`{"blocked": true, "questions": [...]}`) and the off-menu error string must match across surfaces.
- **Change validation.** `resolve` validates against the ambiguity options inline before calling `session.resolve`; keep this in sync with `session.confirm_resolutions` used by the HTTP server.

## Key source files

| File | Role |
|---|---|
| `noether/mcp/server.py` | `NoetherTools` and `create_mcp_server` |
| `noether/mcp/__init__.py` | Exports `NoetherTools`, `create_mcp_server` |
| `noether/orchestrator/view.py` | Shared `session_payload`, `results_payload` |
| `tests/test_mcp.py` | `NoetherTools` method tests plus tool-registry checks |
| `noether/cli/main.py` | `cmd_mcp` launcher |
