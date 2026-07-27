# API

## Purpose

Project Noether exposes the same session surface through two APIs: a FastAPI HTTP server (`noether/server/app.py`, behind the `[server]` extra) and a stdio MCP server (`noether/mcp/server.py`, behind the `[mcp]` extra). Both drive one orchestrator and one session store, so a session created through one surface is visible and continuable through the other. These pages are the contract reference for each surface; the operational "how to run" pages live under `../apps/`.

## Shared backbone

Both APIs are thin transport layers over the same deterministic control plane in `noether/orchestrator/`:

- `SessionStore` (`noether/orchestrator/store.py`) is the single JSON-backed persistence layer. `DEFAULT_STORE` is the same path for both surfaces, so session ids round-trip between HTTP, MCP, and the CLI.
- `Session` (`noether/orchestrator/session.py`) is the state machine both surfaces mutate. Ingest, elicit, resolve, plan, derive, and results all call the same `Session` methods.
- `session_payload` and `results_payload` (`noether/orchestrator/view.py`) are the one shared response shape. HTTP returns them as JSON bodies; MCP returns them as tool results. The CLI reuses the same payload construction.
- Derivation results are `FieldDerivation` objects (`noether/orchestrator/derive.py`) serialized with `model_dump()`. The shape is identical on both surfaces.

Neither surface invents physics. A result that did not come through a kernel carries no provenance and never reaches the user.

## No-guessing contract, identical on both

The contract is enforced in the orchestrator, so it is the same on both surfaces. The only difference is how a refusal is delivered:

| Situation | HTTP | MCP |
| --- | --- | --- |
| Plan while the ambiguity ledger is open | `409` with `{"blocked": true, "questions": [...]}` | Tool result `{"blocked": true, "questions": [...]}` |
| Off-menu resolution choice | `400` with a `ValueError` message; session unchanged | Tool result `{"error": ...}`; session unchanged |
| Unknown ambiguity id in a resolution | `404` | Tool result `{"error": ...}` |
| Derive while the ledger is open | `409` with `{"blocked": true, "questions": [...]}` | Tool result `{"blocked": true, "questions": [...]}` |
| Gated derivation (kernel did not confirm) | `200` with `verified: false` and a non-empty `detail` naming the blocker | Tool result with `verified: false` and a non-empty `detail` |
| Unknown session id | `404` | Tool result `{"error": ...}` |
| Parse error on ingest | `422` | Tool result `{"error": "parse error: ..."}` |

HTTP raises exceptions; MCP returns error/blocked dicts. Both leave the session unmutated on a refusal. A host LLM cannot make Noether guess, and an HTTP client cannot either.

## Endpoint to tool mapping

Every HTTP endpoint has an MCP tool counterpart that performs the same orchestrator call and returns the same payload shape.

| HTTP | MCP tool | Purpose |
| --- | --- | --- |
| `GET /health` | `noether_kernels` | Report kernel availability and version. |
| `GET /sessions` | `noether_sessions` | List stored session ids. |
| `POST /sessions` | `noether_ingest` | Parse a LaTeX action into a new session; returns open questions. |
| `GET /sessions/{id}` | `noether_session` | Show session state, objects, questions, events. |
| `POST /sessions/{id}/elicit` | (none) | Unconfirmed model proposals. MCP has no elicit tool: the host LLM is the proposer, so it does not call out for proposals. |
| `POST /sessions/{id}/resolve` | `noether_resolve` | Record human-confirmed, on-menu resolutions. |
| `GET /sessions/{id}/definitions` | `noether_propose_definitions` | Propose readability shorthands. |
| `POST /sessions/{id}/definitions` | `noether_adopt_definitions` | Adopt selected shorthands by proposal id. |
| `GET /sessions/{id}/plan` | `noether_plan` | Build the derivation plan; blocked while the ledger is open. |
| `POST /sessions/{id}/derive` | `noether_derive` | Run a kernel-checked derivation (`kind=eom\|perturbation\|adm`). |
| `GET /sessions/{id}/results` | `noether_results` | Reload the full derivation history from provenance bundles. |

The one asymmetry is `/elicit`. On HTTP it is a distinct endpoint that returns unconfirmed model proposals for a human to review. On MCP the host LLM is itself the proposer, so there is no `noether_elicit` tool; the host calls `noether_session` to read the open questions and relays them.

## Pages in this section

- `http-endpoints.md` - the FastAPI HTTP session API, endpoint by endpoint, with request and response models.
- `mcp-tools.md` - the stdio MCP server, tool by tool, with args and refusal behavior.

## Related pages

- `../apps/http-server.md` - how to run the HTTP server (`noether serve`, default `127.0.0.1:8754`).
- `../apps/mcp-server.md` - how to run the MCP server (`noether mcp`).
- `../systems/orchestrator.md` - the session state machine both APIs drive.
- `../features/index.md` - the tasks reachable through `derive`: `vary`, `perturb`, `adm`.
- `../how-to-contribute/debugging.md` - common causes of `409` and blocked dicts.
