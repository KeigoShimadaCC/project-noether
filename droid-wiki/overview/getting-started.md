# Getting started

This page covers installing Noether, running the test suite, and driving the tool through its four surfaces.

## Prerequisites

- Python 3.12 or newer.
- Optionally Cadabra2 for the CAS-backed derivations. The Cadabra-backed tests and evals skip cleanly when the kernel is absent; SymPy ships with the core package and covers the component checks.
- Optionally Node.js and npm for the web client.

## Install

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"        # add [server] and/or [mcp]
brew tap kpeeters/repo && brew install cadabra2     # official macOS channel
```

The optional dependency groups are defined in `pyproject.toml`:

- `dev` - `pytest` and `ruff`
- `server` - `fastapi`, `uvicorn`, `httpx` (the HTTP session API)
- `mcp` - the `mcp` package (the stdio server)

Cadabra2 is driven as a sandboxed subprocess (`cadabra2` CLI). Set `NOETHER_CADABRA` to point at a non-default binary.

## The LLM backend

The LLM backend is ambient-auth, no API key. It auto-detects an installed agent CLI (`codex`, `claude`, `gemini`, or `droid`) and runs it one-shot as a sandboxed subprocess, so credentials stay in that CLI's own login session. A deterministic stub backs the tests. See [LLM adapters](../systems/llm.md).

## Run the tests

```sh
.venv/bin/python -m pytest -q                       # full suite; cadabra tests skip if absent
.venv/bin/ruff format . && .venv/bin/ruff check .   # format + lint
```

Test discovery covers both `tests/` and `evals/` (see `pyproject.toml` `[tool.pytest.ini_options]`). Two markers exist: `kernel_cadabra` (requires a working Cadabra kernel) and `slow` (long-running symbolic computation). See [Testing](../how-to-contribute/testing.md).

## Drive it from the CLI

```sh
# List kernel adapters and availability
.venv/bin/python -m noether.cli.main kernels

# Parse an action into a draft NPR and see the questions that block planning
.venv/bin/python -m noether.cli.main ingest "-\tfrac14 F_{\mu\nu} F^{\mu\nu}"

# Ingest, then let a detected agent CLI propose answers (unconfirmed by default)
.venv/bin/python -m noether.cli.main elicit "K(\phi,X) + F(\phi) R" --accept-llm

# Run an eval end to end (walking skeleton, with a provenance bundle)
.venv/bin/python -m noether.cli.main eval1     # also eval2..eval5, eval1s, eval3s, eval4ma

# Conversational loop: ingest, clarify, plan (resumable)
.venv/bin/python -m noether.cli.main chat
.venv/bin/python -m noether.cli.main sessions
.venv/bin/python -m noether.cli.main resume <session-id>
```

After installing the package, the `noether` console script is equivalent to `python -m noether.cli.main`. See [CLI](../apps/cli.md).

## Run the HTTP session API

```sh
.venv/bin/python -m noether.cli.main serve         # FastAPI on 127.0.0.1:8754
```

`POST /sessions` ingests an action; `POST /sessions/{id}/resolve` mutates it against listed options; `GET /sessions/{id}/plan` returns 409 until the problem is well posed; `POST /sessions/{id}/derive` runs a derivation (`kind` = eom | perturbation | adm); `GET /sessions/{id}/results` reloads history. See [HTTP server](../apps/http-server.md).

## Run the MCP server

```sh
.venv/bin/python -m noether.cli.main mcp           # stdio; requires [mcp] extra
```

Exposes the same session surface as tools (`noether_ingest`, `noether_resolve`, `noether_plan`, `noether_derive`, `noether_results`). See [MCP server](../apps/mcp-server.md).

## Run the web client

```sh
cd frontend && npm install && npm run dev          # needs `noether serve` running
```

Next.js App Router + KaTeX. The browser talks only to Next; `/api/*` is proxied to the FastAPI server (`NOETHER_API_URL`, default `http://127.0.0.1:8754`). See [Web frontend](../apps/web-frontend.md).

## Where to go next

- New to the domain vocabulary? Read the [Glossary](glossary.md).
- Want to change code? Read [How to contribute](../how-to-contribute/index.md) and [Patterns and conventions](../how-to-contribute/patterns-and-conventions.md).
- Want to understand a computation path? Start at [Features](../features/index.md).
