# Deployment

Noether is a research prototype with no production deployment pipeline. There is no container image, no environment hierarchy beyond local, and no release process. This page documents what does exist: local install, the two servers, the web client, and the CI workflow that runs on every push.

## Local install

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,server,mcp]"
```

The dependency groups (`dev`, `server`, `mcp`) are defined in `pyproject.toml`. See `../reference/configuration.md` for the full breakdown.

## Cadabra2

Cadabra2 is an external CAS, invoked as a subprocess. Install it via Homebrew:

```sh
brew tap kpeeters/repo && brew install cadabra2
```

Or point Noether at a non-default binary with the `NOETHER_CADABRA` environment variable. The pinned version is `2.5.15` (see `noether/kernels/versions.py`). Tests that require cadabra2 carry the `kernel_cadabra` marker and skip cleanly when the binary is absent.

## Run the HTTP server

```sh
noether serve
```

Binds `127.0.0.1:8754` via uvicorn (loopback only). See `../systems/index.md` for the API surface.

## Run the MCP server

```sh
noether mcp
```

Speaks stdio. Intended for an MCP-capable host (a CLI, an editor, an agent runtime).

## Run the web client

```sh
cd frontend && npm install && npm run dev
```

Next.js dev server. `/api/*` is proxied to the FastAPI session server at `NOETHER_API_URL` (default `http://127.0.0.1:8754`) via `frontend/next.config.mjs` rewrites. The browser holds no physics state.

## CI

There is one workflow, `.github/workflows/ci.yml`, with three jobs:

- `lint-and-test`: ruff format check, ruff lint, pytest on `ubuntu-latest`. Cadabra tests skip because the runner lacks the binary.
- `frontend`: `npm ci`, `npm run build` (includes type checking) on Node 22.
- `cadabra-golden`: the Horizon 1 gate. Installs the upstream cadabra2 deb (`2.5.14`, one patch behind the local pin) on `ubuntu-24.04`, runs the cadabra golden derivations (`evals 1-5`), and runs `eval1` end to end with provenance. The golden tests assert computed residues, not version strings, so this job is meaningful verification rather than a silent pin bump.

## Kernel version pins

The closest thing to a deployment manifest is `noether/kernels/versions.py`:

- `SYMPY_PINNED = "1.14"`
- `CADABRA_PINNED = "2.5.15"`

Bumping either pin is a deliberate act: re-run the full eval suite and the cadabra golden tests, confirm every check is green, and update the file in the same commit. See `../reference/configuration.md`.

## Research-prototype status

No environments beyond local. No secrets management (there are no secrets; see `../security.md`). No telemetry. The HTTP server binds loopback and is not intended to be exposed to a network. See `../security.md` for the trust boundaries this design enforces.
