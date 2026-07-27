# Tooling

The build system, linters, code generators, and CI for Project Noether. The authoritative config files are linked inline; this page is the map.

## Build system

The package is built with `setuptools` (declared in `pyproject.toml`):

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["noether*"]
```

Install for development:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"          # core + pytest + ruff
.venv/bin/python -m pip install -e ".[dev,server]"   # add the FastAPI HTTP API
.venv/bin/python -m pip install -e ".[dev,server,mcp]"  # add the MCP stdio server
```

The optional extras are:

- `dev` - `pytest>=8.0`, `ruff>=0.5`.
- `server` - `fastapi>=0.111`, `uvicorn>=0.30`, `httpx>=0.27`.
- `mcp` - `mcp>=1.2`.

The `noether` console script entry point is `noether.cli.main:main`:

```toml
[project.scripts]
noether = "noether.cli.main:main"
```

After install, `noether ...` is equivalent to `python -m noether.cli.main ...`.

## Lint and format

`ruff` is the only linter and formatter. Config in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Run:

```sh
.venv/bin/ruff format . && .venv/bin/ruff check .
.venv/bin/ruff format --check noether tests evals   # CI mode
.venv/bin/ruff check noether tests evals            # CI mode
```

### Per-file E501 ignores

Frozen kernel-script files are verbatim artifacts; `ruff` does not reflow their lines. The following files carry `E501` (line-too-long) ignores in `[tool.ruff.lint.per-file-ignores]`:

- `noether/kernels/cadabra/templates.py` - frozen audited Cadabra templates.
- `noether/kernels/cadabra/blocks.py` - assembles Cadabra-script fragments.
- `noether/kernels/cadabra/curvature.py` - emits Cadabra substitution fragments.
- `noether/kernels/cadabra/horndeski_g4g5.py` - assembles Cadabra-script fragments.
- `tests/test_curvature.py` - embeds verbatim Cadabra declaration strings.
- `tests/test_horndeski_g4g5.py` - embeds verbatim Cadabra declaration strings.
- `tests/test_einstein_cartan.py` - embeds Cadabra scripts.
- `tests/test_hypermomentum.py` - embeds Cadabra scripts.

Do not reflow these files to satisfy `E501`; they are pinned artifacts. If you edit one, you are changing a frozen kernel script and must re-run the relevant golden tests.

## Kernel version pins

Kernel versions live in exactly one place: `noether/kernels/versions.py`.

- `SYMPY_PINNED = "1.14"` - SymPy major.minor for component verification.
- `CADABRA_PINNED = "2.5.15"` - the `cadabra2` CLI version the audited templates target.

`sympy_matches_pin()` returns True when the installed SymPy is the pinned `1.14.x` series. `tests/test_versions.py` gates drift. Bumping a pin is a deliberate act: re-run the full eval suite and the cadabra golden tests, confirm every check is green, and update `versions.py` in the same commit. Nothing else in the tree should hard-code a kernel version.

## Frontend tooling

The frontend is Next.js 15 with React 19 and TypeScript 5.5, configured in `frontend/`. Scripts from `frontend/package.json`:

- `npm run dev` - `next dev` (needs `noether serve` running; `/api/*` is proxied).
- `npm run build` - `next build`, includes type checking.
- `npm run start` - `next start`.
- `npm run typecheck` - `tsc --noEmit`.
- `npm test` - `jest` (ts-jest, jsdom env, via `frontend/jest.config.ts`).

TypeScript is strict (`"strict": true`), target ES2022, module resolution `bundler`, path alias `@/*` to the frontend root. KaTeX is the math renderer.

## Code generators

There is no separate code-generation step. The closest things are:

- `noether/kernels/cadabra/generate.py` - the model-written-script path: the LLM writes a Cadabra script that the kernel residue-checks. The script is tagged "generated (parameterized; unverified until the ladder confirms it)" in the adapter; the `verified` flag is set by the kernel's own check, never by the model.
- `noether/kernels/cadabra/blocks.py` - the compositional path: when an additive Lagrangian decomposes fully into registered building blocks, `derive_field` assembles one script for the real action plus an independent candidate from the same blocks, and the kernel residue-checks it. No model round-trip.
- `noether/kernels/cadabra/templates.py` - the frozen audited templates the golden tests pin against.

## CI

CI lives in `.github/workflows/ci.yml` and runs on push and pull request to `main`. Three jobs:

### lint-and-test

Runs on `ubuntu-latest` with Python 3.12. Installs `.[dev,server,mcp]`, then:

- `ruff format --check noether tests evals`
- `ruff check noether tests evals`
- `python -m noether.cli.main kernels` (kernel availability report)
- `pytest -q`

The default runner does not have `cadabra2`, so the `kernel_cadabra` tests skip cleanly here. The SymPy kernel is always present.

### frontend

Runs on `ubuntu-latest` with Node 22, working directory `frontend`. Installs with `npm ci` and runs `npm run build` (which includes type checking).

### cadabra-golden

The Horizon 1 gate. Runs on `ubuntu-24.04` and installs `cadabra2` from the upstream prebuilt noble package (`CADABRA_DEB_VERSION = "2.5.14"`, one patch behind the local Homebrew pin `2.5.15`). The job deliberately uses noble's system Python 3.12 inside a venv rather than `actions/setup-python`, because cadabra2's embedded interpreter resolves its python prefix through the `python3` on PATH and the hostedtoolcache Python picks a prefix that lacks cadabra's `cdb` modules. Steps:

- Install `cadabra2` and confirm `cadabra2 --version`.
- Install `.[dev]` into a venv.
- `.venv/bin/python -m noether.cli.main kernels`.
- `.venv/bin/python -m pytest -q -m kernel_cadabra` (the cadabra golden derivations, evals 1 through 5).
- `.venv/bin/python -m noether.cli.main eval1 --results /tmp/results` (end to end with provenance).

The golden tests assert computed residues, not version strings, and every provenance bundle records the kernel version actually used, so this job is meaningful verification rather than a silent pin bump.

## Keeping CI green

Before pushing:

```sh
.venv/bin/ruff format --check noether tests evals
.venv/bin/ruff check noether tests evals
.venv/bin/python -m pytest -q
cd frontend && npm run build && npm test
```

If you have `cadabra2` installed locally, also run `.venv/bin/python -m pytest -q -m kernel_cadabra` and the specific `noether evalN` your change touches. See [Testing](testing.md) for the full marker reference and [Development workflow](development-workflow.md) for the PR cycle.
