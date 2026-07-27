# Configuration

Where Noether's configuration lives, what each knob does, and where the pins are kept.

## pyproject.toml

The package and tooling config, at the repo root.

- Package: `name = "noether"`, `version = "0.1.0"`, `requires-python = ">=3.12"`.
- Core dependencies: `pydantic>=2.7`, `sympy>=1.13`. See [dependencies](dependencies.md).
- Optional dependency groups:
  - `dev`: `pytest>=8.0`, `ruff>=0.5`.
  - `server`: `fastapi>=0.111`, `uvicorn>=0.30`, `httpx>=0.27`.
  - `mcp`: `mcp>=1.2`.
- Console script entry point: `noether = "noether.cli.main:main"`.
- Build backend: `setuptools>=68`, `setuptools.build_meta`, with `setuptools.packages.find` scoped to `noether*`.
- Ruff config:
  - `line-length = 100`, `target-version = "py312"`.
  - Lint rules selected: `E`, `F`, `I`, `UP`, `B`.
  - Per-file `E501` ignores on the frozen kernel-script and verbatim-artifact files: `noether/kernels/cadabra/templates.py`, `noether/kernels/cadabra/blocks.py`, `noether/kernels/cadabra/curvature.py`, `noether/kernels/cadabra/horndeski_g4g5.py`, and the tests that embed verbatim Cadabra strings (`tests/test_curvature.py`, `tests/test_horndeski_g4g5.py`, `tests/test_einstein_cartan.py`, `tests/test_hypermomentum.py`).
- Pytest config:
  - `testpaths = ["tests", "evals"]`.
  - Markers: `kernel_cadabra` (requires a working `cadabra2` kernel) and `slow` (long-running symbolic computation).

## Kernel version pins

`noether/kernels/versions.py` is the single source of truth for kernel pins.

- `SYMPY_PINNED = "1.14"`: the sympy series used for component verification (V0-V3).
- `CADABRA_PINNED = "2.5.15"`: the cadabra2 CLI version the audited templates target.
- `sympy_version()`: returns the installed sympy version string.
- `sympy_matches_pin()`: returns `True` when the installed sympy is the pinned major.minor series.

Bumping either pin requires re-running the full eval suite and the cadabra golden tests. Nothing else in the tree hard-codes a kernel version.

## Default convention block

`noether/npr/conventions.py` defines `NOETHER_DEFAULT_V1`, the repo-default convention block. Every expression that crosses a kernel boundary carries a convention block; no code may assume one silently. See [../primitives/conventions.md](../primitives/conventions.md) for the full field table and the metric-affine extensions.

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `NOETHER_CADABRA` | Path to the `cadabra2` binary | `cadabra2` on `PATH` |
| `NOETHER_API_URL` | Frontend proxy target for `/api/*` | `http://127.0.0.1:8754` |

No other environment variables are read. No secrets are configured or logged. See [../security.md](../security.md).

## Frontend configs

In `frontend/`:

- `next.config.mjs`: rewrites `/api/:path*` to `${NOETHER_API_URL}/:path*`. The browser only ever talks to Next, keeping the frontend same-origin and physics-free.
- `tsconfig.json`: TypeScript 5.5, React 19 JSX, strict.
- `jest.config.ts`: `ts-jest`, `jsdom` environment, `@testing-library/jest-dom`.
- `package.json` scripts: `dev`, `build`, `start`, `typecheck` (`tsc --noEmit`), `test` (jest). See [dependencies](dependencies.md) for the package versions.

## See also

- [data models](data-models.md) for the pydantic models.
- [dependencies](dependencies.md) for the package versions.
- [../how-to-contribute/patterns-and-conventions.md](../how-to-contribute/patterns-and-conventions.md) for build and lint workflow.
- [../how-to-contribute/testing.md](../how-to-contribute/testing.md) for the test markers and gates.
