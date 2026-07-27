# Dependencies

What Noether depends on, where the pins live, and what floats.

## Python core

From `pyproject.toml` `[project].dependencies`:

| Package | Constraint | Notes |
|---------|-----------|-------|
| `pydantic` | `>=2.7` | Floating floor. Backs every data model. |
| `sympy` | `>=1.13` | Floating floor, but the verification ladder targets the pinned series. |

The pinned sympy series is `1.14` (see [configuration](configuration.md) and `noether/kernels/versions.py`). A sympy outside the pinned series still runs, but `sympy_matches_pin()` returns `False` and the golden tests treat that as a drift signal.

## Optional dependency groups

From `pyproject.toml` `[project.optional-dependencies]`:

| Group | Packages | Constraint |
|-------|----------|-----------|
| `dev` | `pytest`, `ruff` | `>=8.0`, `>=0.5` |
| `server` | `fastapi`, `uvicorn`, `httpx` | `>=0.111`, `>=0.30`, `>=0.27` |
| `mcp` | `mcp` | `>=1.2` |

All floors, no upper bounds. Install with `pip install -e ".[dev,server,mcp]"`.

## External tool: Cadabra2

Cadabra2 is not a Python package. It is an external CAS installed via Homebrew (`brew tap kpeeters/repo && brew install cadabra2`) or pointed at via `NOETHER_CADABRA`. It is GPL-3.0 and invoked as a subprocess, not linked. The pinned version is `2.5.15` (`noether/kernels/versions.py`). See [../security.md](../security.md) for the licensing and sandboxing rationale.

## LLM: ambient-auth runtime

The LLM is not a package dependency. `noether/llm/cli.py` detects an installed agent CLI (`codex`, `claude`, `gemini`, `droid`) on `PATH` at runtime and runs it as a subprocess. Credentials stay in that CLI's own login session. See [../systems/llm.md](../systems/llm.md).

## Frontend

From `frontend/package.json`:

| Package | Constraint | Role |
|---------|-----------|------|
| `next` | `^15.3.0` | Framework |
| `react` | `^19.0.0` | UI |
| `react-dom` | `^19.0.0` | DOM renderer |
| `katex` | `^0.16.21` | LaTeX math rendering |

Dev dependencies:

| Package | Constraint | Role |
|---------|-----------|------|
| `typescript` | `^5.5.0` | Type checking |
| `jest` | `^30.4.2` | Test runner |
| `jest-environment-jsdom` | `^30.4.1` | DOM test environment |
| `ts-jest` | `^29.4.11` | TypeScript jest transform |
| `@testing-library/react` | `^16.3.2` | Component testing |
| `@testing-library/jest-dom` | `^6.9.1` | DOM matchers |
| `@types/node` | `^22.0.0` | Node type defs |
| `@types/react`, `@types/react-dom` | `^19.0.0` | React type defs |
| `@types/katex` | `^0.16.7` | KaTeX type defs |
| `@types/jest` | `^30.0.0` | Jest type defs |

Caret ranges; float within the stated major. Locked via `frontend/package-lock.json` and installed with `npm ci` in CI.

## Roadmap: xAct

xAct (wolframscript) is on the roadmap as a cross-check kernel but is not yet a dependency. No code in `noether/` imports it. See [../overview/architecture.md](../overview/architecture.md).

## Pin summary

| What | Pin | Where | Pinned vs floating |
|------|-----|-------|--------------------|
| sympy series | `1.14` | `noether/kernels/versions.py` | Pinned (major.minor) |
| cadabra2 | `2.5.15` | `noether/kernels/versions.py` | Pinned |
| Python deps | floors | `pyproject.toml` | Floating floor |
| Frontend deps | caret | `frontend/package.json` | Floating within major |

See [configuration](configuration.md) for how these pins are enforced.
