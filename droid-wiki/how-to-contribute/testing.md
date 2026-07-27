# Testing

Tests are the acceptance gate. A capability does not exist until its eval passes end to end with checks green, and a physics-bearing change with no kernel-backed test does not merge. This page covers the frameworks, the markers, the golden tests, the eval gates, and the stub adapter that makes deterministic tests possible.

## Runner and paths

`pytest` is the runner. `pyproject.toml` sets `testpaths = ["tests", "evals"]`, so both the unit/adapter suite under `tests/` and the executable evals under `evals/` are collected by default.

```sh
.venv/bin/python -m pytest -q                # full suite
.venv/bin/python -m pytest -q tests/         # unit and adapter tests only
.venv/bin/python -m pytest -q evals/         # eval gates only
.venv/bin/python -m pytest -q -k eval1       # by keyword
```

## Markers

Defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

- `kernel_cadabra` - the test requires a working `cadabra2` kernel. These tests skip cleanly when `cadabra2` is not on `PATH`; they do not fail the suite. Run them explicitly with `.venv/bin/python -m pytest -q -m kernel_cadabra`.
- `slow` - long-running symbolic computation. Run them when you want the full sweep; CI does not gate on them by default.

Use `pytest --markers` to list them.

## Golden-output tests

Every kernel adapter operation has a golden test pinned to a kernel version. The pins live in one place, `noether/kernels/versions.py`:

- `SYMPY_PINNED = "1.14"` - SymPy major.minor for component verification (V0 through V3).
- `CADABRA_PINNED = "2.5.15"` - the `cadabra2` CLI version the audited templates in `noether/kernels/cadabra/templates.py` target.

`tests/test_versions.py` is the drift gate: `test_sympy_matches_pin` fails loudly when the installed SymPy is not the pinned `1.14.x` series, forcing a deliberate re-audit rather than a silent canon-form change. Bumping either pin is a deliberate act: re-run the full eval suite and the cadabra golden tests, confirm every check is still green, and update `versions.py` in the same commit.

The cadabra golden tests live alongside the adapter tests under `tests/` and are marked `kernel_cadabra`. They assert computed residues, not version strings; every provenance bundle records the kernel version actually used.

## Eval gates

Every eval has an executable counterpart under `evals/` plus a `test_evalN.py` gate. The eval registry (`evals/registry.py`) holds one declarative `EvalSpec` per eval: the NPR with its documented elicitation answers, the audited Cadabra templates with the kernel checks that must come back True, and the presented results with their verification ladders and component-evaluation tasks.

Run an eval end to end:

```sh
.venv/bin/python -m noether.cli.main eval1        # also eval2..eval5, eval1s, eval3s,
                                                  # eval4ma, adm-affine, vector-affine
```

Each `evalN` subcommand drives the full ingest, elicit, resolve, plan, derive, verify, present loop and writes a provenance bundle. The `test_evalN.py` gate asserts the kernel checks came back True and the presented results match. When you add a capability, you add the eval and its gate here before the capability.

## Frontend tests

The frontend uses `jest` with `ts-jest` in a `jsdom` environment, configured via `frontend/jest.config.ts` (which wraps `next/jest`).

```sh
cd frontend
npm test            # jest
npm run typecheck   # tsc --noEmit
npm run build       # next build, includes type checking
```

TypeScript is strict (`"strict": true` in `frontend/tsconfig.json`), target ES2022, module resolution `bundler`. The path alias `@/*` maps to the frontend root.

## The StubLLMAdapter

The LLM is ambient-auth and not available in CI, so deterministic tests use the `StubLLMAdapter` (`noether/llm/`). It implements the same `complete(system, prompt) -> text` interface as the real adapter but returns canned responses, so tests exercise the orchestrator, the planner, the elicit/resolve contract, and the derive path without a real model. The stub is what lets the geometry-inference, teaching-channel, and cross-flow suites run in CI.

## Contract suites

The cross-surface contracts are encoded as pytest suites under `tests/`:

- `tests/test_cross_flows.py` - metric-affine cross-surface consistency: HTTP, MCP, and bundle round-trip; stale marking on late resolution; session resume; the MCP blocked/refusal path.
- `tests/test_geometry_inference.py` - the geometry inference contract: `propose_resolutions` for geometry ambiguities, off-menu nulled, never auto-applied, `apply_resolutions` on-menu mutates and off-menu raises, HTTP `/elicit` and `/resolve` enforcement.
- `tests/test_teaching_channel.py` - the teaching narration channel: teaching field distinct from `detail`, teaching mutates no NPR, HTTP payloads expose teaching as a top-level key, verified-vs-reasoned boundary, teaching explains geometry tradeoffs, elicitation rationale preserved.
- `tests/test_dual_gate.py` - the dual gate for metric-affine results: the Cadabra residue check and the SymPy general-connection cross-check must agree before a result is called verified. This is what catches the torsion trap.

When a change touches a contract, run the corresponding suite:

```sh
.venv/bin/python -m pytest -q tests/test_cross_flows.py tests/test_geometry_inference.py \
  tests/test_teaching_channel.py tests/test_dual_gate.py
```

## Cadabra-backed tests skip if the kernel is missing

If `cadabra2` is not installed, the `kernel_cadabra` tests skip rather than fail. This keeps the default `pytest -q` run green on machines without Cadabra. To actually exercise them, install Cadabra (see [Debugging](debugging.md)) and run `.venv/bin/python -m pytest -q -m kernel_cadabra`. The CI `cadabra-golden` job does this on a runner with the kernel installed (see [Tooling](tooling.md)).
