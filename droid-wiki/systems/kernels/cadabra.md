# Cadabra2 adapter

Active contributors: KeigoShimadaCC

## Purpose

Document `noether/kernels/cadabra/` as the subprocess symbolic-derivation adapter: it executes Cadabra scripts, parses sentinel-tagged outputs into `ComputedResult`, and supports three script paths (frozen templates, compositional blocks, and generated scripts).

## Directory layout

```text
noether/kernels/cadabra/
├── __init__.py
├── adapter.py
├── templates.py
├── blocks.py
├── generate.py
├── curvature.py
└── horndeski_g4g5.py
```

## Key abstractions

| Type or constant | File | Description |
|---|---|---|
| `CadabraAdapter` | `noether/kernels/cadabra/adapter.py` | Kernel adapter implementation that runs `cadabra2` in a sandboxed subprocess. |
| `RESULT_SENTINEL` / `CHECK_SENTINEL` / `DETAIL_SENTINEL` / `CONVENTION_SENTINEL` | `noether/kernels/cadabra/adapter.py` | Output markers parsed from stdout into structured result fields. |
| `_find_executable()` | `noether/kernels/cadabra/adapter.py` | Resolves executable from `NOETHER_CADABRA` or `PATH`. |
| `_parse_sentinels()` | `noether/kernels/cadabra/adapter.py` | Treats only sentinel-marked lines as authoritative result data. |
| `_TEMPLATES`, `get()`, `register()` | `noether/kernels/cadabra/templates.py` | Registry of frozen, golden-tested Cadabra scripts (`templates.py` is 2718 lines). |
| `BlockMatch`, `Decomposition` | `noether/kernels/cadabra/blocks.py` | Block decomposition result models for additive Lagrangians. |
| `decompose_scalar()` / `decompose_metric()` | `noether/kernels/cadabra/blocks.py` | Recognize registered scalar and metric blocks in additive actions. |
| `assemble_scalar_eom_script()` / `assemble_metric_eom_script()` | `noether/kernels/cadabra/blocks.py` | Build one compositional residue-check script for the full action. |
| `generate_script()` | `noether/kernels/cadabra/generate.py` | LLM path that outputs Cadabra script text, not equations. |
| `SORTCOVDS_BLOCKER`, `attempt_g4g5_closure()` | `noether/kernels/cadabra/horndeski_g4g5.py` | Gated best-effort path for held G4/G5 terms when normal-ordering is unavailable. |

## How it works

```mermaid
flowchart TD
    T[KernelTask] --> K{payload}
    K -->|template| TP[templates.get(name)]
    K -->|script| SC[inline script text]
    TP --> X[CadabraAdapter._execute]
    SC --> X
    X --> O[cadabra2 subprocess run]
    O --> P[_parse_sentinels(stdout)]
    P --> CR[ComputedResult checks/detail/conventions]
```

## Script paths

1. **Frozen template path**
   - `task.payload["template"]` selects a registered script in `templates.py`.
   - Used for audited eval scaffolds and fixed derivation families.
2. **Compositional block path**
   - `blocks.py` decomposes additive Lagrangians into recognized blocks.
   - It assembles one script for the exact action and an independent candidate target, then residue-checks the full result in-kernel.
3. **Generated script path**
   - `generate.py` asks the LLM for a Cadabra script conforming to the adapter contract.
   - The script has no authority until the kernel check passes.

## Capabilities and selection

`CadabraAdapter.capabilities()` returns:

- `Capability.VARY`
- `Capability.IBP`
- `Capability.CANONICALIZE`
- `Capability.SUBSTITUTE`
- `Capability.INDEPENDENT_CONNECTION`

Planner and derive code select by these capability tags, not by adapter name.

## Subprocess execution and parsing details

- Executable discovery: `NOETHER_CADABRA` first, then `cadabra2` on `PATH`.
- Run model: temporary `script.cdb` + `subprocess.run(...)` with timeout and captured stdout/stderr.
- Parsed result fields come only from sentinel lines:
  - `NOETHER_RESULT: ...`
  - `NOETHER_CHECK: key=value`
  - `NOETHER_DETAIL: ...`
  - `NOETHER_CONVENTION: key=value`
- Everything else in stdout is treated as non-authoritative noise for result extraction.

## Integration points

- Core derive flow dispatches Cadabra tasks in `noether/orchestrator/derive.py`.
- Template and compositional paths are chosen there based on task kind and decomposition state.
- Verification ladder consumes Cadabra `ComputedResult` artifacts through provenance bundles.

## Entry points for modification

1. **Add a frozen script template**
   - Edit `noether/kernels/cadabra/templates.py`.
   - Ensure sentinel prints are present for result and checks.
2. **Extend compositional coverage**
   - Add block matching and script assembly in `noether/kernels/cadabra/blocks.py`.
3. **Adjust generated-script constraints**
   - Edit contracts and variation-key logic in `noether/kernels/cadabra/generate.py`.
4. **Add curvature reduction primitives**
   - Extend `noether/kernels/cadabra/curvature.py` with tested rewrite snippets.
5. **Gate or ungate held Horndeski paths**
   - Update `noether/kernels/cadabra/horndeski_g4g5.py` and blocker semantics.

## Key source files

| File | Role |
|---|---|
| `noether/kernels/cadabra/adapter.py` | Subprocess runner, capability surface, and sentinel parser. |
| `noether/kernels/cadabra/templates.py` | Frozen script registry (`2718` lines). |
| `noether/kernels/cadabra/blocks.py` | Additive block decomposition and compositional script assembly (`868` lines). |
| `noether/kernels/cadabra/generate.py` | LLM script generation contract and template selection. |
| `noether/kernels/cadabra/curvature.py` | Reusable curvature/torsion/non-metricity rewrite primitives (`1209` lines). |
| `noether/kernels/cadabra/horndeski_g4g5.py` | G4/G5 best-effort closure with `SORTCOVDS_BLOCKER`. |
| `noether/kernels/cadabra/__init__.py` | Cadabra adapter package export. |
