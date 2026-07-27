# Data models

The pydantic models that flow through Noether, the file each lives in, and the page that explains it. This is a map; follow the links for the model semantics.

## Core NPR models

| Model | File | Purpose | Linked page |
|-------|------|---------|-------------|
| `NPR` | `noether/npr/schema.py` | The Noether Problem Representation: `npr_version`, `conventions`, `geometry`, `objects`, `action`, `task`, `ambiguities`; `is_well_posed`, `unresolved_ambiguities`, `object_named` | [../systems/npr.md](../systems/npr.md) |
| `ObjectDecl` | `noether/npr/schema.py` | A declared field or object: name, kind, role, symmetry, rank, args, `definition_tex`, `gauge_group` | [../systems/npr.md](../systems/npr.md) |
| `Geometry` | `noether/npr/schema.py` | `metric_name`, `connection_name`, `connection` | [../systems/npr.md](../systems/npr.md) |
| `ConnectionSpec` | `noether/npr/schema.py` | Connection type, torsion, nonmetricity, metric compatibility, `curvature_free`, family | [../systems/npr.md](../systems/npr.md) |
| `Action` | `noether/npr/schema.py` | `measure_tex`, `lagrangian` (an `Expr`), `lagrangian_tex` | [../systems/npr.md](../systems/npr.md) |
| `Task` | `noether/npr/schema.py` | `type` (`vary`, `reduce`, `adm`, `perturb`, `identity-check`), `with_respect_to`, `target_form` | [../systems/npr.md](../systems/npr.md) |
| `Ambiguity` | `noether/npr/schema.py` | `id`, `question`, `kind`, `options`, `resolution`; `resolved` property | [../systems/npr.md](../systems/npr.md) |

## Primitives

| Model | File | Purpose | Linked page |
|-------|------|---------|-------------|
| `Expr` AST nodes (`Num`, `Sym`, `Func`, `Tensor`, `Deriv`, `Pow`, `Prod`, `Sum`) | `noether/npr/ast.py` | The action AST | [../primitives/expression-ast.md](../primitives/expression-ast.md) |
| `Conventions` | `noether/npr/conventions.py` | The convention block carried by every kernel-boundary expression | [../primitives/conventions.md](../primitives/conventions.md) |

## Kernel layer

| Model | File | Purpose | Linked page |
|-------|------|---------|-------------|
| `ComputedResult` | `noether/kernels/base.py` | A kernel-computed expression plus its receipt; the only thing that enters a result | [../primitives/computed-result.md](../primitives/computed-result.md) |
| `KernelTask` | `noether/kernels/base.py` | A capability-tagged unit of kernel work | [../systems/index.md](../systems/index.md) |
| `KernelScript` | `noether/kernels/base.py` | `kernel_name`, `language`, `source` | [../primitives/computed-result.md](../primitives/computed-result.md) |
| `KernelRawOutput` | `noether/kernels/base.py` | `stdout`, `stderr`, `returncode`, `duration_s` | [../primitives/computed-result.md](../primitives/computed-result.md) |

## Orchestrator

| Model | File | Purpose | Linked page |
|-------|------|---------|-------------|
| `Session` | `noether/orchestrator/session.py` | `session_id`, `npr_versions`, `events`, `stale_result_ids`, `result_ids`; the state machine | [../systems/orchestrator.md](../systems/orchestrator.md) |
| `SessionState` | `noether/orchestrator/session.py` | `INGEST`, `ELICIT`, `PLAN`, `COMPUTE`, `VERIFY`, `PRESENT` | [../systems/orchestrator.md](../systems/orchestrator.md) |
| `SessionEvent` | `noether/orchestrator/session.py` | `state`, `detail` | [../systems/orchestrator.md](../systems/orchestrator.md) |
| `Plan` | `noether/orchestrator/planner.py` | `task_type`, `steps`, `verification` | [../systems/orchestrator.md](../systems/orchestrator.md) |
| `PlanStep` | `noether/orchestrator/planner.py` | `capability`, `description`, `payload` | [../systems/orchestrator.md](../systems/orchestrator.md) |
| `FieldDerivation` | `noether/orchestrator/derive.py` | `wrt`, `kind`, `capability`, `result_id`, `result_tex`, `verified`, `checks`, `kernel_name`, `kernel_version`, `llm_name`, `llm_version`, `script`, `bundle_path`, `detail` (always non-empty), `teaching`, `conventions` | [../systems/orchestrator.md](../systems/orchestrator.md) |

`FieldDerivation.detail` carries a pydantic validator that rejects an empty string, so a gated result (`verified=false`, `detail` naming the blocker) is structurally distinguishable from a verified one (`verified=true`, `detail` confirming the check).

## Verification

| Model | File | Purpose | Linked page |
|-------|------|---------|-------------|
| `CheckResult` | `noether/verify/checks.py` | `name`, `rung` (`V0`..`V4`), `passed`, `detail`, `computed_by`, `artifacts` | [../systems/verification.md](../systems/verification.md) |
| `LadderReport` | `noether/verify/ladder.py` | `results` (list of `CheckResult`); `all_passed` | [../systems/verification.md](../systems/verification.md) |

## Provenance

| Model | File | Purpose | Linked page |
|-------|------|---------|-------------|
| `ResultBundle` | `noether/provenance/bundle.py` | `session_id`, `result_id`, `result_tex`, `result_expr`, `npr_snapshot`, `plan`, `computed`, `ladder`, `narrative`, `derivations` | [../systems/provenance.md](../systems/provenance.md) |

## HTTP request models

| Model | File | Purpose |
|-------|------|---------|
| `CreateSessionRequest` | `noether/server/app.py` | `lagrangian`, `measure` (default `d^4x \sqrt{-g}`) |
| `ResolveRequest` | `noether/server/app.py` | `resolutions: dict[str, str]` (non-empty) |
| `AdoptDefinitionsRequest` | `noether/server/app.py` | `accept: list[str]` (non-empty) |
| `DeriveRequest` | `noether/server/app.py` | `with_respect_to: list[str] | None`, `kind` (`"eom"` or `"perturbation"`) |

See [../systems/index.md](../systems/index.md) for the HTTP and MCP surface contracts.

## See also

- [configuration](configuration.md) for where these models are configured.
- [dependencies](dependencies.md) for the pydantic and sympy versions that back them.
