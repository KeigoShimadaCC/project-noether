# Computed result

Active contributors: KeigoShimadaCC

## Purpose

`noether/kernels/base.py` defines the kernel adapter contract: the capability enum, the task/script/output types, the `ComputedResult` type that carries a computed expression and its receipt, the `KernelAdapter` Protocol, and the `KernelUnavailable` error. This is the only object that can carry a computed expression into a result. A result that did not come through `ComputedResult` carries no provenance and must never reach the user (AGENTS.md rule 3).

The contract is backend-agnostic. Nothing outside a kernel adapter may import or depend on a specific CAS (AGENTS.md rule 6). Adapters implement the Protocol; the orchestrator speaks only `KernelTask` and `ComputedResult`.

## Key abstractions

| Abstraction | Kind | Role |
|---|---|---|
| `Capability` | `StrEnum` | The set of kernel operations an adapter can perform. Used to route work and to gate capabilities that need machinery not installed. |
| `KernelTask` | pydantic model | A single capability-tagged unit of kernel work: `capability`, `description`, free-form `payload`. |
| `KernelScript` | pydantic model | The script an adapter runs: `kernel_name`, `language` (`"cadabra"`, `"python-sympy"`, `"wolfram"`), `source`. |
| `KernelRawOutput` | pydantic model | The raw result of a sandboxed run: `stdout`, `stderr`, `returncode`, `duration_s`. |
| `ComputedResult` | pydantic model | A kernel-computed expression plus its receipt: `kernel_name`, `kernel_version`, `script`, `raw`, optional `expression_tex`, optional `value`, `notes`. |
| `KernelAdapter` | `Protocol` (runtime-checkable) | The contract every adapter implements: `available`, `version`, `capabilities`, `run`. |
| `KernelUnavailable` | `RuntimeError` | Raised by `run` when the backing engine is not installed. |

## The Capability enum

| Member | Value | Meaning |
|---|---|---|
| `VARY` | `vary` | Vary an action with respect to a field to get an equation of motion. |
| `IBP` | `integrate-by-parts` | Integration by parts. |
| `CANONICALIZE` | `canonicalize` | Canonicalize an expression to a target form. |
| `SUBSTITUTE` | `substitute` | Substitute a definition or value into an expression. |
| `PERTURB` | `perturb` | Expand an action to quadratic order and check the linearized EOM. |
| `ADM` | `adm` | Compute the ADM (3+1) decomposition. |
| `COMPONENT_EVAL` | `component-eval` | Evaluate components (used by the SymPy component kernel for the ADM split). |
| `INDEPENDENT_CONNECTION` | `independent-connection` | The adapter can vary an independent affine connection (metric-affine work). |

`INDEPENDENT_CONNECTION` is the gate that routes connection variation to the `vary-connection` worked example and prevents a connection field from being silently routed to the metric worked example.

## The ComputedResult type

`ComputedResult` is the receipt. Its fields:

| Field | Type | Meaning |
|---|---|---|
| `kernel_name` | `str` | Name of the adapter that produced the result. |
| `kernel_version` | `str` | Pinned version of the backing engine (see `noether/kernels/versions.py`). |
| `script` | `KernelScript` | The exact script that ran. |
| `raw` | `KernelRawOutput` | The raw stdout, stderr, returncode, and duration. |
| `expression_tex` | `str \| None` | The computed expression rendered to LaTeX, when the result carries one. |
| `value` | `Any` | Structured payload for results that are not a single expression, e.g. check verdict details. |
| `notes` | `list[str]` | Free-form notes the adapter added (e.g. which checks passed or failed). |

The bright line: only a `ComputedResult` carries a computed expression into a result bundle. The model orchestrates and writes scripts, but it cannot inject a computed expression. The verification ladder reads `notes` and `value` to decide verified versus gated.

## The KernelAdapter Protocol

```python
@runtime_checkable
class KernelAdapter(Protocol):
    name: str

    def available(self) -> bool: ...
    def version(self) -> str: ...
    def capabilities(self) -> set[Capability]: ...
    def run(self, task: KernelTask, npr: Any) -> ComputedResult: ...
```

`run` compiles a script, executes it in a sandbox, parses the output, and returns a `ComputedResult`. It raises `KernelUnavailable` if the backing engine is not installed. `npr` is passed as `Any` so the contract stays free of NPR-schema imports; adapters read what they need from it.

## How it works

1. The orchestrator builds a `KernelTask` tagged with the `Capability` it needs.
2. The planner selects an adapter whose `capabilities()` set contains that `Capability`. `INDEPENDENT_CONNECTION` is checked before routing connection variation.
3. The adapter's `run` builds a `KernelScript` (often from a frozen template or an LLM-written script), executes it as a sandboxed subprocess with timeout, and captures a `KernelRawOutput`.
4. The adapter parses the raw output into an `expression_tex` (and/or a structured `value`) and attaches any check verdicts to `notes`.
5. The `ComputedResult` flows back to the orchestrator, which wraps it in a derivation carrying the convention block and verification verdicts. The bundle writer stores it for provenance.

## Integration points

| System | How it uses the contract |
|---|---|
| Orchestrator | Builds `KernelTask`, selects an adapter by `Capability`, wraps `ComputedResult` into a derivation. |
| Kernel adapters | `noether/kernels/cadabra/` and `noether/kernels/sympy_kernel/` implement the Protocol. |
| Verification | The ladder reads `ComputedResult.notes` and `value` to decide verified versus gated. |
| Provenance | The bundle writer stores the script, raw output, kernel version, and notes. |
| MCP and HTTP surfaces | Derivations expose `verified` and a non-empty `detail` derived from the `ComputedResult`. |

## Entry points for modification

- Add a capability: add a member to `Capability`, update every adapter's `capabilities()` set, and wire the planner to route to it.
- Add a new kernel: implement the `KernelAdapter` Protocol in a new adapter module under `noether/kernels/`, pin its version in `noether/kernels/versions.py`, and add a golden-output test.
- Extend `ComputedResult`: add a field, update the bundle writer and reader together, and update the verification ladder if the field affects verdicts.
- Change routing: edit the planner. The contract itself does not change.

## Key source files

| File | Role |
|---|---|
| `noether/kernels/base.py` | The contract: `Capability`, `KernelTask`, `KernelScript`, `KernelRawOutput`, `ComputedResult`, `KernelAdapter`, `KernelUnavailable`. Authoritative source. |
| `noether/kernels/versions.py` | Pinned kernel versions referenced by `ComputedResult.kernel_version`. |
| `noether/kernels/cadabra/` | Cadabra2 adapter(s) implementing the Protocol. |
| `noether/kernels/sympy_kernel/` | SymPy adapter implementing the Protocol. |
| `noether/orchestrator/derive.py` | Builds `KernelTask` and wraps `ComputedResult` into derivations. |
| `noether/verify/` | The ladder that reads `ComputedResult` notes and value. |
