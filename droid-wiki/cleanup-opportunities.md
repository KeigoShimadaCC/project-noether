# Cleanup opportunities

This page records where complexity concentrates in the codebase and which files are natural refactoring candidates versus which should stay as-is. The project carries zero `TODO`, `FIXME`, or `HACK` comments in `noether/` (verified via grep), so there is no deferred-work backlog to surface.

## Complexity hotspots

### Largest source files

| File | Lines | What it holds |
|------|-------|---------------|
| `noether/kernels/cadabra/templates.py` | 2718 | Frozen golden Cadabra scripts, verbatim artifacts |
| `noether/kernels/sympy_kernel/geometry.py` | 1848 | The general-connection oracle |
| `noether/orchestrator/derive.py` | 1396 | The compute beat, routing by task kind |
| `noether/kernels/sympy_kernel/adm.py` | 1291 | SymPy-verified ADM decomposition |
| `noether/kernels/cadabra/curvature.py` | 1209 | Metric-affine Cadabra primitives |
| `noether/kernels/cadabra/blocks.py` | 868 | Additive-action block assembler |
| `noether/kernels/sympy_kernel/ft_tetrad.py` | 656 | Tetrad / $f(T)$ sector |
| `noether/orchestrator/ingest.py` | 533 | LaTeX action to draft NPR |
| `noether/npr/parse.py` | 504 | LaTeX action parser |
| `noether/kernels/sympy_kernel/adapter.py` | 440 | SymPy kernel adapter |
| `noether/cli/main.py` | 420 | CLI entry points |

Line counts are from `find noether -name "*.py" | xargs wc -l | sort -rn`.

### Largest test files

| File | Lines | What it encodes |
|------|-------|-----------------|
| `tests/test_cross_flows.py` | 2863 | Cross-surface consistency (HTTP, MCP, bundle, resume) |
| `tests/test_adm_affine.py` | 1405 | Metric-affine ADM decomposition |
| `tests/test_pert_metric_affine.py` | 1232 | Metric-affine perturbation |
| `tests/test_derive.py` | 1087 | General derive path |
| `tests/test_gated_verdict_surfaces.py` | 988 | Gated vs verified verdict surfacing |
| `tests/test_geometry_inference.py` | 912 | Geometry inference contract |
| `tests/test_teaching_channel.py` | 837 | Teaching narration channel |
| `tests/test_vector_eom_affine.py` | 741 | Vector EOM on affine background |
| `tests/test_teleparallel_fq_ft.py` | 708 | Teleparallel $f(Q)$ / $f(T)$ |

Line counts are from `find tests -name "*.py" | xargs wc -l | sort -rn`. The test files are necessarily large because they encode the cross-surface contracts and the no-guessing guarantees end to end; they are not candidates for splitting.

## Refactoring guidance

`noether/orchestrator/derive.py` and `noether/kernels/sympy_kernel/geometry.py` are the natural refactoring candidates if file size becomes a maintenance burden. `derive.py` routes by task kind (`vary`, `perturb`, `adm`, plus the compositional and G4/G5 paths), so it could be split by task kind without changing behavior. `geometry.py` is the general-connection oracle and could be split by connection family.

`noether/kernels/cadabra/templates.py` and the other verbatim-artifact files (`blocks.py`, `curvature.py`, `horndeski_g4g5.py`) should stay as-is. They hold frozen Cadabra scripts that the golden tests pin byte-for-byte; splitting them would break the frozen-script contract without reducing the actual content. They are E501-exempt in `pyproject.toml` for the same reason.

See `../how-to-contribute/patterns-and-conventions.md` for the coding conventions that govern these files.
