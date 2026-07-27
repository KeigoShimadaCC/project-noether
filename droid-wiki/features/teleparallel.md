# Teleparallel gravity

Active contributors: KeigoShimadaCC

## Purpose

Document Noether's teleparallel `vary` capability for `f(Q)` (symmetric teleparallel) and `f(T)` (metric teleparallel), including which parts are Cadabra residue-verified and which are SymPy component cross-checked.

## How it works

```mermaid
flowchart TD
    A[Teleparallel NPR family] --> B{Family}
    B -->|symmetric-teleparallel| C[f(Q) coincident-gauge path]
    B -->|teleparallel| D[f(T) tetrad/Weitzenbock path]
    C --> E[Cadabra linear template residue check]
    D --> E
    C --> F[SymPy component cross-checks]
    D --> F
    E --> G[Verified or gated derivation]
    F --> G
```

## Key abstractions and scripts

| Item | Role | Source |
| --- | --- | --- |
| `ConnectionSpec(... family="symmetric-teleparallel", curvature_free=True, torsion=False, nonmetricity=True)` | Encodes `f(Q)` geometry class | `evals/eval_fq_symmetric_teleparallel.py`, `noether/npr/schema.py` |
| `ConnectionSpec(... family="teleparallel", curvature_free=True, torsion=True, nonmetricity=False)` | Encodes `f(T)` geometry class | `evals/eval_ft_teleparallel.py`, `noether/npr/schema.py` |
| `fq_coincident.py` | Coincident-gauge `f(Q)` formulas and boundary-identity checks | `noether/kernels/sympy_kernel/fq_coincident.py` |
| `ft_tetrad.py` | Tetrad/Weitzenbock `f(T)` formulas and geometric checks | `noether/kernels/sympy_kernel/ft_tetrad.py` |
| Cadabra linear templates | Kernel residue checks for linear identities (`f(Q)=Q`, `f(T)=T`) | `noether/kernels/cadabra/templates.py` (used via eval paths) |

## f(Q): symmetric teleparallel in coincident gauge

- Geometry: curvature-free, torsion-free, non-metric connection.
- Coincident gauge fixes `Gamma=0`, so `Q_{lambda mu nu}` is built from metric derivatives.
- Uses the boundary identity `Q = R + boundary`.
- Linear case `f(Q)=Q` reduces to the Einstein-Hilbert variation path and passes Cadabra residue checks (`eom_fq_linear_coincident` path in eval notes).
- General `f(Q)` metric-form equation is checked componentwise in SymPy on explicit coincident-gauge backgrounds.

## f(T): metric teleparallel on tetrad/Weitzenbock connection

- Geometry: curvature-free, metric-compatible, torsionful connection.
- Weitzenbock connection from tetrad:
  `Gamma^rho_{mu nu} = E_a^rho partial_mu e^a_nu`.
- Uses boundary identity `T = -R + 2 nabla_mu T^mu`.
- Linear case `f(T)=T` reduces to Einstein-equation form and passes Cadabra residue checks (`eom_ft_linear_tetrad` path in eval notes).
- General `f(T)` metric-form equation is cross-checked componentwise in SymPy; geometry checks verify `R=0`, `Q=0`, and nontrivial torsion on explicit tetrad backgrounds.

## Worked-example pointers

- `evals/eval_ft_teleparallel.py`
- `evals/eval_fq_symmetric_teleparallel.py`
- `tests/test_teleparallel_fq_ft.py`

## Honest limits

- Cadabra residue verification is currently linear-template-backed for `f(Q)=Q` and `f(T)=T`.
- General nonlinear `f(Q)` and `f(T)` behavior is validated through SymPy componentwise checks rather than a single universal closed-form Cadabra residue template.
- Results stay explicit about `verified` and `detail` so gated or partial states are never presented as complete closure.

## Integration points

- [Metric-affine geometry](./metric-affine-geometry.md)
- [Equations of motion](./equations-of-motion.md)
- [Cadabra kernel system](../systems/kernels/cadabra.md)
- [SymPy kernel system](../systems/kernels/sympy.md)
- [Verification system](../systems/verification.md)

## Entry points for modification

- Extend teleparallel NPR parsing and ambiguity handling in ingest/elicit and schema (`noether/npr/schema.py`).
- Add or refine component identities in `noether/kernels/sympy_kernel/fq_coincident.py` and `noether/kernels/sympy_kernel/ft_tetrad.py`.
- Add new Cadabra templates and eval coverage before promoting additional teleparallel closures to fully verified status.

## Key source files

| File | Why it matters |
| --- | --- |
| `noether/kernels/sympy_kernel/fq_coincident.py` | Symmetric teleparallel `f(Q)` identities, equations, and residual checks. |
| `noether/kernels/sympy_kernel/ft_tetrad.py` | Metric teleparallel `f(T)` tetrad construction and EOM checks. |
| `evals/eval_fq_symmetric_teleparallel.py` | Canonical `f(Q)` eval setup and verified-path statements. |
| `evals/eval_ft_teleparallel.py` | Canonical `f(T)` eval setup and verified-path statements. |
| `noether/npr/schema.py` | Connection family and curvature-free flags used by teleparallel setup. |
