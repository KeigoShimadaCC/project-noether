# Perturbation

Active contributors: KeigoShimadaCC

## Purpose

Document the `perturb` task that expands an action to quadratic order around a background and checks the linearized equation of motion. The path uses frozen, audited Cadabra scaffolds per field class. The kernel's own residue and linearized-EOM-match checks decide `verified`; the model only writes the script that parameterizes a scaffold.

## How it works

```mermaid
flowchart TD
    A["Well-posed NPR, kind=perturbation"] --> B["derive_perturbation selects fields"]
    B --> C{"Field kind + geometry"}
    C -->|scalar, no X-coupling| D["pert_scalar_quadratic"]
    C -->|scalar, K(phi,X) X-coupling| E["pert_kessence_quadratic"]
    C -->|metric, Levi-Civita| F["pert_metric_quadratic"]
    C -->|metric, independent connection| G["pert_metric_affine_quadratic"]
    C -->|rank-1 gauge, abelian, Levi-Civita| H["pert_gauge_quadratic"]
    C -->|rank-1 gauge, non-abelian, Levi-Civita| I["pert_yang_mills_quadratic"]
    C -->|rank-1 gauge, independent, F=dA| J["pert_vector_affine_dA_quadratic"]
    C -->|rank-1 gauge, independent, F=nabla A| K["pert_vector_affine_covcurl_quadratic"]
    C -->|other kind| R["Refuse: NotImplementedError"]
    D --> L["Cadabra run + residue_zero + linearized_eom_match"]
    E --> L
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
    L --> M["verified from kernel checks; detail non-empty"]
```

`derive_perturbation` selects the supported dynamical fields, then `derive_field` (with `kind="perturbation"`) routes each to a scaffold via `_variation_key` in `noether/kernels/cadabra/generate.py`. The model writes a Cadabra script that expands the action to second order using `keep_weight` projection, derives the linearized EOM, and states an independent candidate operator; the kernel's `residue_zero` and `linearized_eom_match` checks decide the verdict. Both checks must pass before a result is called verified.

## Scaffolds

| Scaffold | Field class | Geometry | Eval | Notes |
| --- | --- | --- | --- | --- |
| `pert_scalar_quadratic` | scalar field | any (Levi-Civita default) | `evals/eval3p_scalar_perturbation.py` | Quadratic action and linearized Klein-Gordon operator; mass `m^2 = V''(\bar\phi)` |
| `pert_kessence_quadratic` | scalar with `K(phi, X)` X-coupling | any | `evals/eval3k_kessence_perturbation.py` | Expands `X` to its primitive in the kernel; surfaces sound-speed kinetic mixing |
| `pert_metric_quadratic` | metric (graviton) | Levi-Civita | `evals/eval3g_graviton_perturbation.py` | Quadratic Einstein-Hilbert action and linearized Einstein operator |
| `pert_metric_affine_quadratic` | metric on independent connection | metric-affine | `evals/eval4ma_metric_affine_perturbation.py` | Includes connection fluctuation `dG` alongside metric fluctuation `h` |
| `pert_gauge_quadratic` | rank-1 gauge, abelian (`U(1)`/none) | Levi-Civita | `evals/eval3a_maxwell_perturbation.py` | Maxwell quadratic action and linearized wave operator |
| `pert_yang_mills_quadratic` | rank-1 gauge, non-abelian (`SU(N)` etc.) | Levi-Civita | `evals/eval3y_yang_mills_perturbation.py` | Selected by the `gauge_group` marker on the object |
| `pert_vector_affine_dA_quadratic` | rank-1 gauge, independent connection, `F = dA` | metric-affine | `evals/eval_vector_affine.py` | Both checks pass; no connection fluctuation in the result |
| `pert_vector_affine_covcurl_quadratic` | rank-1 gauge, independent connection, `F = \nabla A` | metric-affine | `evals/eval_vector_affine.py` | Residue gated due to Kronecker-delta limitation with mixed-index `dG` objects; SymPy cross-check provides independent verification |

## Metric-affine perturbations

On a metric-affine (independent-connection) background the metric perturbation routes to `pert_metric_affine_quadratic`, which includes the connection fluctuation `dG` alongside the metric fluctuation `h`. The connection object itself is not perturbed independently: there is no separate connection perturbation scaffold, so the connection is excluded from the default perturbation field list. Requesting it explicitly raises `NotImplementedError` naming the field (HTTP 422), rather than guessing a perturbation.

A rank-1 gauge potential on a metric-affine background routes to a torsion-aware vector-affine scaffold, selected by the elicited `field_strength_definition` convention:

- `exterior-derivative` (`F = dA`) routes to `pert_vector_affine_dA_quadratic`. Both kernel checks pass; no connection fluctuation appears in the result.
- `covariant-curl` (`F = \nabla A`) routes to `pert_vector_affine_covcurl_quadratic`. The Cadabra residue check is gated because of a Kronecker-delta limitation with mixed-index `dG` objects; a SymPy component cross-check provides independent verification. The covcurl action retains `a*dG` cross terms (VAL-PERT-018).

The two choices differ by torsion-dependent terms (VAL-PERT-017): the covariant-curl field strength carries `T^\lambda{}_{\mu\nu} A_\lambda` that the exterior-derivative form does not.

## Key abstractions and scripts

| Item | Role | Source |
| --- | --- | --- |
| `derive_perturbation` | Select supported fields and dispatch to `derive_field` with `kind="perturbation"` | `noether/orchestrator/derive.py` |
| `derive_field` | Run the scaffold-parameterized script and set `verified` from kernel checks | `noether/orchestrator/derive.py` |
| `_variation_key` | Pick the worked-example scaffold for the field kind and geometry | `noether/kernels/cadabra/generate.py` |
| `_has_x_coupling` | Detect a `K(phi, X)` coupling that needs the k-essence scaffold | `noether/kernels/cadabra/generate.py` |
| `_is_non_abelian` | Select Yang-Mills vs Maxwell from the `gauge_group` marker | `noether/kernels/cadabra/generate.py` |
| `PERTURBATION_CONTRACT` | Cadabra script generation contract for quadratic expansion and `keep_weight` projection | `noether/kernels/cadabra/generate.py` |
| `Capability.PERTURB` | Capability tag for perturbation tasks | `noether/kernels/base.py` |

## Worked-example pointers

- `evals/eval3p_scalar_perturbation.py` (scalar quadratic action).
- `evals/eval3k_kessence_perturbation.py` (k-essence `X` expansion).
- `evals/eval3g_graviton_perturbation.py` (graviton).
- `evals/eval3a_maxwell_perturbation.py` (Maxwell).
- `evals/eval3y_yang_mills_perturbation.py` (Yang-Mills).
- `evals/eval4ma_metric_affine_perturbation.py` (metric-affine metric perturbation with `dG`).
- `evals/eval_vector_affine.py` (`F = dA` vs `F = \nabla A` on torsionful background).

## Honest limits

- Other field kinds (rank-2 field strength, independent connection perturbation, tetrad perturbation) are refused with `NotImplementedError` naming the field rather than guessed. The connection object on a metric-affine background is excluded from the default perturbation field list because it is not perturbed independently; requesting it raises `NotImplementedError` (HTTP 422).
- The `pert_vector_affine_covcurl_quadratic` residue check is gated by a Kronecker-delta limitation with mixed-index `dG` objects. The result is surfaced as gated with non-empty `detail` naming the blocker, and a SymPy cross-check provides independent verification so the physics is not silently trusted.
- A result is called verified only when both `residue_zero` and `linearized_eom_match` pass. A failed check returns `verified=false` with non-empty `detail`, never a fabricated success.

## Integration points

- [Equations of motion](./equations-of-motion.md) (shares `derive_field` and the kernel residue-check semantics).
- [Metric-affine geometry](./metric-affine-geometry.md) (routes metric and vector perturbations to affine scaffolds).
- [Teaching channel](./teaching-channel.md) (perturbation teaching narrates `h`/`dG` cross-terms on metric-affine backgrounds).
- [Orchestrator system](../systems/orchestrator.md) (perturbation lives here).
- [Cadabra kernel system](../systems/kernels/cadabra.md) (runs the scaffolds).
- [SymPy kernel system](../systems/kernels/sympy.md) (independent cross-check for the gated covcurl case).
- [Verification system](../systems/verification.md) (residue and linearized-EOM-match checks).

## Entry points for modification

- Add a new scaffold in `noether/kernels/cadabra/templates.py` and register it in `_EXAMPLE_TEMPLATE` and `_variation_key` in `noether/kernels/cadabra/generate.py`.
- Extend `_supported` in `derive_perturbation` to admit new field kinds; refuse rather than guess if no scaffold exists.
- Add eval-backed coverage in `evals/` before exposing a new perturbation class.

## Key source files

| File | Why it matters |
| --- | --- |
| `noether/orchestrator/derive.py` | `derive_perturbation` and `derive_field` dispatch and gating. |
| `noether/kernels/cadabra/generate.py` | Scaffold selection (`_variation_key`), X-coupling and non-abelian detection, perturbation contract. |
| `noether/kernels/cadabra/templates.py` | Frozen, golden-tested perturbation scaffolds. |
| `noether/npr/schema.py` | `ObjectDecl.gauge_group` marker for Yang-Mills selection. |
| `evals/eval3p_scalar_perturbation.py` et al. | Per-scaffold eval references. |
