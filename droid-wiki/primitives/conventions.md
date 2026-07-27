# Conventions

Active contributors: KeigoShimadaCC

## Purpose

`noether/npr/conventions.py` defines the `Conventions` pydantic model and the repo-default instance `NOETHER_DEFAULT_V1`. Every expression that crosses a kernel boundary carries one of these. No file, function, or test in this repository may assume a convention silently; repo-wide defaults exist but are referenced by name (`noether-default-v1`), never implied. This is AGENTS.md section 3 rule 2 made structural.

The model is `frozen=True`, so a convention block is immutable once constructed. A session that overrides a default does so by building a new `Conventions` instance through elicitation, not by mutating the default.

## Field table

The authoritative source is `noether/npr/conventions.py`. The table below reproduces the fields, their allowed values, and their meaning.

| Field | Allowed values | Meaning |
|---|---|---|
| `id` | `str` | Human-readable name of the convention block, e.g. `noether-default-v1`. Referenced by name in code and provenance. |
| `dimension` | `int` or `str` | Spacetime dimension. An int (e.g. `4`) is concrete; a string like `"D"` is a symbolic dimension. |
| `signature` | `mostly-plus` / `mostly-minus` | Metric signature. `mostly-plus` is `(-,+,+,+)` in 4D. |
| `riemann_sign` | `+1` / `-1` | Sign in front of the Riemann definition. `+1` means `R^rho_{sigma mu nu} = +(d Gamma^rho_{nu sigma}/d x^mu - ...)`. |
| `torsion_sign` | `+1` / `-1` | `+1` means `T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}`. `-1` flips the order. |
| `nonmetricity_definition` | `nabla-g` / `minus-nabla-g` | `nabla-g`: `Q_{lambda mu nu} = nabla_lambda g_{mu nu}`. `minus-nabla-g`: `Q_{lambda mu nu} = -nabla_lambda g_{mu nu}`. |
| `contortion_sign` | `+1` / `-1` | `+1`: `K^lambda_{mu nu} = +(1/2)(T^lambda_{mu nu} + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu} + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu})`. `-1` flips the leading factor to `-(1/2)`. |
| `disformation_sign` | `+1` / `-1` | `+1`: `L^lambda_{mu nu} = +(1/2) g^{lambda rho}(-Q_{mu nu rho} - Q_{nu rho mu} + Q_{rho mu nu})`. `-1` flips the leading factor to `-(1/2)`. |
| `ricci_contraction` | `first-third` / `first-fourth` | `first-third`: `R_{mu nu} = R^lambda_{mu lambda nu}`. `first-fourth`: `R_{mu nu} = R^lambda_{mu nu lambda}`. Elicited under an independent connection because Ricci is then non-symmetric. |
| `field_strength_definition` | `exterior-derivative` / `covariant-curl` | `exterior-derivative`: `F_{mu nu} = 2 partial_{[mu} A_{nu]} = dA`. `covariant-curl`: `F_{mu nu} = 2 nabla_{[mu} A_{nu]}` with the full affine connection. The two coincide under Levi-Civita but differ by `T^lambda_{mu nu} A_lambda` under torsion (VAL-GEOM-020), so the choice is elicited when a vector field lives on a non-Levi-Civita background. |
| `symmetrization_weight` | `1/n!` / `1` | Weight applied to `(anti)symmetrization`. `1/n!` gives `A_{(mu nu)} = (1/2)(A_{mu nu} + A_{nu mu})`. `1` is the unnormalized convention. |
| `K_sign` | `+1` / `-1` | Extrinsic-curvature sign. `+1`: `K_{ij} = +nabla_i n_j` (expansion-positive, standard for mostly-plus). `-1`: `K_{ij} = -nabla_i n_j` (MTW, common for mostly-minus). |
| `foliation_normal` | `future-directed` / `past-directed` | `future-directed`: `n_mu` is the future-pointing timelike normal (`n_mu = (-N, 0, ..., 0)` for mostly-plus). `past-directed`: `n_mu = (+N, 0, ..., 0)` for mostly-plus. |

## The repo default

`NOETHER_DEFAULT_V1` is the named default block. Its values are:

| Field | Value |
|---|---|
| `id` | `noether-default-v1` |
| `dimension` | `4` |
| `signature` | `mostly-plus` |
| `riemann_sign` | `+1` |
| `torsion_sign` | `+1` |
| `nonmetricity_definition` | `nabla-g` |
| `contortion_sign` | `+1` |
| `disformation_sign` | `+1` |
| `ricci_contraction` | `first-third` |
| `field_strength_definition` | `exterior-derivative` |
| `symmetrization_weight` | `1/n!` |
| `K_sign` | `+1` |
| `foliation_normal` | `future-directed` |

Worked derivations in `docs/04_EVALS.md` are written in these conventions unless a section says otherwise and says so explicitly.

## The metric-affine block

When the connection is independent the convention block carries the metric-affine fields on top of the `noether-default-v1` base. The metric-affine block is not a separate pydantic model; it is the same `Conventions` instance with the metric-affine fields set to the session's elicited values. The fields that become elicited choices under an independent connection are:

- `torsion_sign`
- `nonmetricity_definition`
- `contortion_sign`
- `disformation_sign`
- `ricci_contraction` (non-symmetric Ricci makes this a real choice)
- `field_strength_definition` (differs from `dA` under torsion)
- `K_sign`
- `foliation_normal`

Every metric-affine derivation records these fields in its convention block. Changing any of them changes the result; no field is assumed silently.

## How it works

1. The repo default `NOETHER_DEFAULT_V1` is imported where a convention is needed and no override exists. It is referenced by name, never reconstructed inline.
2. Elicitation (`noether/orchestrator/elicit.py`) proposes resolutions for open convention ambiguities (Ricci-contraction, field-strength definition) with on-menu choices and rationale. The model never auto-applies them.
3. `apply_resolutions` builds a new `Conventions` instance from the human-confirmed choices. Because the model is frozen, the old instance is untouched.
4. The active `Conventions` instance lives on `NPR.conventions` and is threaded through every derive, perturb, and ADM call. Each derivation result carries its convention block so a reader can audit which conventions produced it.
5. The provenance bundle writer stores the convention block alongside the result; a reloaded session restores it verbatim.

## Integration points

| System | How it uses `Conventions` |
|---|---|
| NPR schema | `NPR.conventions: Conventions` is a required field. |
| Orchestrator | Elicitation proposes overrides; derive threads the active block into kernel scripts and result metadata. |
| Kernel adapters | The script generator reads the active conventions to set curvature signs, Ricci contraction, field-strength form, and K-sign in the kernel script. |
| Verification | V0 reads `metric_compatible` (derived from the connection spec, which pairs with conventions) to decide whether index raising/lowering across `nabla` is free. |
| Provenance | The convention block is bundled with every result and reloaded on session resume. |
| LaTeX renderer | Signature-aware rendering depends on the convention block. |

## Entry points for modification

- Add a new convention field: extend the `Conventions` model in `noether/npr/conventions.py`, add it to `NOETHER_DEFAULT_V1`, update `AGENTS.md` section 5, and update every kernel script generator that reads conventions.
- Change a default value: edit `NOETHER_DEFAULT_V1` and update `docs/04_EVALS.md` and any pinned golden output that depends on the old value.
- Make a previously-default field elicited: wire it into `noether/orchestrator/elicit.py` as an open ambiguity and ensure the resolve path can build the new `Conventions` instance.

## Key source files

| File | Role |
|---|---|
| `noether/npr/conventions.py` | The `Conventions` model and `NOETHER_DEFAULT_V1`. Authoritative source. |
| `noether/npr/schema.py` | `NPR.conventions` holds the active block. |
| `noether/orchestrator/elicit.py` | Proposes and applies convention resolutions. |
| `AGENTS.md` | Section 5 prose for the default conventions; section 5 metric-affine extensions. |
| `docs/04_EVALS.md` | Worked derivations stated in `noether-default-v1` unless noted. |
