# Project Noether — document index

Status legend: `draft` (open questions remain), `stable` (agreed, change via
explicit revision), `living` (expected to change continuously).

| # | Document | Purpose | Status |
|---|---|---|---|
| — | `../NORTH_STAR.md` | Vision: destination and why. The constitution. | stable |
| — | `../AGENTS.md` | Working guide for agents and contributors | living |
| 00 | `00_INDEX.md` | This map | living |
| 01 | `01_RESEARCH.md` | CAS landscape, prior art, kernel selection rationale | draft |
| 02 | `02_TECH_SPEC.md` | Architecture, stack, NPR schema, adapters, algorithms | draft |
| 03 | `03_METHODOLOGY.md` | Elicitation, good form, verification ladder, dev process | draft |
| 04 | `04_EVALS.md` | Five acceptance evaluations with worked solutions | stable (1-4); eval 5 draft |

## Relationships

- `NORTH_STAR.md` constrains everything. Nothing below may contradict it.
- `01_RESEARCH.md` justifies the kernel choices that `02_TECH_SPEC.md` commits to.
- `02_TECH_SPEC.md` defines the system; `03_METHODOLOGY.md` defines how the system
  behaves (elicitation, verification) and how we build it (eval-driven).
- `04_EVALS.md` is the executable meaning of "Horizon 1 / Horizon 2 done". Evals
  1 to 4 gate Horizon 1 (equations of motion loop); eval 5 plus the stretch tasks
  gate Horizon 2 (identities, ADM, perturbation).

## Pending documents (create when the work starts)

- `05_NPR_SCHEMA.md` — frozen JSON schema of the problem representation, versioned.
- `06_KERNEL_ADAPTERS.md` — per-kernel capability matrix and adapter contracts.
- `07_PROVENANCE_FORMAT.md` — result bundle layout, reproducibility guarantees.
