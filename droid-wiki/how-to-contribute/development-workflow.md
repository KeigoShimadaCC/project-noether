# Development workflow

The cycle: locate, branch, implement behind the NPR boundary, test with the relevant evals, update docs, open the PR, separate kernel-verified from reasoned-about in the summary, merge. The authoritative source is [`AGENTS.md` §8](../../../AGENTS.md); this page is the operational walkthrough.

## Locate the task

1. Find the task in the horizon plan ([`NORTH_STAR.md` §17](../../../NORTH_STAR.md)) and the relevant section of [`docs/02_TECH_SPEC.md`](../../../docs/02_TECH_SPEC.md). The horizon plan tells you which capability the task belongs to; the tech spec tells you where it lives in the code.
2. If the task expands scope beyond its horizon, flag it on the issue. Do not quietly build scope.
3. If the task adds a capability, write or extend the eval first (see [Testing](testing.md)). The capability does not exist until the eval passes end to end with checks green.

## Branch

```sh
git checkout main
git pull
git checkout -b <short-imperative-name>
```

Keep branches focused on one capability or fix. Small branches review faster and merge sooner.

## Implement behind the NPR boundary

The NPR (Noether Problem Representation) is the only language the orchestrator speaks. Two hard rules:

- **Orchestrator logic stays kernel-agnostic.** Nothing outside a kernel adapter (`noether/kernels/`) may import or depend on a specific CAS. The orchestrator emits NPR and consumes kernel results; it never emits kernel syntax into a result.
- **Kernel specifics stay in adapters.** Cadabra script assembly lives in `noether/kernels/cadabra/`, SymPy component verification in `noether/kernels/sympy_kernel/`. Kernel version pins live in one place: `noether/kernels/versions.py`.

V0 validation stays structural. Do not treat raising or lowering across `\nabla` as free unless the active connection is explicitly metric compatible. The verification ladder is described in [`docs/02_TECH_SPEC.md`](../../../docs/02_TECH_SPEC.md) and the [Verification](../systems/verification.md) page; V0 is well-formedness only.

See [`patterns-and-conventions.md`](patterns-and-conventions.md) for the model-has-no-authority, ambiguity-gate, immutable-NPR-version, and sentinel-parsing patterns you must respect while implementing.

## Test

Run the relevant evals and tests. A physics-bearing change with no kernel-backed test does not merge.

```sh
.venv/bin/python -m pytest -q                         # full suite; cadabra tests skip if absent
.venv/bin/python -m pytest -q tests/test_<area>.py    # scoped
.venv/bin/python -m pytest -q -m kernel_cadabra       # cadabra golden tests only
.venv/bin/python -m noether.cli.main eval1            # end-to-end eval the change touches
```

If you added a capability, its eval must run green here. See [Testing](testing.md) for the full framework and marker reference.

For frontend changes:

```sh
cd frontend
npm run build       # next build, includes type checking
npm test            # jest
npm run typecheck   # tsc --noEmit
```

## Update docs

When you change behavior, update the affected doc in the same change. Stale docs are bugs. The doc map is in [`docs/00_INDEX.md`](../../../docs/00_INDEX.md). If an implementation decision contradicts a doc, either fix the implementation or update the doc in the same PR, with a note on why.

## Commit

Small, imperative subject lines. The body explains the why.

```
Add the cubic-Galileon scalar EOM block

The compositional path had no block for the G3 K(phi) box phi term, so
cubic-Galileon actions fell through to the model-written script path
even though the scalar EOM is canonical. Register the block in blocks.py
so derive_field assembles and residue-checks it without a model round-trip.
```

Do not mix unrelated changes in one commit. Each commit should build and test on its own where practical.

## Open the PR

Push and open a PR against `main`. CI runs `lint-and-test`, `frontend`, and `cadabra-golden` (see [Tooling](tooling.md)). The `cadabra-golden` job installs `cadabra2` from the upstream prebuilt noble package and runs the cadabra-backed golden derivations plus `eval1` end to end.

In the PR description:

- State what changed and why.
- Name the eval(s) that gate the new behavior and confirm they pass.
- Separate **what the kernel verified** from **what you reasoned about**. This boundary is the product's core promise; practice it in development too. For example: "The Cadabra residue check `residue_zero` returned True (kernel-verified). The teaching narration text was written by me and is not kernel-checked (reasoned about)."

## Merge

Squash-merge once CI is green, review comments are resolved, and the change meets the [definition of done](index.md#definition-of-done). Delete the branch after merge.
