# How to contribute

This is the front door for contributors. The authoritative contributor contract is [`AGENTS.md`](../../../AGENTS.md) in the repo root; this page and its siblings summarize the mechanics. Read [`patterns-and-conventions.md`](patterns-and-conventions.md) for the rules and the pages below for the day-to-day workflow.

Project Noether is an agentic symbolic-physics collaborator: a physicist writes an action in LaTeX, answers a short set of clarifying questions, and gets back verified equations of motion, the ADM decomposition, and perturbed theory. Two promises are mechanically enforced, not merely policy: no unearned assertions (only kernel output enters a result) and no silent guessing (an open ambiguity ledger blocks planning). Contributing means working inside those promises.

## Picking up work

1. Locate the task against the horizon plan in [`NORTH_STAR.md` §17](../../../NORTH_STAR.md) and the architecture in [`docs/02_TECH_SPEC.md`](../../../docs/02_TECH_SPEC.md). If the task expands scope, flag it on the issue rather than quietly building it.
2. If the task adds a capability, the eval comes first. A capability does not exist until its eval in [`docs/04_EVALS.md`](../../../docs/04_EVALS.md) passes end to end with checks green. Add the eval before the capability.
3. Implement behind the NPR boundary: orchestrator logic stays kernel-agnostic, kernel specifics stay in adapters. See [`patterns-and-conventions.md`](patterns-and-conventions.md).
4. Run the relevant evals and tests. A physics-bearing change with no kernel-backed test does not merge.
5. Update the docs the change touches, in the same change. Stale docs are bugs.
6. In your PR summary, separate "what the kernel verified" from "what I reasoned about". That boundary is the product's core promise; practice it in development too.

## Definition of done

A change is ready to merge when all of the following hold:

- The relevant evals pass end to end with their kernel checks green (`.venv/bin/python -m pytest -q` and the specific `noether evalN` commands the change touches).
- `ruff format --check` and `ruff check` are clean on the paths you touched.
- Frontend changes, if any, pass `npm run build` and `npm test` in `frontend/`.
- Any doc the change affects is updated in the same PR.
- The PR summary states what the kernel verified versus what was reasoned about, and names the eval(s) that gate the new behavior.

## Pull request process

1. Branch from `main`. Keep branches short-lived and focused on one capability or fix.
2. Commit in small chunks with imperative subject lines (`Add the cubic-Galileon scalar EOM block`, not `Added`/`adding`). The body explains the why, not the what.
3. Open the PR against `main`. CI runs three jobs (see [Tooling](tooling.md)): `lint-and-test` (ruff plus pytest, cadabra tests skip on the default runner), `frontend` (`next build`), and `cadabra-golden` (the cadabra-backed golden derivations on a runner with `cadabra2` installed).
4. Address review comments. Reviewers check the NPR boundary, the convention block, provenance, and whether a kernel-backed test covers any new physics.
5. Squash-merge once CI is green and the change meets the definition of done above.

## Review expectations

Reviewers apply the non-negotiable rules from [`AGENTS.md` §3](../../../AGENTS.md):

- Does any computed expression in code, tests, or docs come from a kernel run or a citable standard result, and is it marked as such?
- Does every expression crossing a kernel boundary carry its convention block?
- Does a returned expression come with its script, kernel version, and assumptions (provenance)?
- Does product code ever silently guess a field role, symmetry, or gauge? It must not.
- Is there a kernel-backed test for any new physics?
- Is anything outside a kernel adapter importing or depending on a specific CAS? It must not.

If a review comment maps to one of those rules, it blocks merge until resolved.

## Sibling pages

- [Development workflow](development-workflow.md) - the branch, code, test, PR, merge cycle in detail.
- [Testing](testing.md) - frameworks, markers, golden tests, eval gates, the stub adapter.
- [Debugging](debugging.md) - logs, common errors, and a troubleshooting runbook.
- [Tooling](tooling.md) - build system, linters, code generators, and CI.
- [Patterns and conventions](patterns-and-conventions.md) - the rules and how they show up in code.

## Related pages

- [Getting started](../overview/getting-started.md) for install and first run.
- [Verification](../systems/verification.md) and [Kernels](../systems/kernels/index.md) for what the tests target.
- [Features](../features/index.md) for how evals gate features.
