# Lore

Data collected on 2026-07-27. All dates are taken from git commit timestamps. The repository was written in a single eight-day sprint, 2026-06-12 to 2026-06-19, so the "history" is really a sequence of phases within one burst rather than a long evolution.

## Eras

### Era 1: Skeleton and Horizon 1 evals (2026-06-12)

The repository opened with the North Star docs and a walking skeleton. The initial commit at 2026-06-12 02:07 brought in `NORTH_STAR.md`, the full `docs/` set, the `noether` package skeleton (`npr`, `kernels`, `cli`), and eval 1 as a trace. Within seven hours the Horizon 1 evals were running end to end: evals 2 through 4 (Palatini, scalar-tensor, Maxwell) landed at 08:50, and eval 5 (Gauss-Bonnet Lovelock algebra and the variational derivation to the Lanczos H) followed across two commits at 09:22 and 09:39.

This era also established the supporting machinery that would survive the whole sprint: the LaTeX action ingest with its deterministic parser and ambiguity ledger (14:19), CI with central kernel version pinning (14:21), and LLM elicitation through an ambient-auth CLI subprocess (15:58). The session surfaces came at the end of the day: the HTTP API (21:32), the MCP server (21:37), the conversational CLI (21:43), a cadabra-equipped CI job (21:46), and the Next.js web frontend (23:58).

### Era 2: Horizon 2, ADM and perturbation (2026-06-12 to 2026-06-15)

The Horizon 2 gates opened the same evening as the skeleton. Eval 1s, the ADM decomposition of GR, landed at 2026-06-12 16:58, and eval 3s, the spectrum around Minkowski, followed at 21:24. The next day brought the general EOM derivation path for arbitrary well-posed actions (2026-06-13 14:27) and the first perturbation scaffold, the kernel-verified scalar quadratic action (16:19), wired into the derive path at 16:58.

After a quiet 2026-06-14, the graviton perturbation scaffold arrived on 2026-06-15 (08:30) and the ADM split was wired through the derive pipeline that evening (21:35). This era appears to be where the product's shape settled: the derive path, the perturbation scaffolds, and the ADM split all became first-class tasks reachable the same way.

### Era 3: Presentation, persistence, and the compositional blocks (2026-06-16 to 2026-06-17)

2026-06-16 opened with the web client gaining derivation-tree and publication-LaTeX export views (00:04) and result history persistence across server, MCP, and web (00:27). Docs were synced to the shipped Horizon 2 state at 14:05. The kinetic shorthand `X` and subscripted coupling parsing arrived at 15:43, followed by the compositional blocks path: eval 6 (cubic Galileon scalar EOM) at 16:03, eval 7 (general scalar Horndeski by block composition) at 18:20, and eval 8 (curvature-coupled blocks and the metric EOM) at 21:57.

2026-06-17 continued the compositional push: the cubic Galileon metric EOM at 09:48, gauge-field perturbations for abelian and non-abelian potentials (evals 3a, 3y) at 10:28, and the k-essence quadratic action with sound speed (eval 3k) at 10:45. A held-out note at 10:50 documented why the higher Horndeski densities G4(phi,X)R and G5 stay out of the compositional path. The afternoon added a residue-checked curvature reduction layer for those higher Horndeski terms (15:14) and a targeted quartic box-commutator with a pinned closure technique (15:27), including a fix for a Cadabra 2.5.14 crash (18:56).

### Era 4: The metric-affine lift (2026-06-17 to 2026-06-18)

The largest rewrite of the sprint began the evening of 2026-06-17 and consumed most of 2026-06-18. Until this point the geometry was Levi-Civita only. Starting at 21:29 the connection identity was preserved in parsed curvature AST, torsion and non-metricity were promoted to geometric shorthands (21:37), metric-affine NPR conventions were modeled explicitly (21:49), and metric-affine validation was gated on explicit metric compatibility (21:57).

Through the night and morning of 2026-06-18 the post-Riemannian machinery was built out: metric-affine geometry questions at ingest (22:12), geometry resolutions wired into live planning (22:28), metric-affine definition shorthands (22:37), curvature primitives for an independent connection (23:27), the torsion primitive and irreducible decomposition (23:51), the torsionful commutator and non-symmetric scalar Hessian (00:12), the non-metricity primitive and irreducible decomposition (00:35), the post-Riemannian decomposition `Gamma = LC + K(T) + L(Q)` (01:54), modified Bianchi identities (02:42), the exterior-derivative-versus-covariant-curl identity for 1-forms (02:57), and the dual gate enforcement with a convention sign falsifier (03:10).

The connection then became a variational field: connection variation was routed through the general derivation path (04:20), both Palatini Einstein-Hilbert equations were derived (04:39), the connection EOM was made reachable through MCP and HTTP with refusal discipline (05:24), the Einstein-Cartan connection equation and hypermomentum decomposition arrived (05:49), and a curvature-free flag enabled teleparallel and symmetric-teleparallel family routing with gated f(T)/f(Q) evals (06:08).

### Era 5: Teleparallel f(T) and symmetric-teleparallel f(Q) (2026-06-18)

A focused sub-era on 2026-06-18 morning: the f(Q) coincident-gauge EOM template with a SymPy cross-check (09:14), a comprehensive README (09:14), f(Q) verification with SymPy cross-checks (09:20), and the verified f(T) tetrad EOM via the Weitzenbock formulation (09:56). A later commit at 13:41 fixed the f(Q) coincident-gauge boundary-term identity, correcting the P tensor, Q scalar, and boundary identity sign. The f(T) docs were flipped to verified at 14:36.

### Era 6: Vector-affine and metric-affine perturbation (2026-06-18)

The afternoon of 2026-06-18 extended the perturbation task to the metric-affine background. The gauge field-strength definition was elicited under a non-Levi-Civita connection (10:15), the vector Maxwell EOM on a metric-affine background was derived with the field-strength choice consequence (10:59), and multi-field Palatini scalar-tensor EOMs were derived with the Levi-Civita limit pinned (11:23). The G4/G5 Horndeski best-effort path was wired through the general derive path at 11:44.

The metric-affine perturbation scaffolds followed: `pert_metric_affine_quadratic` (15:54), acceptance gating and the T=Q=0 Levi-Civita limit (16:15), metric-affine vector and coupled perturbation modes (16:42), reachability and persistence across surfaces (17:07), conventions threaded through perturbation templates (17:30), and gauge-field perturbations on metric-affine backgrounds routed to vector-affine templates (18:07). The metric-affine ADM (3+1) decomposition on the general adm path arrived at 19:08, with the ADM verification model pinned at 19:44, matter hypermomentum surfaced in the constraint structure at 20:30, and ADM conventions, no-metric refusal, and cross-surface persistence at 21:00.

### Era 7: Cross-surface consistency and polish (2026-06-18 to 2026-06-19)

The sprint closed with consistency and cleanup work. Conventions were threaded through EOM and perturbation derivations with cross-area integration flows verified (2026-06-18 22:51). On 2026-06-19 the geometry inference contract tests and doc coverage landed (00:08), inference was grounded in action cues with conventions proposed on-menu (00:44), and the first-class teaching narration channel on `FieldDerivation` was added (01:19). The verified/gated verdict and reason were surfaced consistently across backend surfaces (02:12), and the web client gained teaching panels, gated verdict badges, and cross-flow provenance rendering (02:29, 02:55).

The final stretch was cleanup: removing dead `_has_explicit_connection` from ingest.py (04:41), moving inline Cadabra scripts from test files into registered templates (07:10), adding the `K_sign` and `foliation_normal` convention fields (08:21), a Pydantic validator rejecting empty `FieldDerivation.detail` (11:09), replacing a bare `TypeError` with `UnhandledASTNodeError` (11:44), Jest component tests for perturbation and ADM rendering (12:13), and the final two cleanup commits removing the dead `_ADM_K_TEX` constant and a duplicate unused `render()` call (14:50, 14:51). The README was updated to the final metric-affine state at 15:43, the last commit of the sprint.

## Longest-standing features

Several pieces from the initial commit on 2026-06-12 are still actively used and structurally central:

- The NPR schema (`noether/npr/`), including the convention block, the AST, and the LaTeX parser.
- The session state machine and the planner with its ambiguity gate (`noether/orchestrator/`).
- The Cadabra adapter, driving Cadabra2 as a sandboxed subprocess with frozen golden templates (`noether/kernels/cadabra/`).
- The SymPy kernel adapter, later expanded into the geometry oracle and ADM verifier.
- The HTTP session API and the MCP server, both carrying the no-guessing contract from their first commit.

These were not throwaway scaffolds. They were the load-bearing core, and later eras extended them rather than replacing them.

## Deprecated and removed features

A few things were built and then explicitly removed or replaced, visible in late-sprint cleanup commits:

- Inline Cadabra scripts in test files were replaced by calls to `templates.get()` (2026-06-19 07:10 and 13:15). The tests originally embedded their own scripts; the cleanup moved them into the registered template set so the kernel's template registry became the single source.
- The dead `_ADM_K_TEX` constant was removed (2026-06-19 14:50). It appears to have been a leftover from an earlier ADM display path that was superseded when ADM conventions were threaded through.
- The dead `_has_explicit_connection` helper was removed from `ingest.py` (2026-06-19 04:41), likely made redundant when geometry inference was grounded in action cues.
- A duplicate unused `render()` call was removed from `PerturbationAdmTree.test.tsx` (2026-06-19 14:51).

## Major rewrites

Two rewrites stand out, both in the second half of the sprint.

The metric-affine lift (Era 4) rewrote the geometry from Levi-Civita only to a general affine connection with torsion and non-metricity. This touched the parser (preserving connection identity in curvature AST), the conventions schema (adding torsion sign, non-metricity definition, contortion and disformation signs, Ricci contraction, field-strength definition, K sign, and foliation normal), the geometry oracle (`geometry.py`, which became the second-largest file in the repo), the validation layer (gating on explicit metric compatibility), and the derive path (routing connection variation and adding the dual gate). It also produced a new curvature reduction layer (`curvature.py`) and the post-Riemannian decomposition primitives.

The compositional blocks path (Era 3) replaced per-theory templates for the scalar-tensor and Galileon sectors with a single decompose-and-assemble flow in `blocks.py`. Instead of a frozen template per theory, an additive Lagrangian is decomposed into registered building blocks and one script is assembled for the real action plus an independent candidate, then residue-checked. This covered evals 6, 7, and 8, and the higher Horndeski densities were held out to a separate best-effort G4/G5 path.

## Growth trajectory

The package grew from a walking skeleton to 51 Python files across 8 subpackages in eight days. The initial commit already carried the `npr`, `kernels`, and `cli` subpackages plus the full docs set. The orchestrator, verify, provenance, llm, server, and mcp subpackages were added the same day. The subpackage count did not grow after 2026-06-12; what grew was the depth and capability of each, especially `kernels` (which split into the cadabra and sympy_kernel subtrees and added `blocks.py`, `curvature.py`, and `ft_tetrad.py`) and `orchestrator` (which added the derive, perturbation, and ADM paths). The frontend arrived on day one as a thin client and gained its presentation features (derivation tree, LaTeX export, teaching panel, badges) only in the final era.
