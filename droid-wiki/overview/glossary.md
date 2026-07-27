# Glossary

Project-specific terms and domain vocabulary used across the wiki and the codebase.

## System terms

**NPR (Noether Problem Representation)** - The backend-agnostic contract between "what the physicist meant" and "what any kernel executes". A versioned, diffable pydantic schema holding conventions, geometry, objects, the action, the task, and the ambiguity ledger. Defined in `noether/npr/schema.py`. See [NPR](../systems/npr.md).

**Ambiguity ledger** - The list of unresolved questions on an NPR. A non-empty ledger structurally blocks planning (`build_plan` raises `AmbiguityBlocked`). This is the mechanical form of "no silent guessing".

**Well-posed** - An NPR with an empty unresolved-ambiguity ledger. Only a well-posed NPR can be planned and derived.

**Convention block** - A named, frozen set of physics conventions (dimension, signature, curvature sign, torsion sign, and so on) that travels with every expression crossing a kernel boundary. The repo default is `noether-default-v1`; metric-affine work adds `metric-affine-v1`. Defined in `noether/npr/conventions.py`. See [Conventions](../primitives/conventions.md).

**Expr / AST** - The typed expression tree (`Num`, `Sym`, `Func`, `Tensor`, `Deriv`, `Pow`, `Prod`, `Sum`) that represents the Lagrangian and every kernel result. LaTeX is a rendering of this tree, never the source of truth. Defined in `noether/npr/ast.py`. See [Expression AST](../primitives/expression-ast.md).

**ComputedResult** - A kernel-computed expression plus its receipt: the kernel name and version, the script, the raw output, and any check verdicts. The only object allowed to carry a computed expression into a result. Defined in `noether/kernels/base.py`. See [Computed results and provenance](../primitives/computed-result.md).

**Capability** - A capability-tagged unit of kernel work (`VARY`, `IBP`, `CANONICALIZE`, `SUBSTITUTE`, `PERTURB`, `ADM`, `COMPONENT_EVAL`, `INDEPENDENT_CONNECTION`). The planner selects kernels by capability, never by name.

**Provenance bundle** - The on-disk record of a derivation: `result.json`, `assumptions.json`, `plan.json`, `checks.json`, `derivations.json`, plus the scripts and raw output. Written under `results/<session>/<result-id>/`. See [Provenance](../systems/provenance.md).

**FieldDerivation** - The presentation-shaped result for one field's derivation, carrying the result LaTeX, the `verified` verdict, per-check status, the `detail` (always non-empty), the `teaching` prose, and the convention block. Defined in `noether/orchestrator/derive.py`.

**Teaching channel** - A separate prose field that narrates the tradeoffs of geometric choices (torsion to spin coupling, projective freedom to a non-unique connection). It is reasoned, not kernel-verified, and never sets a result or mutates the NPR. See [Teaching channel](../features/teaching-channel.md).

## Verification terms

**Verification ladder** - The five-rung check stack: V0 well-formedness, V1 structural invariants, V2 identity checks, V3 limiting cases, V4 independent recomputation. See [Verification](../systems/verification.md).

**Dual gate** - The rule that a metric-affine result is verified only when both the Cadabra in-script residue check and the independent SymPy general-connection cross-check agree.

**Torsion trap** - The failure mode where a Levi-Civita shortcut silently drops a torsion term yet still reports a zero residue. The dual gate exists to catch it.

**Residue check** - A kernel's in-script comparison of an independently derived variation against an independently stated candidate equation. A zero residue is a V3-style equality verified by computation.

**Gated result** - A derivation returned with `verified=false` and a non-empty `detail` naming the blocker, rather than a fabricated answer. Used for capabilities that need machinery not installed here (for example the higher Horndeski `G4(phi,X)R`/`G5` closures that need xAct's `SortCovDs`).

**Stale result** - A prior result whose NPR version has been superseded by a later resolution. It still holds for the version it was computed against, so it is marked stale rather than dropped or silently trusted.

## Physics terms

**Action** - The starting point: `S = \int d^4x \sqrt{-g}\, \mathcal{L}`. The physicist supplies it in LaTeX.

**EOM (equations of motion)** - The result of varying the action with respect to a field (the `vary` task).

**ADM (3+1) decomposition** - The split of the gravitational sector into lapse, shift, and spatial metric on a foliation of spacetime; produces the Gauss-Codazzi split and the constraint structure (the `adm` task). See [ADM decomposition](../features/adm-decomposition.md).

**Perturbation** - Expansion of the action to quadratic order around a background to read off the linearized equations of motion (the `perturb` task). See [Perturbation](../features/perturbation.md).

**Metric-affine geometry** - The general setting where the connection is independent of the metric, carrying torsion `T` and non-metricity `Q`. Levi-Civita is the `T = Q = 0` special case. See [Metric-affine geometry](../features/metric-affine-geometry.md).

**Levi-Civita connection** - The unique torsion-free, metric-compatible connection determined by the metric.

**Torsion** `T^lambda_{mu nu}` - The antisymmetric part of the connection, `Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}`.

**Non-metricity** `Q_{lambda mu nu}` - The failure of the connection to preserve the metric, `nabla_lambda g_{mu nu}`.

**Contortion `K` and disformation `L`** - The tensors in the post-Riemannian decomposition `Gamma = LC + K(T) + L(Q)`, splitting the affine connection into its Levi-Civita part, the torsion contribution, and the non-metricity contribution.

**Palatini variation** - Treating the metric and the connection as independent variables and varying with respect to each. The pure Einstein-Hilbert case has a projective mode: `Gamma = LC(g) + delta^lambda_nu A_mu` with `A_mu` arbitrary, so the connection is never uniquely fixed.

**Einstein-Cartan** - A metric-affine theory with torsion sourced by spin; the connection equation relates torsion to the hypermomentum.

**Teleparallel gravity** - Gravity described by torsion (`f(T)`, metric teleparallel on the tetrad/Weitzenbock connection) or by non-metricity (`f(Q)`, symmetric teleparallel in the coincident gauge), with vanishing curvature. See [Teleparallel gravity](../features/teleparallel.md).

**Horndeski / Galileon** - Scalar-tensor theories with second-order equations of motion. Noether covers the cubic Galileon and nonminimal scalar-tensor sectors by composition; the higher `G4(phi,X)R`/`G5` densities are gated.

**Field-strength definition** - For a gauge potential `A_mu`, either `F = dA` (exterior derivative, `2 partial_{[mu} A_{nu]}`) or `F = nabla A` (covariant curl, `2 nabla_{[mu} A_{nu]}`). The two differ by `T^lambda_{mu nu} A_lambda` under torsion, so the choice is elicited on a non-Levi-Civita background.
