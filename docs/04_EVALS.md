# 04 — Evaluation suite: five action-to-result pairs

**Status:** stable; all five evals plus the stretch tasks 1s (ADM of GR),
3s (spectrum around Minkowski), 3p (scalar quadratic action in Cadabra),
3g (graviton quadratic action in Cadabra), 3a (Maxwell quadratic action),
3y (Yang-Mills quadratic action), 3k (k-essence quadratic action with the
sound-speed kinetic mixing), 6 (cubic Galileon scalar EOM,
the first verified Horndeski member past scalar-tensor), 7 (k-essence and the
general scalar Horndeski sector by block decomposition, no per-theory template),
and 8 (nonminimal scalar-tensor by composition, the metric sector and the
curvature-coupled `F(φ)R` block, both equations of motion) implemented and
kernel-verified.
**Implementation (2026-06-12):** all five evals are executable (`evals/`, run
via `noether eval1` .. `noether eval5`) and kernel-checked against cadabra2
2.5.15: the eval 1 and 3 variation residues are zero against the targets
below, the eval 2 connection equation is solved identically by the projective
family (with `Ricci(LC + projective) - Ricci(LC) = dA` checked exactly), and
the eval 4 residue is zero with the metric never varied. SymPy component
checks confirm the eval 1 ladder, eval 2 projective inertness, the eval 3
generalized-Bianchi link in the minimal limit, and the eval 4 Noether
identity off shell, all on seeded random curved backgrounds. Signs below are
therefore pinned under `noether-default-v1`.
**Eval 5 scope:** cadabra derives the field equation from the action: Palatini
variation of all three quadratic invariants, double integration by parts,
then reduction by the contracted second Bianchi identities, the rank-2
commutator and the definitional Riemann traces, with residue exactly zero
against the Lanczos form in general dimension. Every reduction identity was
first verified numerically in the sympy kernel on a curved background under
`noether-default-v1`. Cadabra additionally verifies the Lovelock p=2 delta
algebra in symbolic D (the GB scalar and the Lovelock field-equation
contraction, Lovelock 1971); sympy verifies by component evaluation that the
Lanczos tensor is symmetric, divergence-free, identically zero in D=4 (on a
background with nonzero GB scalar, so the cancellation is real) and nonzero
in D=5.
**Role:** these five pairs are the acceptance tests for Horizons 1 and 2
(NORTH_STAR §17). Each pair specifies the input a user would give, the
elicitation Noether must perform, the result it must return, and the
verification checks that must pass. Evals 1 and 2 were specified by the project
owner; 3 to 5 extend coverage to the North Star reference classes
(scalar-tensor, gauge theory, Lovelock).

**Conventions:** all solutions below use `noether-default-v1` (AGENTS.md §5):
dimension 4 unless stated, signature `(-,+,+,+)`,
`R^ρ_{σμν} = ∂_μΓ^ρ_{νσ} - ∂_νΓ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ}`,
`R_{μν} = R^λ_{μλν}`, `T^λ_{μν} = Γ^λ_{μν} - Γ^λ_{νμ}`.

**A note on these worked solutions.** They are standard results, written down by
hand for the eval targets. Per our own rules, the implemented system must derive
them from kernels; and before the evals are frozen as executable tests, each
solution below gets one kernel-verified pass to pin exact signs under
`noether-default-v1`. Where an intermediate step is convention-sensitive, the
eval says so explicitly.

---

## Eval 1 — Einstein-Hilbert in trace form (metric variation)

**Capabilities tested:** LaTeX ingestion; role elicitation; composite-tensor
expansion before variation; metric variation with IBP; good form.

### Input

```latex
S = \int d^4x \, \sqrt{-g}\, g^{\mu\nu} G_{\mu\nu}
```

User intent: `G_{μν}` is the Einstein tensor of the Levi-Civita connection of a
symmetric metric `g`, the metric is the only dynamical field. Ask: equations of
motion.

### Expected elicitation

Noether must establish (infer-and-confirm where possible):

1. `G_{μν}` is the Einstein tensor `R_{μν} - ½g_{μν}R` built from `g`
   (not an independent field). This is the critical question: treating `G` as an
   opaque tensor makes the problem ill-posed.
2. Metric symmetric, connection Levi-Civita, dimension 4, signature, curvature
   conventions.
3. Vary with respect to `g^{μν}` (or `g_{μν}`; sign bookkeeping noted).

A sharp implementation also observes and tells the user:
`g^{μν}G_{μν} = R - 2R = -R` in d = 4, so the action is `-∫√-g R`, the
Einstein-Hilbert action with flipped sign.

### Expected result

```latex
\frac{1}{\sqrt{-g}}\frac{\delta S}{\delta g^{\mu\nu}}
  = -\left( R_{\mu\nu} - \tfrac{1}{2} g_{\mu\nu} R \right) = 0
\quad\Longleftrightarrow\quad
G_{\mu\nu} = 0
```

Good form: the canonical presentation is `G_{μν} = 0`, with the overall factor
`(-1)` reported in the provenance, not left to clutter the displayed equation.

### Derivation sketch

1. Expand: `g^{μν}G_{μν} = g^{μν}R_{μν} - ½·4·R = -R`.
2. `δ(√-g) = -½√-g g_{μν} δg^{μν}`; `δR = R_{μν}δg^{μν} + g^{μν}δR_{μν}` with
   `g^{μν}δR_{μν} = ∇_μ v^μ` a total derivative (IBP, surface term recorded).
3. Collect: `δS = -∫√-g (R_{μν} - ½g_{μν}R) δg^{μν}`.

### Verification checks

- V0: index balance, symmetry of result in `μν`.
- V1: result symmetric (required for a metric EOM).
- V2: `∇^μ G_{μν} ≡ 0` identically (contracted Bianchi), computed by kernel.
- V3: agreement with the standard EH variation: varying `+∫√-g R` must give
  `+G_{μν}` and the two runs must differ by exactly the overall sign.
- V4 (H2): cross-kernel recomputation, or random-metric component spot check.

### Stretch task (Horizon 2 gate)

ADM decomposition of the same action with respect to a stated foliation: return
lapse/shift/spatial-metric form, Hamiltonian and momentum constraints separated
from evolution terms; regression target is the textbook ADM form of GR.

**Status: implemented (eval 1s; `evals/eval1s_adm.py`).** Conventions:
`noether-default-v1`, foliation by `t = const` spacelike slices, future-pointing
unit normal `n_μ = (-N, 0, 0, 0)`, lapse `N`, shift `N^i`, induced metric
`h_ij`, `D` the Levi-Civita derivative of `h` on the slice. Every sign below
was pinned by kernel computation (residue exactly zero on a nondegenerate
1+2 component background; alternatives rejected), not asserted:

- `K_ij = (∂_t h_ij - D_i N_j - D_j N_i)/(2N) = +∇_i n_j`
  (expansion-positive convention, elicited).
- `√-g R = N√h (R⁽³⁾ + K_ij K^ij - K²) - 2 ∂_μ(√-g v^μ)`,
  `v^μ = n^ν∇_ν n^μ - n^μ∇_ν n^ν`.
- Hamiltonian constraint (vacuum): `R⁽³⁾ + K² - K_ij K^ij = 0`, kernel-equal to
  `2 G_{μν} n^μ n^ν` and to the lapse Euler-Lagrange equation of the bulk
  (the lapse enters the bulk undifferentiated).
- Momentum constraint (vacuum): `D_j (K^j_i - δ^j_i K) = 0`, kernel-equal to
  `G_{μi} n^μ`.

The constraints are the normal projections of the Einstein equations and are
first order in time derivatives (they contain only `h` and `K`), so they
constrain initial data; the spatial-spatial projections are the evolution
equations. The component background switches on every structural feature
(time-dependent curved slice, off-diagonal `h`, nonzero shift, nonconstant
lapse) and the falsifier check confirms none of the verified forms vanish
identically on it.

This decomposition is reachable through the derive pipeline: `derive_adm`
(`kind="adm"` on the server, MCP, and web clients) returns the Gauss-Codazzi
split and the Einstein-tensor projections, with the six component checks above
as its verdict. The split and the projections are universal foliation geometry,
so any well-posed action carrying a metric is accepted (an action with no metric
is refused); for pure GR in vacuum the projection left-hand sides vanish and
recover the constraints stated here. `evals/test_eval_general.py` gates that the
orchestration reproduces the kernel-verified decomposition end to end.

---

## Eval 2 — Palatini in trace form (metric + independent connection)

**Capabilities tested:** independent-connection elicitation; Palatini variation
with torsion allowed; projective gauge freedom detection; two coupled EOMs;
honest reporting of an underdetermined sector.

### Input

```latex
S = \int d^4x \, \sqrt{-g}\, g^{\mu\nu} G_{\mu\nu}(\Gamma)
```

User intent: same-looking action, but the dynamical variables are the symmetric
metric `g_{μν}` **and** an independent connection `Γ^λ_{μν}`, not assumed
symmetric in `μν` (torsion allowed). `G_{μν}(Γ)` is the Einstein tensor built
from the Riemann tensor of `Γ`:
`G_{μν}(Γ) = R_{μν}(Γ) - ½ g_{μν} g^{αβ}R_{αβ}(Γ)`.

### Expected elicitation

1. Two dynamical fields: `g` (symmetric) and `Γ` (independent; no symmetry in
   lower indices, so torsion is allowed; metric compatibility not assumed).
   Ingest gets there by opening the geometry questionnaire rather than by
   hard-coding the Palatini reading: connection type, torsion, non-metricity,
   and metric compatibility are all surfaced as on-menu questions, with no
   answer pre-selected.
2. `R_{μν}(Γ)` depends on `Γ` only; the metric enters only through `√-g` and
   contractions. Note: `R_{μν}(Γ)` is **not** symmetric in general; only its
   symmetric part couples to `δg^{μν}`.
3. Once the connection is confirmed independent, elicitation must surface the
   Ricci-contraction convention as an open on-menu choice before planning can
   continue.
4. Conventions as default otherwise; ask: equations of motion for both fields.
5. The conversational CLI loop (`noether chat` / `resume`) must render those
   geometry questions with numbered options, persist each on-menu answer through
   the shared session store, and print a plan that includes the
   `independent-connection` step once the Ricci-contraction choice is confirmed.

Once that geometry is confirmed, the definitions surface may also offer the
optional readability notation `K(T)`, `L(Q)`, and the `f(Q)` scalar `Q`. These
are human-adopted shorthands only, not derived results.

The distance between eval 1 and eval 2 is exactly one elicitation answer
(`connection.type`), which is the point: the same LaTeX string is two different
theories, and only asking can tell them apart.

### Expected result

Metric equation (using `R̃ ≡ g^{αβ}R_{αβ}(Γ)`; note `g^{μν}G_{μν}(Γ) = -R̃`):

```latex
R_{(\mu\nu)}(\Gamma) - \tfrac{1}{2} g_{\mu\nu} \tilde{R} = 0
```

Connection equation: the variation `δΓ^λ_{μν}` yields a tensor equation linear
in `∇_λ(√-g g^{μν})` and the torsion trace. Its general solution (a standard
result of the metric-affine literature) is Levi-Civita up to an arbitrary
projective mode:

```latex
\Gamma^\lambda{}_{\mu\nu}
  = \left\{ {}^\lambda_{\mu\nu} \right\}(g) + \delta^\lambda{}_\nu\, \xi_\mu ,
\qquad \xi_\mu \ \text{arbitrary}
```

because the action is invariant under the projective shift
`Γ^λ_{μν} → Γ^λ_{μν} + δ^λ_ν ξ_μ` (index placement is convention-sensitive; the
kernel must verify the invariance under the declared conventions, and the eval
accepts whichever placement the verified invariance fixes).

The projective mode drops out of `R_{(μν)}(Γ)`, so substituting back:

```latex
G_{\mu\nu}(g) = 0
```

Vacuum Palatini gravity in this form is dynamically identical to GR. Noether
must say this, must flag that `ξ_μ` is physically inert here but would matter
with matter couplings sensitive to the connection, and must not pretend the
connection is uniquely determined.

### Derivation sketch

1. Trace observation: `g^{μν}G_{μν}(Γ) = -g^{μν}R_{μν}(Γ)`, so
   `S = -∫√-g g^{μν}R_{μν}(Γ)`: minus the standard Palatini EH action.
2. `δg`: only `√-g` and the explicit `g^{μν}` vary; no IBP needed. Gives the
   metric equation with `R_{(μν)}`.
3. `δΓ`: Palatini identity with torsion,
   `δR^ρ_{σμν} = ∇_μ δΓ^ρ_{νσ} - ∇_ν δΓ^ρ_{μσ} + T^λ_{μν} δΓ^ρ_{λσ}`;
   integrate by parts with the **independent** connection (non-metricity means
   `∇√-g ≠ 0`; the kernel handles the density weight). The precise index form of
   the resulting equation is kernel territory; the acceptance criterion is its
   general solution above, checked against the metric-affine literature.
4. Detect projective invariance; report the solution family; substitute the
   Levi-Civita representative; reduce the metric equation to `G_{μν}(g) = 0`.

### Verification checks

- V0/V1 on both equations; metric equation symmetric. V0 stays structural on
  the metric-affine path, so a balanced expression mixing `R_{\mu\nu}(\Gamma)`,
  `T`, and `Q` passes, while genuinely malformed index use still fails.
- V2: projective Noether identity: the trace of the connection equation that
  corresponds to the projective direction must vanish identically (kernel
  computation). Generalized Bianchi identity for diffeomorphisms relates the
  two equations; check the on-shell consistency.
- V3: limiting case: imposing `Γ = Levi-Civita(g)` by hand must reproduce
  eval 1's result exactly. Imposing zero torsion and metric compatibility must
  reduce the connection equation to `0 = 0`.
- V4 (H2): cross-kernel; component spot check on a random metric with the
  solved connection.

### Stretch task

Repeat with symmetric `Γ` (torsion forbidden): the solution family becomes the
symmetrized projective mode `δ^λ_{(μ}ξ_{ν)}`. Noether must handle the changed
constraint and explain the difference.

---

## Eval 3 — Nonminimally coupled scalar-tensor theory

**Capabilities tested:** multi-field variation; coupling functions of fields;
IBP producing `∇∇F` structures; collected good form; generalized Bianchi check.

### Input

```latex
S = \int d^4x \sqrt{-g}\,\left[ F(\phi) R
    - \tfrac{1}{2}\nabla_\mu\phi \nabla^\mu\phi - V(\phi) \right]
```

Dynamical: `g_{μν}` and `φ`. `F`, `V` arbitrary coupling functions. Ask: both
equations of motion.

### Expected elicitation

Roles (`F`, `V` are functions of `φ` only, not independent fields), Levi-Civita
connection, conventions, vary w.r.t. both fields.

### Expected result

Metric equation:

```latex
F(\phi) G_{\mu\nu}
  + g_{\mu\nu} \Box F(\phi) - \nabla_\mu \nabla_\nu F(\phi)
  - \tfrac{1}{2}\nabla_\mu\phi\,\nabla_\nu\phi
  + \tfrac{1}{4} g_{\mu\nu}\,\nabla_\alpha\phi\nabla^\alpha\phi
  + \tfrac{1}{2} g_{\mu\nu} V(\phi) = 0
```

(with `□F = F''∇_αφ∇^αφ + F'□φ` expandable on request; good form keeps `□F`
compact by default and the expansion negotiable).

Scalar equation:

```latex
\Box\phi + F'(\phi)\, R - V'(\phi) = 0
```

### Derivation sketch

Standard variation; the `F(φ)R` term produces
`F R_{μν} - ½ g_{μν} F R + g_{μν}□F - ∇_μ∇_νF` after two IBPs because
`g^{μν}δR_{μν}` no longer integrates to zero against a non-constant `F`.
Surface terms recorded in provenance.

### Verification checks

- V0/V1 as always; metric equation symmetric.
- V2: generalized Bianchi: `∇^μ(metric EOM)_{μν} = ½ (scalar EOM)·∇_νφ` must
  hold as a kernel-verified identity (diffeomorphism Noether identity).
- V3: `F = 1/(2κ)` constant must give
  `G_{μν} = κ(∇_μφ∇_νφ - ½g_{μν}∇_αφ∇^αφ - g_{μν}V)` and `□φ = V'`:
  the textbook minimally coupled scalar. `φ = const` must give
  `F G_{μν} + ½g_{μν}V = 0`, GR with a cosmological constant `Λ = V/(2F)`.
- V4 (H2): cross-kernel.

### Stretch task (Horizon 2 gate)

Perturb around Minkowski with `φ = φ₀` constant, `V(φ₀) = V'(φ₀) = 0`, to
quadratic order in a stated gauge: recover the massless graviton plus a scalar
with kinetic mixing from `F'(φ₀)`, and report the diagonalized kinetic
structure (no ghost for `F(φ₀) > 0` and positive effective scalar kinetic
term). This is the eval for the "spectrum" beat of the guiding scenario.

**Status: implemented (eval 3s; `evals/eval3s_spectrum.py`).** Conventions:
`noether-default-v1`; `F₀ = F(φ₀)`, `F₁ = F'(φ₀)`. The linearized geometry is
computed from the definitional formulas with arithmetic truncated at `O(ε²)`,
and the linear equations of motion are anchored to the cadabra-verified full
eval-3 equations by recomputing their `ε`-derivative on concrete component
fields with exact geometry (no truncation). Every coefficient was pinned by
kernel computation, with falsifiers:

- Linear EOMs: `F₀ G⁽¹⁾_{μν}[h] + (η_{μν}□ - ∂_μ∂_ν)(F₁χ) = 0` and
  `□χ + F₁ R⁽¹⁾[h] - V''₀ χ = 0`.
- The shift `h_{μν} = h̄_{μν} - (F₁/F₀) χ η_{μν}` cancels the kinetic mixing
  exactly; the opposite sign does not (kernel-rejected). Metric equation
  becomes `F₀ G⁽¹⁾[h̄] = 0`; its trace forces `R⁽¹⁾[h̄] = 0` on shell
  (`η^{μν}G⁽¹⁾ = -R⁽¹⁾`, kernel-pinned).
- With `R⁽¹⁾[χη] = -3□χ` (kernel-pinned), the scalar sector becomes
  `K_χ □χ = V''₀ χ` with `K_χ = (F₀ + 3F₁²)/F₀`, so `m²_χ = V''₀/K_χ`.
- Graviton: a TT plane wave solves `G⁽¹⁾[h̄] = 0` iff its momentum is null
  (massless; non-null momentum is kernel-rejected). Two TT polarizations.
- No ghost: `F₀ > 0` (graviton sector) and `F₀ + 3F₁² > 0` (scalar sector).

The diagonalization is performed at the level of the equations of motion, so
no gauge choice enters the `K_χ` statement; TT enters only the mode-counting
statement.

**Status: implemented (eval 3p; `evals/eval3p_scalar_perturbation.py`,
template `pert_scalar_quadratic`).** Where eval 3s reads the scalar mass off a
flat-background Fourier analysis in SymPy, eval 3p derives the same physics
symbolically in Cadabra for a general fixed background. For the canonical scalar
sector `S = ∫√-g(-½(∇φ)² - V)`, it expands about `φ → φ̄ + χ` (with `χ` carrying
a smallness weight and `∇` inheriting it), projects onto the quadratic part with
`keep_weight`, and obtains `S₂ = ∫√-g(-½(∇χ)² - ½V''(φ̄)χ²)`. Two kernel checks,
both `noether-default-v1`: `δS₂/δχ` equals the documented operator
`√-g(□χ - V''(φ̄)χ)` (`residue_zero`), and linearizing the full nonlinear EOM
`□φ - V'` independently reproduces it (`linearized_eom_match`). So the linearized
Klein-Gordon mass is `m² = V''(φ̄)`, massless when `V'' = 0`, on any fixed
background. This scaffold now runs through the model-written derivation path:
`derive_perturbation` (and `kind="perturbation"` on the server, MCP, and web
clients) drives it for dynamical scalar fields, and `evals/test_eval_general.py`
gates that the orchestration reproduces this kernel-verified quadratic action
end to end.

### Eval 3g — graviton quadratic action

**Status: implemented (eval 3g; `evals/eval3g_graviton_perturbation.py`,
template `pert_metric_quadratic`).** Eval 3g is the spin-2 counterpart of
eval 3p. For pure gravity `S = ∫d⁴x √-g R` it expands about a flat background
`g → η + h` and keeps the quadratic part. With `ginv = η - h` (enough for the
Christoffels to second order) and `R⁽⁰⁾ = 0` on a flat background, the quadratic
Lagrangian is `L₂ = R⁽²⁾ + ½ h^γ_γ R⁽¹⁾`, whose variation is the linearized
vacuum Einstein equation `G⁽¹⁾_μν = 0`, the massless graviton. Two kernel checks,
both `noether-default-v1`: `δS₂/δh` matches the documented linearized Einstein
operator `-G⁽¹⁾` (`residue_zero`), and a `G⁽¹⁾` built independently from the
linearized Christoffels and Ricci tensor reproduces it (`linearized_eom_match`).
Two Cadabra mechanics matter here. A second derivative comes off the test field
in two `integrate_by_parts` passes, because the routine peels one derivative at
a time and `∇` is declared a `::Derivative`. And `meld` will not collapse equal
terms written at different index heights, so the scaffold lowers every index to
one explicit-`η` convention and rewrites `∇` as a commuting `::PartialDerivative`
before the equal-but-differently-written terms cancel.

### Eval 3a — Maxwell quadratic action

**Status: implemented (eval 3a; `evals/eval3a_maxwell_perturbation.py`,
template `pert_gauge_quadratic`).** Eval 3a is the spin-1 perturbation. For
`S = -¼∫d⁴x √-g F_μν F^μν` the potential splits `A_μ → Āμ + a_μ`, so the field
strength splits `F = F̄ + f` with the linearized strength
`f_μν = ∇_μ a_ν - ∇_ν a_μ`. Maxwell is already quadratic, so `keep_weight(eps=2)`
leaves the fluctuation action `-¼ √-g f_μν f^μν`, whose equation of motion is the
source-free linearized Maxwell operator `∇_μ f^μν = 0`, the wave operator behind
the photon's two transverse polarizations. Two kernel checks, both
`noether-default-v1`: `δS₂/δa` matches the documented operator `√-g ∇_μ f^μν`
(`residue_zero`), and linearizing the full nonlinear equation `∇_μ F^μν = 0`
reproduces it (`linearized_eom_match`). The cross-check reduces in a loop
(`eliminate_kronecker` / `canonicalise` / `meld`) until the differently-routed
`a` terms line up, since `∇` does not commute and `meld` will not melt terms at
different index heights.

### Eval 3y — Yang-Mills quadratic action

**Status: implemented (eval 3y; `evals/eval3y_yang_mills_perturbation.py`,
template `pert_yang_mills_quadratic`).** Eval 3y is the non-abelian sector. For
`S = -¼∫d⁴x √-g F^a_μν F^{a μν}` with
`F^a_μν = ∇_μ A^a_ν - ∇_ν A^a_μ + g f^{abc} A^b_μ A^c_ν`, the expansion about
`A → Ā + v` gives the quadratic action
`S₂ = -¼ √-g (f1^a_μν f1^{a μν} + 2 g f^{abc} F̄^a_μν v^b_μ v^c_ν)`, with the
background-covariant linearized strength `f1^a_μν = D̄_μ v^a_ν - D̄_ν v^a_μ` and
`D̄_μ v^a_ν = ∇_μ v^a_ν + g f^{abc} Ā^b_μ v^c_ν`. Its equation of motion is the
photon operator plus the gluon self-coupling to the background,
`D̄_μ f1^{a μν} + g f^{abc} v^b_μ F̄^{c μν} = 0`. Two kernel checks, both
`noether-default-v1`: `δS₂/δv` matches that documented operator (`residue_zero`),
and linearizing the full nonlinear Yang-Mills equation `D_μ F^{a μν} = 0`
reproduces it (`linearized_eom_match`). The adjoint indices form a second index
group with a Killing metric and `position=independent`, so the totally
antisymmetric structure constants contract and collapse; the fluctuation is
named `v`/`dv` to avoid clashing with the adjoint index `a`, and the structure
constant `fc` to avoid shadowing Python's `str`. The independent route keeps
weight `eps=2` (the test field `dv` carries `eps=1`) and then expands
`∇(Ā v)` by `product_rule` before the cross-check.

### Eval 3k — k-essence quadratic action

**Status: implemented (eval 3k; `evals/eval3k_kessence_perturbation.py`,
template `pert_kessence_quadratic`).** Eval 3k is the X-dependent scalar, where
the perturbation must expand `X` itself. For `S = ∫d⁴x √-g K(φ, X)` with
`X = -½(∇φ)²`, the fluctuation `φ → φ̄ + χ` carries through to
`δX = -∇φ̄·∇χ - ½(∇χ)²`, a linear and a quadratic piece. Taylor-expanding `K`
and keeping the quadratic part gives
`S₂ = ∫√-g(-½K_X(∇χ)² + ½K_XX(∇φ̄·∇χ)² - K_φX χ(∇φ̄·∇χ) + ½K_φφ χ²)`.
The two kinetic terms are the whole point: they combine into the effective
inverse metric `G^{ab} = K_X g^{ab} + K_XX ∇^aφ̄∇^bφ̄`, whose sound speed
`c_s² = K_X/(K_X + 2X̄K_XX)` is not 1 once `K_XX ≠ 0`. That kinetic mixing is
exactly what eval 3p (a plain scalar, `c_s² = 1`) does not have. The background
is taken with `∇∇φ̄ = 0` (covariantly constant gradient), the standard setup for
reading off `c_s²`; on it `∇X̄ = 0`, so the coupling gradients close under the
single chain rule `∇K_X → K_φX ∇φ̄` and its kin. Two kernel checks, both
`noether-default-v1`: `δS₂/δχ` matches the documented k-essence operator
`K_X □χ - K_XX ∇^aφ̄∇^bφ̄ ∇_a∇_bχ + …` (`residue_zero`), and linearizing the full
nonlinear equation `∇_a(K_X∇^aφ) + K_φ = 0` reproduces it
(`linearized_eom_match`).

The perturbation scaffolds now cover dynamical scalar fields (plain and
X-dependent), the metric, the metric-affine metric (with independent
connection), and rank-1 gauge potentials, abelian and non-abelian.
`derive_perturbation` selects by field kind, reads the `gauge_group` marker to
tell Maxwell from Yang-Mills and an `X`-dependent coupling to tell k-essence from
a plain scalar, and routes to the metric-affine scaffold when the connection is
independent. It refuses other field kinds (the rank-2 field strength, say)
rather than guessing.

### Eval 4ma — metric-affine (Palatini EH) quadratic perturbation

**Status: implemented (eval 4ma;
`evals/eval4ma_metric_affine_perturbation.py`,
template `pert_metric_affine_quadratic`).** Eval 4ma is the metric-affine
perturbation. For the Palatini Einstein-Hilbert action
`S = ∫d⁴x √-g g^{σν} R_{σν}(Γ)` it expands about a flat Minkowski background
with `Γ=0` in Cartesian coordinates, perturbing both
`g_{μν} → η_{μν} + h_{μν}` and `Γ^λ_{μν} → dG^λ_{μν}`, so the quadratic action
contains cross terms `h * dG` and `dG * dG` in addition to the standard graviton
terms. The connection fluctuation `dG` appears explicitly in the result,
capturing torsion and non-metricity modes. The linearized Palatini metric
equation is `R^{(1)}_{(αβ)}(dG) - ½ η_{αβ} R̃^{(1)}(dG) = 0`. Two kernel
checks, both `noether-default-v1 + metric-affine-v1`: `δS₂/δh` matches the
documented linearized Palatini metric operator (`residue_zero`), and
independently linearizing the full Palatini metric equation
`R_{(μν)} - ½ g_{μν} R̃ = 0` reproduces it (`linearized_eom_match`). The Ricci
scalar is built as a fully contracted scalar expression (not as `R_{σν}` with
free indices then contracted) to avoid a Cadabra free-index clash where the
derivative index in the second Palatini term conflicts with the contraction
index.

**Acceptance gating (VAL-PERT-006/007/008/014).** The concrete acceptance
case (Palatini EH around Minkowski) returns `verified=True` with both checks
True. The XOR condition holds: either the run is verified (both checks True) or
it is gated (`verified=False` with a non-empty `detail` naming the specific
failure mode). The detail distinguishes three failure modes: no residue check
(script did not run to completion), nonzero residue (candidate does not match
its own derivation), and residue zero with cross-check mismatch (the
independent linearized-EOM route disagrees). The SymPy component cross-check
on explicit random metric-affine backgrounds confirms the core physics claim
(the Ricci tensor of a general asymmetric connection is non-symmetric, the
linearized Palatini EOM has real metric-affine content, and the dG*dG part of
the Ricci scalar is nonzero), and verified is gated behind this cross-check
(the dual-gate invariant). At `T=Q=0`, the metric-affine perturbation path
reproduces the corresponding Levi-Civita result: the linearized Palatini
metric equation equals the linearized Einstein tensor `G^{(1)}_{μν}` (matching
the eval 3g operator), the Ricci of the LC connection is symmetric, and torsion
is zero. The pytest gates are in `tests/test_pert_metric_affine.py`.

---

## Eval 4 — Maxwell field on a fixed curved background

**Capabilities tested:** fixed-background vs. dynamical role separation;
antisymmetric field strength as a declared shorthand; gauge symmetry and its
Noether identity; not varying what must not be varied.

### Input

```latex
S = -\tfrac{1}{4}\int d^4x \sqrt{-g}\, F_{\mu\nu}F^{\mu\nu},
\qquad F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu
```

Dynamical: `A_μ` only. `g_{μν}` is a fixed background. Ask: equation of motion;
also ask Noether to state the gauge symmetry and its consequence.

### Expected elicitation

Mostly inference-and-confirm: `F` antisymmetric by construction;
`∂` vs `∇` equivalence inside `F` (Levi-Civita, torsion-free) noted; `g` fixed
(so no stress-tensor equation unless asked); `A_μ` the only variation target.

### Expected result

```latex
\nabla_\mu F^{\mu\nu} = 0
```

Gauge symmetry: `A_μ → A_μ + ∇_μ λ` leaves `F` (hence `S`) invariant; the
associated Noether identity `∇_ν∇_μF^{μν} ≡ 0` holds identically by
antisymmetry, which Noether must verify by kernel and present as the reason the
four equations contain only three independent ones.

### Verification checks

- V0/V1; result gauge covariant.
- V2: Noether identity `∇_ν∇_μF^{μν} ≡ 0` computed, not asserted.
- V3: flat-space limit reproduces `∂_μF^{μν} = 0`.
- Role discipline check (this eval's special point): the provenance must show
  **no** variation with respect to `g` was performed. A system that "helpfully"
  returns the Einstein-Maxwell equations fails the eval.

### Stretch task

User then promotes `g` to dynamical and adds `(2κ)^{-1}R`: Noether must reuse
the session state, vary the full action, and return Einstein-Maxwell with
`T_{μν} = F_μ{}^α F_{να} - ¼ g_{μν}F_{αβ}F^{αβ}`, plus the V2 check that
`T^μ{}_μ = 0` in d = 4 (computed, with the dimension dependence noted).

---

## Eval 5 — Gauss-Bonnet: dimension-dependent identities and good form

**Capabilities tested:** higher-curvature variation (the explosion stress
test); Bianchi-identity reduction until fixpoint; dimension-dependent
identities; honest dimension handling; canonical minimal form.

### Input

```latex
S = \int d^Dx \sqrt{-g}\,\left( R^2 - 4 R_{\mu\nu}R^{\mu\nu}
    + R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma} \right)
```

Dynamical: `g_{μν}`. Ask: equations of motion, first in general `D`, then in
`D = 4`.

### Expected elicitation

Dimension left symbolic (`D`) per the user's measure; conventions; Levi-Civita.
A sharp system recognizes the Gauss-Bonnet combination and says so.

### Expected result

General `D` (the Lanczos-Lovelock tensor):

```latex
H_{\mu\nu} = 2\Big( R\,R_{\mu\nu} - 2 R_{\mu\alpha}R^{\alpha}{}_{\nu}
  - 2 R^{\alpha\beta} R_{\mu\alpha\nu\beta}
  + R_{\mu}{}^{\alpha\beta\gamma} R_{\nu\alpha\beta\gamma} \Big)
  - \tfrac{1}{2} g_{\mu\nu} \mathcal{G} = 0,
\qquad
\mathcal{G} \equiv R^2 - 4R_{\alpha\beta}R^{\alpha\beta}
  + R_{\alpha\beta\gamma\delta}R^{\alpha\beta\gamma\delta}
```

The headline structural fact, which the system must surface rather than bury:
all `∇∇R`-type terms produced by varying each piece **cancel in the
combination** after Bianchi-identity reduction. `H_{μν}` is second order in
derivatives of the metric. A result left with uncancelled `□R_{μν}` or
`∇_μ∇_νR` terms fails good form even if formally equal.

In `D = 4`:

```latex
H_{\mu\nu} \equiv 0
```

identically, by a dimension-dependent identity (the Gauss-Bonnet term is
topological in four dimensions). Noether must return "identically zero", state
the reason, and not present `0 = 0` as a dynamical equation.

### Verification checks

- V0/V1; `H_{μν}` symmetric.
- V2: `∇^μ H_{μν} ≡ 0` identically (it is a Lovelock tensor), kernel-verified.
- V3: `D = 4` gives identical zero; `D = 5` gives a nonvanishing `H_{μν}`
  (spot-checked on a random 5-metric component evaluation).
- V4 (H2): cross-kernel canonical-form agreement for general `D`, where
  dimension-dependent identity handling differs most between kernels; this is
  the eval most likely to expose canonicalization gaps.

### Stretch task

Add `(2κ)^{-1}R + α𝒢` and ask for EOM in `D = 5`: the Einstein-Gauss-Bonnet
field equations; regression against the published Lovelock literature form.

---

## Eval 6 — Cubic Galileon scalar sector (Horndeski G3)

**Capabilities tested:** a coupling function multiplying `□φ`; variation that
splits `δ(K□φ)`; a second derivative peeled off the test field by a two-pass
integration by parts; the coupling chain rule when `□` lands on `K`.

**Status: implemented (eval 6; `evals/eval6_cubic_galileon.py`, template
`eom_cubic_galileon_scalar`).** This is the Horndeski G3 term `K(φ)□φ` added to a
canonical scalar, scoped to the scalar field equation. It is the first verified
member past the scalar-tensor sector of eval 3, and the action that motivated
the fidelity pass on `X` and subscripted couplings (see `docs/02_TECH_SPEC.md`
section 6.1).

### Input

```latex
S = \int d^4x \, \sqrt{-g}\, \left( -\tfrac12 \nabla_\mu\phi \nabla^\mu\phi
    - V(\phi) + K(\phi)\,\Box\phi \right)
```

User intent: `K(φ)` and `V(φ)` are arbitrary functions of `φ`, vary with respect
to `φ`. Ask: the scalar equation of motion.

### Result (noether-default-v1)

```
(1 + 2K'(φ))\,□φ + K''(φ)\,∇_μφ∇^μφ - V'(φ) = 0
```

### Derivation sketch

The canonical sector gives `□φ - V'`. The cubic term carries the new structure.
Varying `K(φ)□φ` splits by the product rule into `K'(φ)δφ □φ + K □(δφ)`. The
second piece has two covariant derivatives on `δφ`; integrating by parts twice
moves both onto `K`, leaving `(□K)δφ`. The chain rule expands the d'Alembertian
of the coupling, `□K = ∇_μ(K'∇^μφ) = K''∇_μφ∇^μφ + K'□φ`. Summing the two
pieces, the cubic term contributes `2K'□φ + K''(∇φ)²`, the braiding that adds to
the canonical `□φ - V'`.

### Verification

The audited Cadabra scaffold derives the variation, peels the second derivative
with a two-pass `integrate_by_parts` (seeded first on `∇_β(δφ)` to strip the
outer derivative, then on `δφ`), substitutes the coupling chain rule
`∇_μK → K'∇_μφ` and `∇_μK' → K''∇_μφ`, and checks the residue against the
independently stated target above. `residue_zero` is the kernel's verdict.
`evals/test_eval6.py` runs both the frozen template and the model-written path
(model stubbed to the audited script) end to end.

The metric sector now composes too. Varying `√-g G(φ)□φ` with respect to the
metric needs one new step beyond the eval-3 machinery: the second covariant
derivative of the field varies as `δ(∇_α∇_βφ) = -δΓ^λ_{αβ}∇_λφ`, handled in the
assembler through a symmetric stand-in `Hess_{αβ}` whose variation rule supplies
the `δΓ` term. After the two integration-by-parts passes and the coupling chain
rule `∇G → G_φ∇φ`, the `G∇∇φ` pieces cancel and the residue is checked against
the cubic stress tensor

```
E^{(3)}_{μν} = -G_φ ∇_μφ∇_νφ + ½ G_φ g_{μν}(∇φ)²,
```

the kinetic stress with coupling `-G_φ`, as expected from `G□φ ≡ -G_φ(∇φ)²` up
to a boundary term. `test_eval6.py` verifies the cubic Galileon coupled to
Einstein gravity (`R - ½(∇φ)² - V + G(φ)□φ`) this way, so the cubic block now
yields both equations of motion with no per-theory template.

---

## Eval 7 — k-essence and the general scalar Horndeski sector (composition)

**Capabilities tested:** an `X`-dependent coupling `K(φ,X)` (Horndeski G2);
expanding the shorthand `X` to its primitive inside the kernel and collapsing it
back for display; decomposing an additive Lagrangian into building blocks and
deriving the equation of motion without a per-theory template; refusing a term
that matches no block.

**Status: implemented (eval 7; `evals/eval7_kessence.py`,
`noether/kernels/cadabra/blocks.py`).** This is the first capability that does
not lean on a hand-written scaffold for the specific theory. The scalar
Lagrangian is decomposed into registered blocks (canonical kinetic, potential,
cubic Galileon, k-essence), one Cadabra script is assembled for the action the
user actually entered, and the kernel's residue check verifies it. k-essence
`K(φ,X)` is the block the fidelity pass left unverified, because `X` was never
expanded into `φ`-derivatives; the block now does that expansion in the kernel.

### Input

```latex
S = \int d^4x \, \sqrt{-g}\, \left( K(\phi, X) - V(\phi) + G(\phi)\,\Box\phi \right)
```

with `X = -½∇_μφ∇^μφ`. User intent: `K(φ,X)`, `V(φ)`, `G(φ)` are arbitrary
functions, vary with respect to `φ`. Ask: the scalar equation of motion.

### Result (noether-default-v1)

```
K_φ + K_X□φ + K_{Xφ}(∇φ)² - K_{XX}∇^μφ∇^νφ∇_μ∇_νφ
    - V_φ + 2G_φ□φ + G_{φφ}(∇φ)² = 0
```

The shorthands `K_X`, `K_{Xφ}`, `K_{XX}` are exactly the readability definitions
the system proposes for an `X`-dependent coupling (`orchestrator/definitions.py`,
which now also offers the mixed second derivative `K_{Xφ}`).

### Derivation sketch

Variation is linear, so the equation of motion of the sum is the sum of the
blocks' contributions:

- canonical kinetic `-½(∇φ)²` gives `□φ` (here folded into `K_X` at `K = X`),
- potential `-V(φ)` gives `-V_φ`,
- cubic `G(φ)□φ` gives `2G_φ□φ + G_{φφ}(∇φ)²` (eval 6),
- k-essence `K(φ,X)` gives `∇_μ(K_X∇^μφ) + K_φ`, which expands through
  `∇_μX = -∇_μ∇_νφ∇^νφ` to `K_φ + K_X□φ + K_{Xφ}(∇φ)² - K_{XX}∇^μφ∇^νφ∇_μ∇_νφ`.

### Verification

The decomposition (`decompose_scalar`) matches each additive term to a block and
reads off its coupling. The assembler (`assemble_scalar_eom_script`) emits one
Cadabra script that carries the real integrand (the exact couplings and
coefficients the parser found) and an independent candidate built from the same
blocks, then peels derivatives with the two-pass `integrate_by_parts`, expands
`X` and the coupling chain rules in the kernel, and checks the residue. This is
not a trusted sum of pre-blessed formulas: the kernel verifies the assembled
action, so a wrong block contribution would leave a nonzero residue. A term that
matches no block (an `X`-dependent curvature coupling `G(φ,X)R`, the Horndeski
G4 density) leaves the decomposition partial, and the caller falls back to the
model-written path or refuses, rather than guessing. `evals/test_eval7.py`
checks the full composition, the k-essence block alone, and the refusal. The
nonminimal `F(φ)R` term and the metric sector are eval 8; the higher Horndeski
densities (G4, G5) stay out of this path until they verify.

---

## Eval 8 — nonminimal scalar-tensor gravity by composition

**Capabilities tested:** a curvature-coupled block (`F(φ)R`); the metric
equation of motion derived compositionally, not only the scalar one; both
equations of motion of one theory from the same building blocks; refusing the
X-dependent curvature coupling that is Horndeski G4.

**Status: implemented (eval 8; `evals/eval8_nonminimal.py`,
`noether/kernels/cadabra/blocks.py`).** This is the same theory as eval 3, the
nonminimal scalar-tensor action, but derived with no per-theory template. It is
the first compositional capability that reaches the metric sector, so the
curvature-coupled `F(φ)R` block and the Einstein-Hilbert block join the scalar
blocks of eval 7.

### Input

```latex
S = \int d^4x \, \sqrt{-g}\, \left( F(\phi) R
    - \tfrac12 \nabla_\mu\phi \nabla^\mu\phi - V(\phi) \right)
```

User intent: `F(φ)` and `V(φ)` are arbitrary functions of `φ`, vary with respect
to both `g` and `φ`. Ask: both equations of motion.

### Result (noether-default-v1)

Scalar equation (vary `φ`):

```
F_φ R + □φ - V_φ = 0
```

Metric equation (vary `g`):

```
F R_{μν} - ½g_{μν}FR + g_{μν}□F - ∇_μ∇_νF
    - ½∇_μφ∇_νφ + ¼g_{μν}(∇φ)² + ½g_{μν}V = 0
```

### Derivation sketch

Both equations are sums of block contributions. The scalar equation adds the
nonminimal `F_φ R` (the `φ`-variation of `F(φ)R`, with `R` inert) to the
canonical `□φ - V_φ`. The metric equation runs the eval-3 metric machinery once
over the assembled integrand: vary `g^{αβ} → -h^{αβ}` and the Ricci tensor into
`δΓ`, substitute the connection variation, integrate by parts twice to peel the
two derivatives off `h`, and lower every `h` to one explicit-`g` convention
before comparing. Per block: Einstein-Hilbert `R` gives `G_{μν}`, nonminimal
`F(φ)R` gives `F G_{μν} + g_{μν}□F - ∇_μ∇_νF`, the kinetic term gives the scalar
stress, and the potential gives `½g_{μν}V`.

### Verification

`decompose_metric` matches each term to a metric block; `assemble_metric_eom_script`
emits one Cadabra script that varies the real action and residue-checks it
against a candidate built from the same blocks. The four metric blocks were each
confirmed against the kernel before wiring: Einstein-Hilbert alone returns the
Einstein tensor, and `F(φ)R + kinetic + potential` reproduces eval 3's residue.
`evals/test_eval8.py` runs both equations of motion through `derive_eom`, checks
the kernel residue is zero for each, and confirms a vacuum action (`R` alone)
verifies as `G_{μν} = 0`. An X-dependent curvature coupling `G(φ,X)R` (Horndeski
G4) matches no block and is refused. G4 and G5 stay out of this path until their
covariant equations of motion verify; they are held rather than added partially.

---

## Eval vector-affine — Maxwell on a metric-affine background

**Capabilities tested:** field-strength choice consequence (exterior derivative
vs covariant curl); hypermomentum from the covariant-curl choice; full-
connection divergence carrying T/Q contributions; the two definitions differ
exactly in the connection-equation source.

### Input

```latex
S = -\tfrac{1}{4}\int d^4x \sqrt{-g}\, F_{\mu\nu}F^{\mu\nu}
```

on an independent-connection background (torsion and non-metricity allowed).
Dynamical: `A_μ` and `Γ^λ_{μν}`. `g_{μν}` is a fixed background.

### Field-strength choices

1. **F = dA** (exterior derivative, the default):
   `F_{μν} = ∂_μ A_ν - ∂_ν A_μ`.  No Γ dependence.  EOM:
   `∇^{LC}_μ F^{μν} = 0`.  Hypermomentum: zero (no Γ dependence in the
   action).  When expressed with the full-connection divergence:
   `∇^{aff}_μ F^{μν} = (K^ρ_{ρμ} + L^ρ_{ρμ}) F^{μν} + ½ T^ν_{μρ} F^{μρ}`
   (T/Q correction terms present).

2. **F = ∇A** (covariant curl, elicited under independent connection):
   `F_{μν} = ∇_μ A_ν - ∇_ν A_μ = dA_{μν} - T^λ_{μν} A_λ`.  EOM:
   `(1/√{-g}) ∂_μ(√{-g} F^{μν}) + ½ T^ν_{μρ} F^{μρ} = 0`
   (full-connection divergence with torsion term).  Hypermomentum:
   `Δ^λ_{μν} = -2 A_λ F^{μν}` (nonzero, purely spin-type, antisymmetric in
   μ,ν).  This couples the gauge and connection sectors.

The two derivations differ exactly in the connection-equation source:
zero (dA) vs `−2AF` (covcurl).

### Verification checks

- Cadabra residue: dA EOM residue zero (nabla + LC-substitution approach);
  dA hypermomentum zero (no dG terms); covcurl hypermomentum nonzero (dG
  terms present).
- SymPy cross-check: affine-LC divergence difference equals T/Q correction
  on random affine backgrounds; dA hypermomentum zero structurally; covcurl
  hypermomentum antisymmetric and equal to `−2AF` via exact symbolic
  derivative; Euler-Lagrange decomposition of the covcurl EOM verified
  (explicit A-derivative = torsion term, ∂A-derivative = `−√{-g} F`).
- Covcurl EOM Cadabra residue: gated (mixed-index G terms that canonicalise
  cannot resolve, per cadabra-gotchas.md); SymPy Euler-Lagrange
  cross-check provides independent verification.

### Tests

`tests/test_vector_eom_affine.py` (31 tests: 3 Cadabra residue checks,
28 SymPy cross-checks on 3 random affine backgrounds).  Eval module:
`evals/eval_vector_affine.py`, `evals/test_eval_vector_affine.py`.

---

## Summary matrix

| Eval | Theory class | Distinctive demand | Horizon gate |
|---|---|---|---|
| 1 | GR (trace form) | composite expansion before variation | H1 |
| 2 | Palatini / metric-affine | independent Γ, torsion, projective freedom | H1 |
| 3 | scalar-tensor | multi-field, coupling functions, Bianchi link | H1 |
| 4 | gauge field on background | role discipline, Noether identity | H1 |
| 5 | Lovelock / Gauss-Bonnet | identity reduction to fixpoint, dimension handling | H2 |
| 1s | ADM of GR | foliation, constraint/evolution split | H2 |
| 3s | spectrum around Minkowski | perturbation, gauge, kinetic diagonalization | H2 |
| 3p | scalar quadratic action | symbolic quadratic expansion on a general background | H2 |
| 3g | graviton quadratic action | spin-2 expansion, linearized vacuum Einstein equation | H2 |
| 3a | Maxwell quadratic action | spin-1 expansion, source-free linearized Maxwell operator | H2 |
| 3y | Yang-Mills quadratic action | non-abelian expansion, background-covariant derivative, gluon self-coupling | H2 |
| 3k | k-essence quadratic action | X-expansion in the fluctuation, sound-speed kinetic mixing, coupling chain rule | H2 |
| 6 | cubic Galileon (Horndeski G3) | coupling times box phi, two-pass IBP, coupling chain rule, both EOMs by composition | H3 |
| 7 | k-essence / general scalar Horndeski | X-dependent coupling, block decomposition, no per-theory template | H3 |
| 8 | nonminimal scalar-tensor by composition | curvature block F(phi)R, metric EOM compositional, both EOMs, no template | H3 |
| fT | metric teleparallel f(T) | boundary-term identity T = -R + B, tetrad/Weitzenbock formulation, Cadabra residue-zero | H3 |
| vector-affine | Maxwell on metric-affine background | field-strength choice (dA vs nabla A), zero vs nonzero hypermomentum, T/Q correction | M3 |
| EC-algebraic | Einstein-Cartan algebraic torsion | Palatini connection EOM algebraic in K (no dK terms), SymPy cross-check on Q=0 T!=0 backgrounds | M3 |
| ST-affine | Palatini scalar-tensor F(phi)R(Gamma) | three variations (g, Gamma, phi), dF non-metricity source, boundary assumption, LC limit, genuine residue-zero check on connection EOM via Euler-Lagrange target | M3 |
| fQ | symmetric teleparallel f(Q) | coincident gauge, boundary-term identity Q = R + B (numerically verified), Cadabra residue-zero | H3 |

Each eval ships in two forms: this document (human-auditable worked target) and
an executable spec under `evals/` (machine-checkable: input transcript, expected
canonical forms, required check verdicts). The executable form is created when
implementation starts, and the worked solutions above get their kernel-verified
sign-pinning pass at that time.

### f(T) eval (verified, tetrad/Weitzenbock formulation)

**Action:** S = int sqrt(-g) f(T), where T is the torsion scalar (the
teleparallel equivalent of the Ricci scalar).

**Geometry:** teleparallel (curvature-free, metric-compatible, torsionful).
`ConnectionSpec` carries `curvature_free=True`, `family="teleparallel"`.

**Status: verified (tetrad/Weitzenbock formulation).** The linear f(T) = T
EOM is derived via the boundary-term identity T = -R + 2 nabla_mu T^mu,
which reduces the variation to the Einstein-Hilbert path (G_{mu nu} = 0).
The Cadabra template `eom_ft_linear_tetrad` exercises this path and passes
the residue check (residue_zero == True). The SymPy componentwise
cross-check confirms the EOM formula and the Weitzenbock geometry on a
metric-compatible torsionful (T!=0, Q=0) background with the
tetrad-teleparallel-v1 convention block. The Weitzenbock connection
Gamma^rho_{mu nu} = E_a^rho partial_mu e^a_nu is flat (R=0),
metric-compatible (Q=0), and torsionful (T!=0). For general f(T) the field
equation is
f'(T) [G_{mu nu} - (1/2) g_{mu nu} T] + S_{mu nu}^rho nabla_rho f'(T)
+ (1/2) g_{mu nu} [f(T) - T f'(T)] = 0,
where S^{rho mu nu} is the modified superpotential.

### f(Q) eval (verified, coincident gauge)

**Action:** S = int sqrt(-g) f(Q), where Q is the non-metricity scalar (the
symmetric teleparallel equivalent of the Ricci scalar).

**Geometry:** symmetric teleparallel (curvature-free, torsion-free,
non-metric). `ConnectionSpec` carries `curvature_free=True`,
`family="symmetric-teleparallel"`.

**Status: verified (coincident gauge).** The linear f(Q) = Q EOM is derived
in coincident gauge, where the flat torsion-free connection is set to zero so
that Q_{lambda mu nu} = partial_lambda g_{mu nu} and the action becomes a
pure-metric functional. By the boundary-term identity Q = R + boundary
(De, Loo, Saridakis 2023, eq 2.14; verified numerically by
`boundary_term_identity_residual` in `fq_coincident.py` on explicit
backgrounds with multiple seeds in dim=4), the f(Q) = Q action equals
S_EH + boundary, so the metric EOM is G_{mu nu} = 0. The non-metricity
scalar Q is computed as Q = Q_{alpha beta gamma} P^{alpha beta gamma}
using the standard superpotential P (Beltran Jimenez, Heisenberg, Koivisto
2018, eq 2.11). The Cadabra template `eom_fq_linear_coincident` exercises
this coincident-gauge variation and passes the residue check
(residue_zero == True). The SymPy componentwise cross-check confirms the
general f(Q) EOM formula on coincident-gauge backgrounds with Q != 0.
SymPy cross-checks also verify that Gamma = LC + L(Q) is torsion-free with
nonzero non-metricity, and that the curvature-free constraint is nontrivial
(arbitrary non-metricity does not produce R = 0).
