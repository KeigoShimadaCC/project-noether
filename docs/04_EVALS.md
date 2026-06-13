# 04 — Evaluation suite: five action-to-result pairs

**Status:** stable; all five evals plus the stretch tasks 1s (ADM of GR),
3s (spectrum around Minkowski), and 3p (scalar quadratic action in Cadabra)
implemented and kernel-verified.
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
2. `R_{μν}(Γ)` depends on `Γ` only; the metric enters only through `√-g` and
   contractions. Note: `R_{μν}(Γ)` is **not** symmetric in general; only its
   symmetric part couples to `δg^{μν}`.
3. Conventions as default; ask: equations of motion for both fields.

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

- V0/V1 on both equations; metric equation symmetric.
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
end to end. The expansion is still scalar-only; other field kinds are refused
rather than guessed.

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

Each eval ships in two forms: this document (human-auditable worked target) and
an executable spec under `evals/` (machine-checkable: input transcript, expected
canonical forms, required check verdicts). The executable form is created when
implementation starts, and the worked solutions above get their kernel-verified
sign-pinning pass at that time.
