# Pitfalls

These are the danger zones and known traps a contributor can fall into while working inside the gates. Each one states the trap, why it bites, and the file or rule that guards against it. None of these are theoretical: each has either caused a bug or is mechanically guarded because the failure mode is known.

## The torsion trap

A Levi-Civita shortcut applied to a metric-affine problem silently drops a torsion term and still reports a zero residue, because the dropped term was never present to leave a residue. The result looks verified and is wrong. Never shortcut a metric-affine derivation through Levi-Civita primitives. The dual gate (Cadabra residue check plus SymPy general-connection cross-check on explicit random backgrounds) is what catches the drop; both must agree before `verified` is set. See `docs/02_TECH_SPEC.md` section 6 and the dual-gate rationale in [./design-decisions.md](./design-decisions.md).

## Treating raising and lowering across nabla as free

V0 validation in `noether/npr/validate.py` is structural: it checks index balance and free-index agreement, not metric compatibility. Raising or lowering an index across `nabla` is only a no-op when the active connection is explicitly metric compatible. Under an independent connection with non-metricity, that move changes the expression. Do not assume it is free unless the NPR's connection is Levi-Civita or otherwise metric compatible. AGENTS.md section 8, item 3 states this as a working rule.

## Adding an AST node type without a cue case

`noether/orchestrator/elicit.py` `_detect_geometric_cues` walks the action AST with a match statement covering every current node type in `noether/npr/ast.py` (`Num`, `Sym`, `Func`, `Tensor`, `Deriv`, `Pow`, `Prod`, `Sum`). Adding a new node type to `ast.py` without adding the corresponding case raises `UnhandledASTNodeError` at runtime. This is fail-loud by design: there is no silent skip. The fix is to add the case, not to suppress the error. AGENTS.md section 6 documents the coupling.

## Bumping a kernel version pin without re-running the suite

`noether/kernels/versions.py` is the single source of truth for pinned kernel versions (`CADABRA_PINNED = "2.5.15"`, `SYMPY_PINNED = "1.14"`). The golden adapter tests and the audited cadabra templates were validated against these versions. Bumping either pin is a deliberate act: re-run the full eval suite and the cadabra golden tests, confirm every check is still green, and update this file in the same commit. Nothing else in the tree should hard-code a kernel version.

## Letting the model inject an answer

The bright line is that `verified` is set by the kernel, not by the model or the orchestrator. The model writes scripts; it does not write answers. If a contributor wires a model output directly into a result expression or sets `verified` from model confidence, the central promise of the product breaks. AGENTS.md rule 1 and the `ComputedResult` type in `noether/kernels/base.py` are the guard. See [./design-decisions.md](./design-decisions.md) on the provenance boundary.

## Silently guessing a convention

Every expression crossing a kernel boundary carries its convention block (signature, torsion sign, non-metricity definition, Ricci-contraction, contortion sign, disformation sign, K-sign, foliation/normal convention; for metric-affine NPRs also the field-strength definition). Under an independent connection, the Ricci-contraction and field-strength choices are elicited, not defaulted, because Ricci is then non-symmetric and the two field-strength definitions differ by a torsion term. Silently picking one produces a confident wrong answer. AGENTS.md rule 2 ("conventions are always explicit") and section 5 define the block; `noether/npr/conventions.py` is the authoritative source.

## Editing a prior NPR version

NPR versions are immutable and append-only. A late resolution creates a new version and marks prior results stale; it does not rewrite the version a result was computed against. Editing a prior version breaks the provenance receipt, which describes the problem the kernel actually saw. See [./design-decisions.md](./design-decisions.md) on immutable NPR versions.

## Presenting the Palatini connection as uniquely fixed

The pure Palatini Einstein-Hilbert connection equation has a projective mode: `Gamma = LC(g) + delta^lambda_nu A_mu` with `A_mu` arbitrary. The verified `eval2_palatini_connection` template surfaces this family (checks `solution_zero` and `ricci_shift_is_dA`) and never presents the connection as uniquely fixed. A contributor who reports only the Levi-Civita piece is dropping a verified degree of freedom. AGENTS.md rule 1 applies: do not assert a result narrower than the kernel computed.

## Trusting non-sentinel Cadabra stdout

Only the sentinel lines `NOETHER_RESULT:`, `NOETHER_CHECK:`, `NOETHER_DETAIL:`, and `NOETHER_CONVENTION:` count as kernel output. Everything else on Cadabra stdout is noise (banner, progress, diagnostics) and must never be parsed as a result or a check verdict. A contributor who greps stdout for a "0" or a LaTeX fragment is reading noise. The adapter parses the sentinels; product code should consume `ComputedResult`, not raw stdout. See [../systems/kernels/cadabra.md](../systems/kernels/cadabra.md).

## Caching, approximating, or truncating a symbolic answer

Correctness over speed, everywhere. A cache hit, a series truncation, or a parallel split that changes a symbolic term changes the answer, and a changed answer is a wrong answer no matter how fast it arrived. NORTH_STAR.md principle 1 ("correctness is sacred; speed is negotiable") and AGENTS.md rule 7 state this outright. Do not cache, approximate, truncate, or parallelize in any way that can alter a symbolic expression; if a cache must exist, key it on the full NPR version and convention block so a stale entry can never satisfy a different problem.
