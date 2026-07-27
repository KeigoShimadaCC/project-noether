# Fun facts

Data collected on 2026-07-27.

## The eight-day sprint

The entire codebase, 112 commits and roughly 43,261 lines of Python, was written between 2026-06-12 02:07 and 2026-06-19 15:43. That is eight calendar days, with seven active commit days (2026-06-14 was quiet). The single busiest day, 2026-06-18, holds 39 commits, more than a third of the total, and most of those landed between midnight and noon as the metric-affine lift was built out through the night.

## Oldest surviving code

The oldest code in the repository is the initial commit at 2026-06-12 02:07. Several of the files it introduced are still load-bearing and largely unchanged in role: `NORTH_STAR.md`, the `noether/npr/` schema (`ast.py`, `conventions.py`, `latex.py`), the kernel base contract, the Cadabra adapter and its templates file, and the SymPy kernel adapter. The walking skeleton was not scrapped and rewritten; it was extended. If you `git blame` the convention block in `conventions.py`, the original lines trace back to day one, with later eras adding metric-affine fields around them rather than replacing them.

## The longest file

`noether/kernels/cadabra/templates.py` is 2,718 lines, the largest file in the repository by a wide margin. It holds the frozen golden Cadabra scripts that the kernel runs as sandboxed subprocesses, pinned to kernel versions. Because each template is a self-contained, kernel-verified derivation, the file reads more like an appendix of certified scripts than a normal module. A gentle refactoring hint: the templates are naturally grouped by task (vary, perturb, adm) and by theory family (Einstein-Hilbert, scalar-tensor, Galileon, metric-affine, vector-affine, teleparallel). Splitting the file along those seams into a `templates/` package would not change any behavior, since the registry is what callers hit, not the file path. No one is required to do this; the file works, and the 0 TODO count suggests nobody has felt the need.

## Zero debt markers

A grep for `TODO`, `FIXME`, and `HACK` across `noether/` returns zero matches. The source tree is unusually clean of explicit debt markers. This is a count of labeled debt only; it says nothing about implicit debt that was never tagged. Still, for a codebase written in eight days, the absence of any leftover "come back to this" notes is notable. The closest thing to a debt marker in the repo is the held-out note for the higher Horndeski G4(phi,X)R and G5 densities, which is documented as an intentional limitation rather than a TODO.

## Naming origins

- **Noether** is Emmy Noether, the mathematician whose theorem links continuous symmetries to conservation laws. The fit is direct: the tool varies actions to find equations of motion, and the variational principle is exactly where Noether's theorem lives.
- **NPR** stands for Noether Problem Representation, the kernel-agnostic intermediate form the orchestrator speaks. It is a playful echo of the news organization National Public Radio, but inside the codebase it is always the problem representation. The name appears throughout: the NPR schema, the NPR version history, the metric-affine NPR conventions.
- **Torsion trap** is the project's name for the silent-failure mode the dual gate catches. When a derivation is run under an independent connection with torsion, terms that vanish in the Levi-Civita limit can silently drop if a script assumes metric compatibility. The dual gate runs a convention sign falsifier that flips torsion and non-metricity signs and checks the result changes, catching derivations that secretly assumed them away.
