"""Eval 3g: quadratic-action expansion of the metric (graviton perturbation).

Input action (noether-default-v1, flat background):
  S = \\int d^4x \\sqrt{-g} R

Expand about g_{mu nu} -> eta_{mu nu} + h_{mu nu} and keep the quadratic part.
With ginv = eta - h (enough for the Christoffels to second order) and
R^{(0)} = 0 on a flat background, the quadratic Lagrangian is

  L2 = R^{(2)} + 1/2 h^{gamma}_{gamma} R^{(1)},

whose variation is the linearized vacuum Einstein equation

  G^{(1)}_{mu nu} = 0,

i.e. the massless graviton. This is the spin-2 counterpart of eval 3p (scalar)
and the symbolic form of the graviton sector eval 3s reads off the Minkowski
spectrum.

The result is verified two independent ways inside the kernel:
  - delta S2 / delta h matches the documented operator -G^{(1)} (residue_zero);
  - the linearized Einstein tensor built separately from the linearized
    Christoffels and Ricci tensor reproduces it (linearized_eom_match).

The Cadabra scaffold is the frozen template `pert_metric_quadratic`; the
pytest gate lives in evals/test_eval3g.py.
"""

CONVENTIONS = "noether-default-v1"

# Documented quadratic action and linearized EOM (for the human audit; the
# kernel derives and checks these, the strings are not used as input).
QUADRATIC_ACTION_TEX = (
    r"\int d^4x\,\sqrt{-g}\left(R^{(2)} + \tfrac12 h^{\gamma}{}_{\gamma}\,R^{(1)}\right)"
)
LINEARIZED_EOM_TEX = r"G^{(1)}_{\mu\nu} = 0"
TEMPLATE = "pert_metric_quadratic"
