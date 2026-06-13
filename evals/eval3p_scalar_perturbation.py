"""Eval 3p: quadratic-action expansion of a scalar field (perturbation sector).

Input action (noether-default-v1, fixed background metric):
  S = \\int d^4x \\sqrt{-g} ( -1/2 (nabla phi)^2 - V(phi) )

Expand about phi -> phibar + chi and keep the genuinely quadratic part:
  S2 = \\int d^4x \\sqrt{-g} ( -1/2 (nabla chi)^2 - 1/2 V''(phibar) chi^2 )

whose equation of motion is the linearized Klein-Gordon operator
  box chi - V''(phibar) chi = 0,   i.e.   m^2 = V''(phibar).

This is the symbolic counterpart of eval 3s, which read the same mass off a
flat-background Fourier analysis in SymPy. Here the mass term is derived for a
general fixed background by Cadabra, and verified two independent ways:
  - delta S2 / delta chi matches the documented operator (residue_zero);
  - linearizing the full nonlinear EOM (box phi - V') reproduces it
    (linearized_eom_match).

The Cadabra scaffold is the frozen template `pert_scalar_quadratic`; the
pytest gate lives in evals/test_eval3p.py.
"""

CONVENTIONS = "noether-default-v1"

# Documented quadratic action and linearized EOM (for the doc / human audit;
# the kernel derives and checks these, the strings are not used as input).
QUADRATIC_ACTION_TEX = (
    r"\int d^4x\,\sqrt{-g}\left("
    r"-\tfrac12 (\nabla\chi)^2 - \tfrac12 V''(\bar\phi)\,\chi^2\right)"
)
LINEARIZED_EOM_TEX = r"\Box\chi - V''(\bar\phi)\,\chi = 0"
TEMPLATE = "pert_scalar_quadratic"
