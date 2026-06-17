"""Eval 3a: quadratic-action expansion of the Maxwell field (gauge sector).

Input action (noether-default-v1, fixed background metric):
  S = -1/4 \\int d^4x \\sqrt{-g} F_{mu nu} F^{mu nu},  F = dA.

Split the potential into background plus fluctuation, A_mu -> Abar_mu + a_mu,
so the field strength splits F = Fbar + f with the linearized strength
f_{mu nu} = nabla_mu a_nu - nabla_nu a_mu. Maxwell is already quadratic, so the
fluctuation action is just -1/4 sqrt(-g) f_{mu nu} f^{mu nu}, whose equation of
motion is the source-free linearized Maxwell operator

  nabla_mu f^{mu nu} = 0,

the wave operator behind the photon's two transverse polarizations. It is
verified two independent ways inside the kernel:
  - delta S2 / delta a matches the documented operator sqrt(-g) nabla_mu f^{mu nu}
    (residue_zero);
  - linearizing the full nonlinear EOM nabla_mu F^{mu nu} = 0 reproduces it
    (linearized_eom_match).

The Cadabra scaffold is the frozen template `pert_gauge_quadratic`; the pytest
gate lives in evals/test_eval3a.py.
"""

CONVENTIONS = "noether-default-v1"

# Documented quadratic action and linearized EOM (for the human audit; the
# kernel derives and checks these, the strings are not used as input).
QUADRATIC_ACTION_TEX = (
    r"-\tfrac14 \int d^4x\,\sqrt{-g}\,f_{\mu\nu} f^{\mu\nu},\quad"
    r"f_{\mu\nu} = \nabla_\mu a_\nu - \nabla_\nu a_\mu"
)
LINEARIZED_EOM_TEX = r"\nabla_\mu f^{\mu\nu} = 0"
TEMPLATE = "pert_gauge_quadratic"
