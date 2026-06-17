"""Eval 3k: quadratic-action expansion of k-essence (X-dependent scalar sector).

Input action (noether-default-v1, fixed background metric):
  S = \\int d^4x \\sqrt{-g} K(phi, X),   X = -1/2 (nabla phi)^2.

Expand about phi -> phibar + chi on a covariantly-constant-gradient background
(nabla nabla phibar = 0, so nabla Xbar = 0; the standard setup that reads off
the sound speed). The fluctuation of X splits into a linear and a quadratic
piece, dX = -nabla phibar . nabla chi - 1/2 (nabla chi)^2, and the second-order
Taylor expansion of K (kept at eps=2) is

  S2 = \\int sqrt(-g) ( -1/2 KX (nabla chi)^2 + 1/2 KXX (nabla phibar.nabla chi)^2
                        - KphiX chi (nabla phibar.nabla chi) + 1/2 Kphiphi chi^2 ).

The two kinetic terms combine into the effective inverse metric
  G^{ab} = KX g^{ab} + KXX nabla^a phibar nabla^b phibar,
whose timelike/spacelike ratio is the k-essence sound speed
  c_s^2 = KX / (KX + 2 Xbar KXX),   Xbar = -1/2 (nabla phibar)^2,
the genuinely new content over eval 3p (a plain scalar has c_s^2 = 1).

It is verified two independent ways inside the kernel:
  - delta S2 / delta chi matches the documented k-essence linearized operator
    KX box chi - KXX nabla^a phibar nabla^b phibar nabla_a nabla_b chi + ...
    (residue_zero);
  - linearizing the full nonlinear EOM nabla_a(KX nabla^a phi) + Kphi = 0
    reproduces it (linearized_eom_match).

The Cadabra scaffold is the frozen template `pert_kessence_quadratic`; the
pytest gate lives in evals/test_eval3k.py.
"""

CONVENTIONS = "noether-default-v1"

# Documented quadratic action, sound speed, and linearized EOM (for the human
# audit; the kernel derives and checks these, the strings are not used as input).
QUADRATIC_ACTION_TEX = (
    r"\int d^4x\,\sqrt{-g}\left("
    r"-\tfrac12 K_X (\nabla\chi)^2 + \tfrac12 K_{XX}(\nabla\bar\phi\cdot\nabla\chi)^2"
    r" - K_{\phi X}\,\chi\,(\nabla\bar\phi\cdot\nabla\chi)"
    r" + \tfrac12 K_{\phi\phi}\,\chi^2\right)"
)
SOUND_SPEED_TEX = r"c_s^2 = \frac{K_X}{K_X + 2\bar X\,K_{XX}}"
LINEARIZED_EOM_TEX = (
    r"K_X\,\Box\chi - K_{XX}\,\nabla^a\bar\phi\,\nabla^b\bar\phi\,"
    r"\nabla_a\nabla_b\chi + \dots = 0"
)
TEMPLATE = "pert_kessence_quadratic"
