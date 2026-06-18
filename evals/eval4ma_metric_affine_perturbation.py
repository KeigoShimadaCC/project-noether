"""Eval 4ma: quadratic-action expansion of the Palatini metric-affine action.

Input action (noether-default-v1 + metric-affine-v1, flat background):
  S = \\int d^4x \\sqrt{-g} g^{\\sigma\\nu} R_{\\sigma\\nu}(\\Gamma)

Expand about g_{mu nu} -> eta_{mu nu} + h_{mu nu},
Gamma^{lambda}_{mu nu} -> dG^{lambda}_{mu nu} (background Gamma=0 in Cartesian
Minkowski) and keep the quadratic part. The connection fluctuation dG appears
explicitly in the result, capturing the torsion/non-metricity modes alongside
the metric fluctuation h.

The linearized Palatini metric equation is:
  R^{(1)}_{(mu nu)}(dG) - 1/2 eta_{mu nu} Rtilde^{(1)}(dG) = 0

The result is verified two independent ways inside the kernel:
  - delta S2 / delta h matches the documented linearized Palatini metric
    equation (residue_zero);
  - the same operator follows from independently linearizing the full
    Palatini metric equation (linearized_eom_match).

The Cadabra scaffold is the frozen template `pert_metric_affine_quadratic`;
the pytest gate lives in tests/test_pert_metric_affine.py.
"""

CONVENTIONS = "noether-default-v1 + metric-affine-v1"

QUADRATIC_ACTION_TEX = (
    r"\int d^4x\,\bigl[R^{(2)}_{\tilde{}}(\mathrm{d}G)"
    r" + \tfrac12 h^{\gamma}{}_{\gamma}\,R^{(1)}_{\tilde{}}(\mathrm{d}G)"
    r" - h^{\mu\nu}\,R^{(1)}_{\mu\nu}(\mathrm{d}G)\bigr]"
)
LINEARIZED_EOM_TEX = (
    r"R^{(1)}_{(\mu\nu)}(\mathrm{d}G)"
    r" - \tfrac12 \eta_{\mu\nu}\,\tilde{R}^{(1)}(\mathrm{d}G) = 0"
)
TEMPLATE = "pert_metric_affine_quadratic"
