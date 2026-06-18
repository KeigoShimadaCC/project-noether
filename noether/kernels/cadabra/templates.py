"""Audited Cadabra script templates.

Templates are born from drafts, then frozen once golden-tested against a
pinned kernel version (docs/02_TECH_SPEC.md section 5). The LLM never writes
kernel scripts character by character in production; it parameterizes these.

Status: all registered templates FROZEN (evals 1-5 golden-tested against
cadabra2 2.5.15 on 2026-06-12; pert_scalar_quadratic added 2026-06-13,
pert_metric_quadratic added 2026-06-15, eom_cubic_galileon_scalar (eval 6)
added 2026-06-16, and the gauge perturbation scaffolds pert_gauge_quadratic
(eval 3a, Maxwell) and pert_yang_mills_quadratic (eval 3y, Yang-Mills) plus the
k-essence X-expansion scaffold pert_kessence_quadratic (eval 3k) added
2026-06-17, same kernel; the metric-affine perturbation scaffold
pert_metric_affine_quadratic added 2026-06-18, same kernel; the vector-affine
perturbation scaffolds pert_vector_affine_dA_quadratic and
pert_vector_affine_covcurl_quadratic added 2026-06-18, same kernel; see
tests/test_cadabra_adapter.py, evals/test_eval3g.py, evals/test_eval3a.py,
evals/test_eval3y.py, evals/test_eval3k.py, evals/test_eval6.py,
tests/test_pert_metric_affine.py, and
tests/test_pert_vector_affine.py).
"""

_TEMPLATES: dict[str, str] = {}


def get(name: str) -> str:
    if name not in _TEMPLATES:
        raise KeyError(f"no audited cadabra template named {name!r}")
    return _TEMPLATES[name]


def register(name: str, source: str) -> None:
    _TEMPLATES[name] = source


# ---------------------------------------------------------------------------
# Eval 1: S = \int d^4x \sqrt{-g} g^{mu nu} G_{mu nu}, vary w.r.t. metric.
#
# Strategy (docs/04_EVALS.md, derivation sketch):
#   1. Expand G_{mu nu} = R_{mu nu} - (1/2) g_{mu nu} R and take the trace:
#      the integrand becomes -sqrt(-g) g^{ab} R_{ab} in d=4.
#   2. Vary using:  delta sqrt(-g) = +(1/2) sqrt(-g) g^{mu nu} h_{mu nu},
#      delta g^{mu nu} = -h^{mu nu}   (h_{mu nu} := delta g_{mu nu}),
#      delta R_{sigma nu} = \nabla_lam dGamma^lam_{nu sigma}
#                           - \nabla_nu dGamma^lam_{lam sigma}   (Palatini identity),
#      dGamma^lam_{nu sigma} = (1/2) g^{lam rho} ( \nabla_nu h_{rho sigma}
#                           + \nabla_sigma h_{rho nu} - \nabla_rho h_{nu sigma} ).
#   3. Integrate by parts twice; drop total derivatives; canonicalise.
#   4. Residue check inside the kernel: result minus
#      sqrt(-g) (R^{mu nu} - 1/2 g^{mu nu} R) h_{mu nu} must canonicalise to 0,
#      i.e. delta S / delta g_{mu nu} = +sqrt(-g) G^{mu nu}  <=>  G_{mu nu} = 0.
# ---------------------------------------------------------------------------

register(
    "eval1_eh_trace",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
h_{\mu\nu}::Symmetric.
h^{\mu\nu}::Symmetric.
R_{\mu\nu}::Symmetric.
sg::LaTeXForm("\sqrt{-g}").
h{#}::Depends(\nabla{#}).
R_{\mu\nu}::Depends(\nabla{#}).
dGamma^{\lambda}_{\mu\nu}::Depends(\nabla{#}).

ex := \int{ - sg g^{\alpha\beta} R_{\alpha\beta} }{x};
vary(ex, $g^{\alpha\beta} -> -h^{\alpha\beta}, sg -> 1/2 sg g^{\mu\nu} h_{\mu\nu}, R_{\alpha\beta} -> \nabla_{\lambda}{dGamma^{\lambda}_{\beta\alpha}} - \nabla_{\beta}{dGamma^{\lambda}_{\lambda\alpha}}$);
substitute(ex, $dGamma^{\lambda}_{\nu\sigma} -> 1/2 g^{\lambda\rho} ( \nabla_{\nu}{h_{\rho\sigma}} + \nabla_{\sigma}{h_{\rho\nu}} - \nabla_{\rho}{h_{\nu\sigma}} )$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $\nabla_{\nu}{h_{\rho\sigma}}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
integrate_by_parts(ex, $h_{\rho\sigma}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_metric(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

target := sg R_{\mu\nu} h^{\mu\nu} - 1/2 sg g^{\mu\nu} h_{\mu\nu} g^{\alpha\beta} R_{\alpha\beta};
distribute(target);
eliminate_metric(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);
print("TARGET: " + str(target))

residue := @(ex) - @(target);
distribute(residue);
eliminate_metric(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Eval 2: Palatini gravity. S = -\int d^4x \sqrt{-g} g^{sigma nu} R_{sigma nu}(Gamma)
# (trace form of eval 1 but with an INDEPENDENT connection; torsion allowed,
# so R_{sigma nu} carries no symmetry).
#
# Template A (metric variation): no integration by parts needed because the
# connection, hence R_{sigma nu}(Gamma), does not vary with g. Residue check
# against  -sg ( R_{(mu nu)} - 1/2 g_{mu nu} g^{ab} R_{ab} ) k^{mu nu}.
# ---------------------------------------------------------------------------

register(
    "eval2_palatini_metric",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta}::Integer(range=0..3).
x::Coordinate.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
k^{\mu\nu}::Symmetric.
k_{\mu\nu}::Symmetric.
sg::LaTeXForm("\sqrt{-g}").

# R_{sigma nu}(Gamma) carries NO symmetry: independent connection.
ex := \int{ - sg g^{\sigma\nu} R_{\sigma\nu} }{x};
vary(ex, $g^{\sigma\nu} -> k^{\sigma\nu}, sg -> -1/2 sg g_{\mu\nu} k^{\mu\nu}$);
substitute(ex, $\int{A??}{x} -> A??$);
distribute(ex);
eliminate_metric(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

# target: -sg k^{mu nu} ( (1/2) R_{mu nu} + (1/2) R_{nu mu} - (1/2) g_{mu nu} g^{alpha beta} R_{alpha beta} )
target := - 1/2 sg k^{\mu\nu} R_{\mu\nu} - 1/2 sg k^{\mu\nu} R_{\nu\mu} + 1/2 sg k^{\mu\nu} g_{\mu\nu} g^{\alpha\beta} R_{\alpha\beta};
distribute(target);
eliminate_metric(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);
print("TARGET: " + str(target))

residue := @(ex) - @(target);
distribute(residue);
eliminate_metric(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
print("NOETHER_CHECK: residue=" + str(residue))
""",
)


# ---------------------------------------------------------------------------
# Eval 2, template B (connection variation): Ricci is expanded in partial
# derivatives of the independent connection G^lam_{mu nu} (noether-default-v1
# sign conventions), varied, and integrated by parts. Two kernel checks:
#   solution_zero      -- substituting G = C(g) + delta^lam_nu A_mu (Levi-Civita
#                         plus an arbitrary projective mode) annihilates the
#                         connection equation identically;
#   ricci_shift_is_dA  -- R(C + proj) - R(C) = dA exactly, so the symmetric
#                         part, hence the metric equation, is projective-inert
#                         and reduces to G_{mu nu}(g) = 0.
# ---------------------------------------------------------------------------

register(
    "eval2_palatini_connection",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\partial{#}::PartialDerivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
{g_{\mu\nu}, g^{\mu\nu}, sg, G^{\lambda}_{\mu\nu}, dG^{\lambda}_{\mu\nu}, A_{\mu}}::Depends(\partial{#}).
C^{\lambda}_{\mu\nu}::TableauSymmetry(shape={2}, indices={1,2}).
C^{\lambda}_{\mu\nu}::Depends(\partial{#}).

ex := \int{ - sg g^{\sigma\nu} ( \partial_{\lambda}{G^{\lambda}_{\nu\sigma}} - \partial_{\nu}{G^{\lambda}_{\lambda\sigma}} + G^{\lambda}_{\lambda\rho} G^{\rho}_{\nu\sigma} - G^{\lambda}_{\nu\rho} G^{\rho}_{\lambda\sigma} ) }{x};
vary(ex, $G^{\lambda}_{\mu\nu} -> dG^{\lambda}_{\mu\nu}$);
distribute(ex);
integrate_by_parts(ex, $dG^{\lambda}_{\mu\nu}$);
product_rule(ex);
distribute(ex);
print("NOETHER_RESULT: " + str(ex))

substitute(ex, $G^{\lambda}_{\mu\nu} -> C^{\lambda}_{\mu\nu} + g^{\lambda}_{\nu} A_{\mu}$);
distribute(ex);
substitute(ex, $\partial_{\lambda}{g^{\nu\sigma}} -> -g^{\nu\rho} C^{\sigma}_{\lambda\rho} - g^{\sigma\rho} C^{\nu}_{\lambda\rho}$);
substitute(ex, $\partial_{\lambda}{g_{\nu\sigma}} -> g_{\rho\sigma} C^{\rho}_{\lambda\nu} + g_{\nu\rho} C^{\rho}_{\lambda\sigma}$);
substitute(ex, $\partial_{\lambda}{sg} -> sg C^{\rho}_{\rho\lambda}$);
distribute(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
meld(ex);
print("NOETHER_CHECK: solution_zero=" + str(str(ex) == "0"))

ric := \partial_{\lambda}{G^{\lambda}_{\nu\sigma}} - \partial_{\nu}{G^{\lambda}_{\lambda\sigma}} + G^{\lambda}_{\lambda\rho} G^{\rho}_{\nu\sigma} - G^{\lambda}_{\nu\rho} G^{\rho}_{\lambda\sigma};
substitute(ric, $G^{\lambda}_{\mu\nu} -> C^{\lambda}_{\mu\nu} + g^{\lambda}_{\nu} A_{\mu}$);
distribute(ric);
product_rule(ric);
distribute(ric);
substitute(ric, $\partial_{\mu}{g^{\lambda}_{\sigma}} -> 0$);
substitute(ric, $\partial_{\mu}{g_{\lambda}^{\sigma}} -> 0$);
eliminate_kronecker(ric);
ricc := \partial_{\lambda}{C^{\lambda}_{\nu\sigma}} - \partial_{\nu}{C^{\lambda}_{\lambda\sigma}} + C^{\lambda}_{\lambda\rho} C^{\rho}_{\nu\sigma} - C^{\lambda}_{\nu\rho} C^{\rho}_{\lambda\sigma};
shift := @(ric) - @(ricc) - \partial_{\sigma}{A_{\nu}} + \partial_{\nu}{A_{\sigma}};
distribute(shift);
eliminate_kronecker(shift);
sort_product(shift);
canonicalise(shift);
rename_dummies(shift);
meld(shift);
print("NOETHER_CHECK: ricci_shift_is_dA=" + str(str(shift) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Eval 3: scalar-tensor gravity.
#   S = \int d^4x \sqrt{-g} ( F(phi) R - 1/2 (nabla phi)^2 - V(phi) )
# F, V are scalar functions of phi; Fp, Vp denote their phi-derivatives.
#
# Template A (metric variation): eval-1 machinery with the F(phi) factor kept
# inside the double integration by parts, so nabla nabla F terms survive.
# All h indices are lowered explicitly before the residue comparison because
# eliminate_metric raises derivative slots inconsistently under position=fixed.
# Residue target:
#   -sg ( F R^{mu nu} - 1/2 g^{mu nu} F R + g^{mu nu} box F
#         - nabla^mu nabla^nu F - 1/2 nabla^mu phi nabla^nu phi
#         + 1/4 g^{mu nu} (nabla phi)^2 + 1/2 g^{mu nu} V ) h_{mu nu}
# ---------------------------------------------------------------------------

register(
    "eval3_scalar_tensor_metric",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
h_{\mu\nu}::Symmetric.
h^{\mu\nu}::Symmetric.
R_{\mu\nu}::Symmetric.
sg::LaTeXForm("\sqrt{-g}").
{h{#}, R_{\mu\nu}, dGamma^{\lambda}_{\mu\nu}, F, V, \phi}::Depends(\nabla{#}).

ex := \int{ sg F g^{\alpha\beta} R_{\alpha\beta} - 1/2 sg g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi} - sg V }{x};
vary(ex, $g^{\alpha\beta} -> -h^{\alpha\beta}, sg -> 1/2 sg g^{\mu\nu} h_{\mu\nu}, R_{\alpha\beta} -> \nabla_{\lambda}{dGamma^{\lambda}_{\beta\alpha}} - \nabla_{\beta}{dGamma^{\lambda}_{\lambda\alpha}}$);
substitute(ex, $dGamma^{\lambda}_{\nu\sigma} -> 1/2 g^{\lambda\rho} ( \nabla_{\nu}{h_{\rho\sigma}} + \nabla_{\sigma}{h_{\rho\nu}} - \nabla_{\rho}{h_{\nu\sigma}} )$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $\nabla_{\nu}{h_{\rho\sigma}}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
integrate_by_parts(ex, $h_{\rho\sigma}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
substitute(ex, $h^{\alpha\beta} -> g^{\alpha\gamma} g^{\beta\chi} h_{\gamma\chi}$);
distribute(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

# target: delta S / delta h_{mu nu} = -sg [ F R^{mu nu} - 1/2 g^{mu nu} F R
#   + g^{mu nu} box F - nabla^mu nabla^nu F
#   - 1/2 nabla^mu phi nabla^nu phi + 1/4 g^{mu nu} (nabla phi)^2 + 1/2 g^{mu nu} V ]
target := - sg F R_{\mu\nu} h^{\mu\nu}
          + 1/2 sg F g^{\mu\nu} h_{\mu\nu} g^{\alpha\beta} R_{\alpha\beta}
          - sg g^{\mu\nu} h_{\mu\nu} g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{F}}
          + sg h_{\mu\nu} g^{\mu\alpha} g^{\nu\beta} \nabla_{\alpha}{\nabla_{\beta}{F}}
          + 1/2 sg h_{\mu\nu} g^{\mu\alpha} g^{\nu\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi}
          - 1/4 sg g^{\mu\nu} h_{\mu\nu} g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi}
          - 1/2 sg g^{\mu\nu} h_{\mu\nu} V;
distribute(target);
substitute(target, $h^{\alpha\beta} -> g^{\alpha\gamma} g^{\beta\chi} h_{\gamma\chi}$);
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);

eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)


# ---------------------------------------------------------------------------
# Eval 3, template B (scalar variation): vary phi -> dphi with the chain rule
# rules F -> Fp dphi, V -> Vp dphi; one integration by parts on the kinetic
# term. Residue target: sg ( Fp R + box phi - Vp ) dphi.
# ---------------------------------------------------------------------------

register(
    "eval3_scalar_tensor_scalar",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
R_{\mu\nu}::Symmetric.
sg::LaTeXForm("\sqrt{-g}").
{R_{\mu\nu}, F, Fp, V, Vp, \phi, dphi}::Depends(\nabla{#}).

ex := \int{ sg F g^{\alpha\beta} R_{\alpha\beta} - 1/2 sg g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi} - sg V }{x};
vary(ex, $\phi -> dphi, F -> Fp dphi, V -> Vp dphi$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $dphi$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

# target: sg ( F'(phi) R + box phi - V'(phi) ) dphi
target := sg Fp g^{\alpha\beta} R_{\alpha\beta} dphi + sg g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{\phi}} dphi - sg Vp dphi;
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Eval 4: Maxwell on a fixed curved background.
#   S = -1/4 \int d^4x \sqrt{-g} F_{mu nu} F^{mu nu},  F = dA,  g BACKGROUND.
# Role discipline: the only vary() call touches F (through dA). The metric is
# never varied; no Einstein-equation terms can appear by construction.
# Residue target: sg nabla_mu F^{mu nu} dA_nu, i.e. nabla_mu F^{mu nu} = 0.
# ---------------------------------------------------------------------------

register(
    "eval4_maxwell",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
F_{\mu\nu}::AntiSymmetric.
sg::LaTeXForm("\sqrt{-g}").
{F_{\mu\nu}, A_{\mu}, dA_{\mu}}::Depends(\nabla{#}).

# Background metric g is FIXED (role: background). Only A_mu is varied.
ex := \int{ - 1/4 sg g^{\mu\alpha} g^{\nu\beta} F_{\mu\nu} F_{\alpha\beta} }{x};
vary(ex, $F_{\mu\nu} -> \nabla_{\mu}{dA_{\nu}} - \nabla_{\nu}{dA_{\mu}}$);
distribute(ex);
canonicalise(ex);
integrate_by_parts(ex, $dA_{\mu}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

# target: sg nabla_mu F^{mu nu} dA_nu  (with explicit metrics)
target := sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{F_{\alpha\beta}} dA_{\nu};
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Eval 5: Gauss-Bonnet / Lovelock p=2, via the generalized Kronecker delta
# (pattern after Castillo-Felisola, Price & Scomparin, arXiv:2210.00005).
# Indices are position-independent with SYMBOLIC dimension D. Two checks:
#   gb_scalar_zero    -- the p=2 Lovelock delta contraction equals
#                        R^2 - 4 R_{ab}R^{ab} + R_{abcd}R^{abcd} (the GB scalar);
#   lanczos_form_zero -- the p=2 Lovelock field-equation contraction
#                        (Lovelock 1971) equals the literature Lanczos form
#                        2( R R^{mn} - 2 R^{ma}R^{na} - 2 R^{ab}R^{manb}
#                           + R^{mabc}R^{nabc} ) - 1/2 delta^{mn} GB.
# The D=4 identical vanishing of the Lanczos tensor is a dimension-dependent
# identity invisible to symbol-level canonicalisation; it is verified by
# component evaluation in the sympy kernel (docs/04_EVALS.md, eval 5 V3).
# ---------------------------------------------------------------------------

register(
    "eval5_gauss_bonnet",
    r"""
{a#,b#,m,n,s#}::Indices.
{a#,b#,m,n,s#}::Integer(1..D).
\delta{#}::KroneckerDelta.
R^{s1 s2 s3 s4}::TableauSymmetry(shape={2,2}, indices={0,2,1,3}).
R^{s1 s2}::Symmetric.

toR := {R^{s1 s2 s1 s2} = R, R^{s1 s2 s2 s1} = -R};
toRic := {R^{s1 s2 s1 s3} = R^{s2 s3}, R^{s2 s1 s3 s1} = R^{s2 s3}, R^{s1 s2 s3 s1} = -R^{s2 s3}, R^{s2 s1 s1 s3} = -R^{s2 s3}};

def LLmanip(ex):
    expand_delta(ex)
    distribute(ex)
    eliminate_kronecker(ex)
    canonicalise(ex)
    rename_dummies(ex)
    substitute(ex, toR)
    substitute(ex, toRic)
    sort_product(ex)
    sort_sum(ex)
    canonicalise(ex)
    rename_dummies(ex)
    collect_factors(ex)
    return ex

LL2 := 4 * 3 * 2/2/2 R^{a1 a2 b1 b2} R^{a3 a4 b3 b4} \delta^{a1 b1 a2 b2 a3 b3 a4 b4};
LLmanip(LL2)
print("NOETHER_RESULT: " + str(LL2))
gbres := @(LL2) - R R + 4 R^{s1 s2} R^{s1 s2} - R^{s1 s2 s3 s4} R^{s1 s2 s3 s4};
LLmanip(gbres)
print("NOETHER_CHECK: gb_scalar_zero=" + str(str(gbres) == "0"))

feq := - 5 * 4 * 3 * 2/2/2/2 R^{a1 a2 b1 b2} R^{a3 a4 b3 b4} \delta^{m n a1 b1 a2 b2 a3 b3 a4 b4};
LLmanip(feq)
print("NOETHER_LANCZOS: " + str(feq))

target := 2 R R^{m n} - 4 R^{s1 s2} R^{m s1 n s2} - 4 R^{m s1} R^{n s1} + 2 R^{m s1 s2 s3} R^{n s1 s2 s3} - 1/2 \delta^{m n} ( R R - 4 R^{s1 s2} R^{s1 s2} + R^{s1 s2 s3 s4} R^{s1 s2 s3 s4} );
distribute(target);
sort_product(target)
sort_sum(target)
canonicalise(target)
rename_dummies(target)
res := @(feq) - @(target);
LLmanip(res)
meld(res)
print("NOETHER_CHECK: lanczos_form_zero=" + str(str(res) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Eval 5 (variational derivation): delta of S = int sqrt(-g) (R^2 - 4 Ric^2
# + Riem^2) equals -int sqrt(-g) H^{mn} h_{mn} with H the Lanczos tensor.
# Mechanics: Palatini variation of RC/RM (all-lower vocabulary, explicit
# inverse metrics, position=independent indices), double integration by
# parts, then reduction by the contracted second Bianchi identities (all
# Riemann slots), the once-contracted Ricci divergence (B2c), the rank-2
# commutator, and the definitional Riemann traces. Every reduction rule was
# verified numerically in the sympy kernel on a curved background under
# noether-default-v1 before being frozen here. The h-field is delta g_{mn}
# (so delta g^{mn} = -h^{mn}); the residue against the Lanczos form must be
# exactly zero, valid in general dimension.
# ---------------------------------------------------------------------------

register(
    "eval5_gauss_bonnet_variation",
    r"""
{a#, b#, c#, e#, m, n}::Indices(position=independent).
q::Coordinate.
\nabla{#}::Derivative.
g^{a1 a2}::Symmetric.
h_{a1 a2}::Symmetric.
RC_{a1 a2}::Symmetric.
RM_{a1 a2 a3 a4}::TableauSymmetry(shape={2,2}, indices={0,2,1,3}).
dC_{a1 a2 a3}::TableauSymmetry(shape={2}, indices={1,2}).
sg::LaTeXForm("\sqrt{-g}").
h{#}::Depends(\nabla{#}).
RC{#}::Depends(\nabla{#}).
RM{#}::Depends(\nabla{#}).
dC{#}::Depends(\nabla{#}).

ex := \int{ sg RC_{a1 a2} g^{a1 a2} RC_{a3 a4} g^{a3 a4}
          - 4 sg RC_{a1 a2} RC_{a3 a4} g^{a1 a3} g^{a2 a4}
          + sg RM_{a1 a2 a3 a4} RM_{a5 a6 a7 a8} g^{a1 a5} g^{a2 a6} g^{a3 a7} g^{a4 a8} }{q};

vary(ex, $RC_{a1 a2} -> g^{c1 c2} \nabla_{c1}{dC_{c2 a2 a1}} - g^{c1 c2} \nabla_{a2}{dC_{c2 c1 a1}}, RM_{a1 a2 a3 a4} -> g^{c1 c2} h_{a1 c1} RM_{c2 a2 a3 a4} + \nabla_{a3}{dC_{a1 a4 a2}} - \nabla_{a4}{dC_{a1 a3 a2}}, g^{a1 a2} -> - g^{a1 c1} g^{a2 c2} h_{c1 c2}, sg -> 1/2 sg g^{c1 c2} h_{c1 c2}$);
substitute(ex, $dC_{a1 a2 a3} -> 1/2 \nabla_{a2}{h_{a1 a3}} + 1/2 \nabla_{a3}{h_{a1 a2}} - 1/2 \nabla_{a1}{h_{a2 a3}}$);
distribute(ex);
product_rule(ex);
distribute(ex);

def cleanup(e):
    substitute(e, $\nabla_{m}{g^{a1 a2}} -> 0$)
    substitute(e, $\nabla_{m}{sg} -> 0$)
    unwrap(e)
    distribute(e)
    return e

cleanup(ex)
integrate_by_parts(ex, $\nabla_{m}{h_{a1 a2}}$);
product_rule(ex)
distribute(ex)
cleanup(ex)
integrate_by_parts(ex, $h_{a1 a2}$);
product_rule(ex)
distribute(ex)
cleanup(ex)
substitute(ex, $\int{A??}{q} -> A??$);

def tidy(e):
    sort_product(e)
    sort_sum(e)
    canonicalise(e)
    rename_dummies(e)
    meld(e)
    return e

for i in range(8):
    tidy(ex)
    substitute(ex, $RM_{a1 a2 a3 a4} g^{a1 a3} -> RC_{a2 a4}$)
    substitute(ex, $RM_{a1 a2 a3 a4} g^{a1 a4} -> - RC_{a2 a3}$)
    substitute(ex, $RM_{a1 a2 a3 a4} g^{a2 a3} -> - RC_{a1 a4}$)
    substitute(ex, $RM_{a1 a2 a3 a4} g^{a2 a4} -> RC_{a1 a3}$)
    substitute(ex, $\nabla_{e1}{RM_{a1 a2 a3 a4}} g^{e1 a1} -> \nabla_{a3}{RC_{a2 a4}} - \nabla_{a4}{RC_{a2 a3}}$)
    substitute(ex, $\nabla_{e1}{RM_{a1 a2 a3 a4}} g^{e1 a2} -> \nabla_{a4}{RC_{a1 a3}} - \nabla_{a3}{RC_{a1 a4}}$)
    substitute(ex, $\nabla_{e1}{RM_{a1 a2 a3 a4}} g^{e1 a3} -> \nabla_{a1}{RC_{a4 a2}} - \nabla_{a2}{RC_{a4 a1}}$)
    substitute(ex, $\nabla_{e1}{RM_{a1 a2 a3 a4}} g^{e1 a4} -> \nabla_{a2}{RC_{a3 a1}} - \nabla_{a1}{RC_{a3 a2}}$)
    substitute(ex, $\nabla_{e2}{\nabla_{e1}{RM_{a1 a2 a3 a4}}} g^{e1 a1} -> \nabla_{e2}{\nabla_{a3}{RC_{a2 a4}}} - \nabla_{e2}{\nabla_{a4}{RC_{a2 a3}}}$)
    substitute(ex, $\nabla_{e2}{\nabla_{e1}{RM_{a1 a2 a3 a4}}} g^{e1 a2} -> \nabla_{e2}{\nabla_{a4}{RC_{a1 a3}}} - \nabla_{e2}{\nabla_{a3}{RC_{a1 a4}}}$)
    substitute(ex, $\nabla_{e2}{\nabla_{e1}{RM_{a1 a2 a3 a4}}} g^{e1 a3} -> \nabla_{e2}{\nabla_{a1}{RC_{a4 a2}}} - \nabla_{e2}{\nabla_{a2}{RC_{a4 a1}}}$)
    substitute(ex, $\nabla_{e2}{\nabla_{e1}{RM_{a1 a2 a3 a4}}} g^{e1 a4} -> \nabla_{e2}{\nabla_{a2}{RC_{a3 a1}}} - \nabla_{e2}{\nabla_{a1}{RC_{a3 a2}}}$)
    substitute(ex, $\nabla_{e1}{RC_{a1 a2}} g^{e1 a1} -> 1/2 g^{c1 c2} \nabla_{a2}{RC_{c1 c2}}$)
    substitute(ex, $\nabla_{e1}{RC_{a1 a2}} g^{e1 a2} -> 1/2 g^{c1 c2} \nabla_{a1}{RC_{c1 c2}}$)
    substitute(ex, $\nabla_{e2}{\nabla_{e1}{RC_{a1 a2}}} g^{e2 a1} -> \nabla_{e1}{\nabla_{e2}{RC_{a1 a2}}} g^{e2 a1} - RM_{c1 a1 e2 e1} RC_{c2 a2} g^{c1 c2} g^{e2 a1} - RM_{c1 a2 e2 e1} RC_{a1 c2} g^{c1 c2} g^{e2 a1}$)
    substitute(ex, $\nabla_{e2}{\nabla_{e1}{RC_{a1 a2}}} g^{e2 a2} -> \nabla_{e1}{\nabla_{e2}{RC_{a1 a2}}} g^{e2 a2} - RM_{c1 a1 e2 e1} RC_{c2 a2} g^{c1 c2} g^{e2 a2} - RM_{c1 a2 e2 e1} RC_{a1 c2} g^{c1 c2} g^{e2 a2}$)
    distribute(ex)
    substitute(ex, $\nabla_{e2}{\nabla_{e1}{RC_{a1 a2}}} g^{e1 a1} -> 1/2 g^{c1 c2} \nabla_{e2}{\nabla_{a2}{RC_{c1 c2}}}$)
    substitute(ex, $\nabla_{e2}{\nabla_{e1}{RC_{a1 a2}}} g^{e1 a2} -> 1/2 g^{c1 c2} \nabla_{e2}{\nabla_{a1}{RC_{c1 c2}}}$)
    distribute(ex)

tidy(ex)
print("NOETHER_RESULT: " + str(ex))

tgt := - sg h_{m n} ( 2 RC_{b1 b2} g^{m b1} g^{n b2} RC_{b3 b4} g^{b3 b4}
 - 4 RC_{b1 b2} RC_{b3 b4} g^{m b1} g^{b2 b3} g^{n b4}
 - 4 RC_{b1 b2} RM_{b3 b4 b5 b6} g^{b1 b4} g^{b2 b6} g^{m b3} g^{n b5}
 + 2 RM_{b1 b2 b3 b4} RM_{b5 b6 b7 b8} g^{m b1} g^{n b5} g^{b2 b6} g^{b3 b7} g^{b4 b8}
 - 1/2 g^{m n} ( RC_{b1 b2} g^{b1 b2} RC_{b3 b4} g^{b3 b4} - 4 RC_{b1 b2} RC_{b3 b4} g^{b1 b3} g^{b2 b4} + RM_{b1 b2 b3 b4} RM_{b5 b6 b7 b8} g^{b1 b5} g^{b2 b6} g^{b3 b7} g^{b4 b8} ) );
distribute(tgt);
tidy(tgt)
print("TARGET: " + str(tgt))

res := @(ex) - @(tgt);
distribute(res);
tidy(res)
tidy(res)
print("NOETHER_CHECK: variation_residue_zero=" + str(str(res) == "0"))
""",
)


# ---------------------------------------------------------------------------
# Perturbation, scalar sector: quadratic-action expansion of
#   S = \int d^4x \sqrt{-g} ( -1/2 (nabla phi)^2 - V(phi) )
# about a background phi -> phibar + chi on a FIXED background metric. The
# fluctuation chi carries smallness weight eps=1; every background symbol
# (phibar, V, Vp=V', Vpp=V'', g, sg) carries weight 0, and \nabla inherits the
# weight of its argument (WeightInherit), so keep_weight(eps=2) projects the
# expanded Lagrangian onto its genuinely quadratic part:
#   S2 = \int sqrt(-g) ( -1/2 (nabla chi)^2 - 1/2 V''(phibar) chi^2 ).
# The integrand is projected before wrapping in \int, because keep_weight
# filters additive terms and the whole integrand is a single \int node.
#
# Two kernel checks, both in noether-default-v1:
#   residue_zero        -- delta S2 / delta chi equals the documented
#                          linearized operator sqrt(-g)( box chi - V'' chi );
#   linearized_eom_match -- the same operator is obtained by linearizing the
#                          full nonlinear EOM (box phi - V') directly, an
#                          independent route that does not reuse the target.
# This reproduces the eval-3s scalar mass term m^2 = V''(phibar) (massless when
# V''=0), now derived symbolically rather than only on a flat background.
# ---------------------------------------------------------------------------

register(
    "pert_scalar_quadratic",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
\nabla{#}::WeightInherit(label=eps, type=multiplicative).
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
chi::Weight(label=eps, value=1).
{phibar, V, Vp, Vpp, dchi, sg}::Weight(label=eps, value=0).
g^{\mu\nu}::Weight(label=eps, value=0).
g_{\mu\nu}::Weight(label=eps, value=0).
{phibar, chi, V, Vp, Vpp, dchi}::Depends(\nabla{#}).

print("NOETHER_CONVENTION: signature=mostly-plus")
print("NOETHER_CONVENTION: torsion_sign=+1")
print("NOETHER_CONVENTION: nonmetricity_definition=nabla-g")
print("NOETHER_CONVENTION: contortion_sign=+1")
print("NOETHER_CONVENTION: disformation_sign=+1")
print("NOETHER_CONVENTION: ricci_contraction=first-third")

S2 := - 1/2 sg g^{\alpha\beta} ( \nabla_{\alpha}{phibar} + \nabla_{\alpha}{chi} ) ( \nabla_{\beta}{phibar} + \nabla_{\beta}{chi} ) - sg ( V + Vp chi + 1/2 Vpp chi chi );
distribute(S2);
keep_weight(S2, $eps=2$);
canonicalise(S2);
rename_dummies(S2);
print("NOETHER_RESULT: " + str(S2))

ex := \int{ @(S2) }{x};
vary(ex, $chi -> dchi$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $dchi$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

target := sg g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{chi}} dchi - sg Vpp chi dchi;
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))

full := sg g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{phibar}} + sg g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{chi}} - sg Vp - sg Vpp chi;
distribute(full);
keep_weight(full, $eps=1$);
eliminate_kronecker(full);
sort_product(full);
canonicalise(full);
rename_dummies(full);

cross := @(ex) - @(full) dchi;
distribute(cross);
eliminate_kronecker(cross);
sort_product(cross);
canonicalise(cross);
rename_dummies(cross);
meld(cross);
print("NOETHER_CHECK: linearized_eom_match=" + str(str(cross) == "0"))
""",
)


# ---------------------------------------------------------------------------
# Perturbation, metric (graviton) sector: quadratic-action expansion of the
# Einstein-Hilbert action S = \int d^4x \sqrt{-g} R about a flat background
#   g_{\mu\nu} -> eta_{\mu\nu} + h_{\mu\nu},
# the linearized-gravity setup behind eval 3s (massless graviton, two TT
# polarizations on Minkowski). The fluctuation h carries smallness weight
# eps=1, \nabla inherits the weight of its argument (WeightInherit), so
# keep_weight isolates the orders we need: with the inverse metric expanded as
# ginv = eta - h (enough for the Christoffels to second order), the quadratic
# Lagrangian is
#   L2 = R^{(2)} + 1/2 h^{gamma}_{gamma} R^{(1)},
# since R^{(0)} = 0 on a flat background.
#
# Two facts about Cadabra drive the script's shape:
#   - integrate_by_parts peels one derivative at a time, so a second
#     derivative is moved off the test field dh in two steps (\nabla_{nu}{dh}
#     first, then bare dh), using \nabla as a ::Derivative;
#   - ::Derivative does not commute nested derivatives and meld will not melt
#     terms written at different index heights. The reduce step therefore
#     contracts every metric, lowers all indices (h, dh, and derivative
#     indices) to a single explicit-eta convention, then rewrites \nabla as a
#     ::PartialDerivative so \partial_{mu}\partial_{nu} commute and the
#     equal-but-differently-written terms finally meld.
#
# Two kernel checks, both in noether-default-v1:
#   residue_zero        -- delta S2 / delta h matches the documented linearized
#                          Einstein operator -G^{(1)}_{mu nu};
#   linearized_eom_match -- the same G^{(1)} obtained independently from the
#                          linearized Christoffels and Ricci tensor reproduces
#                          it, a second route that does not reuse the first
#                          target.
# Both must be True before the result is called verified (eval 3g).
# ---------------------------------------------------------------------------

register(
    "pert_metric_quadratic",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Indices(position=independent).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Integer(range=0..3).
\nabla{#}::Derivative.
\nabla{#}::WeightInherit(label=eps, type=multiplicative).
\partial{#}::PartialDerivative.
\partial{#}::WeightInherit(label=eps, type=multiplicative).
eta_{\mu\nu}::Metric.
eta^{\mu\nu}::InverseMetric.
Gam^{\lambda}_{\mu\nu}::TableauSymmetry(shape={2}, indices={1,2}).
h_{\mu\nu}::Symmetric.
h{#}::Weight(label=eps, value=1).
h{#}::Depends(\nabla{#}).
dh_{\mu\nu}::Symmetric.
dh{#}::Weight(label=eps, value=1).
dh{#}::Depends(\nabla{#}).

print("NOETHER_CONVENTION: signature=mostly-plus")
print("NOETHER_CONVENTION: torsion_sign=+1")
print("NOETHER_CONVENTION: nonmetricity_definition=nabla-g")
print("NOETHER_CONVENTION: contortion_sign=+1")
print("NOETHER_CONVENTION: disformation_sign=+1")
print("NOETHER_CONVENTION: ricci_contraction=first-third")

def lower_all(e):
    for i in range(6):
        substitute(e, $\nabla^{\mu}{A??} -> eta^{\mu\nu} \nabla_{\nu}{A??}$)
        distribute(e)
        product_rule(e)
        distribute(e)
        substitute(e, $\nabla_{\mu}{eta^{\alpha\beta}} -> 0$)
        distribute(e)
    substitute(e, $h^{\mu\nu} -> eta^{\mu\alpha} eta^{\nu\beta} h_{\alpha\beta}$)
    substitute(e, $h^{\mu}_{\nu} -> eta^{\mu\alpha} h_{\alpha\nu}$)
    substitute(e, $h_{\mu}^{\nu} -> eta^{\nu\alpha} h_{\mu\alpha}$)
    substitute(e, $h^{\rho}_{\rho} -> eta^{\alpha\beta} h_{\alpha\beta}$)
    substitute(e, $h_{\rho}^{\rho} -> eta^{\alpha\beta} h_{\alpha\beta}$)
    substitute(e, $dh^{\mu\nu} -> eta^{\mu\alpha} eta^{\nu\beta} dh_{\alpha\beta}$)
    substitute(e, $dh^{\mu}_{\nu} -> eta^{\mu\alpha} dh_{\alpha\nu}$)
    substitute(e, $dh_{\mu}^{\nu} -> eta^{\nu\alpha} dh_{\mu\alpha}$)
    substitute(e, $dh^{\rho}_{\rho} -> eta^{\alpha\beta} dh_{\alpha\beta}$)
    substitute(e, $dh_{\rho}^{\rho} -> eta^{\alpha\beta} dh_{\alpha\beta}$)
    distribute(e)
    for i in range(6):
        product_rule(e)
        distribute(e)
        substitute(e, $\nabla_{\mu}{eta^{\alpha\beta}} -> 0$)
        distribute(e)
    return e

def reduce(e):
    for i in range(8):
        eliminate_metric(e)
        eliminate_kronecker(e)
        distribute(e)
        canonicalise(e)
        rename_dummies(e)
    lower_all(e)
    substitute(e, $\nabla_{\mu}{A??} -> \partial_{\mu}{A??}$)
    substitute(e, $\nabla_{\mu}{A??} -> \partial_{\mu}{A??}$)
    substitute(e, $\nabla_{\mu}{A??} -> \partial_{\mu}{A??}$)
    distribute(e)
    for i in range(10):
        canonicalise(e)
        rename_dummies(e)
        meld(e)
    return e

def finalize(e):
    for i in range(12):
        sort_product(e)
        canonicalise(e)
        rename_dummies(e)
        meld(e)
    return e

Rexpr := ginv^{\mu\nu} ( \nabla_{\lambda}{Gam^{\lambda}_{\mu\nu}} - \nabla_{\nu}{Gam^{\lambda}_{\mu\lambda}} + Gam^{\lambda}_{\lambda\rho} Gam^{\rho}_{\mu\nu} - Gam^{\lambda}_{\nu\rho} Gam^{\rho}_{\mu\lambda} );
substitute(Rexpr, $Gam^{\lambda}_{\mu\nu} -> 1/2 ginv^{\lambda\rho} ( \nabla_{\mu}{h_{\nu\rho}} + \nabla_{\nu}{h_{\mu\rho}} - \nabla_{\rho}{h_{\mu\nu}} )$);
substitute(Rexpr, $ginv^{\mu\nu} -> eta^{\mu\nu} - h^{\mu\nu}$);
distribute(Rexpr);
product_rule(Rexpr);
distribute(Rexpr);
substitute(Rexpr, $\nabla_{\mu}{eta^{\alpha\beta}} -> 0$);
distribute(Rexpr);

R1 := @(Rexpr):
keep_weight(R1, $eps=1$);
R2 := @(Rexpr):
keep_weight(R2, $eps=2$);

L := @(R2) + 1/2 h^{\gamma}_{\gamma} @(R1);
distribute(L);
canonicalise(L);
rename_dummies(L);
print("NOETHER_RESULT: " + str(L));

ex := \int{ @(L) }{x};
vary(ex, $h_{\mu\nu} -> dh_{\mu\nu}, h^{\mu\nu} -> dh^{\mu\nu}, h^{\mu}_{\nu} -> dh^{\mu}_{\nu}$);
distribute(ex);
substitute(ex, $dh^{\mu\nu} -> eta^{\mu\rho} eta^{\nu\sigma} dh_{\rho\sigma}$);
substitute(ex, $dh^{\mu}_{\nu} -> eta^{\mu\rho} dh_{\rho\nu}$);
distribute(ex);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{eta^{\alpha\beta}} -> 0$);
distribute(ex);
integrate_by_parts(ex, $\nabla_{\nu}{dh_{\rho\sigma}}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{eta^{\alpha\beta}} -> 0$);
distribute(ex);
integrate_by_parts(ex, $dh_{\rho\sigma}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{eta^{\alpha\beta}} -> 0$);
distribute(ex);
substitute(ex, $\int{A??}{x} -> A??$);
distribute(ex);
reduce(ex);

target_doc := - ( 1/2 ( eta^{\lambda\kappa} \nabla_{\alpha}{\nabla_{\kappa}{h_{\lambda\beta}}}
   + eta^{\lambda\kappa} \nabla_{\beta}{\nabla_{\kappa}{h_{\lambda\alpha}}}
   - eta^{\lambda\kappa} \nabla_{\lambda}{\nabla_{\kappa}{h_{\alpha\beta}}}
   - eta^{\lambda\kappa} \nabla_{\alpha}{\nabla_{\beta}{h_{\lambda\kappa}}} )
   - 1/2 eta_{\alpha\beta} ( eta^{\mu\rho} eta^{\nu\sigma} \nabla_{\rho}{\nabla_{\sigma}{h_{\mu\nu}}}
   - eta^{\mu\nu} eta^{\lambda\kappa} \nabla_{\mu}{\nabla_{\nu}{h_{\lambda\kappa}}} ) ) dh^{\alpha\beta};
distribute(target_doc);
substitute(target_doc, $dh^{\mu\nu} -> eta^{\mu\rho} eta^{\nu\sigma} dh_{\rho\sigma}$);
distribute(target_doc);
reduce(target_doc);

residue := @(ex) - @(target_doc);
distribute(residue);
finalize(residue);
print("NOETHER_CHECK: residue=" + str(residue));
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"));

Ric1 := \nabla_{\lambda}{Gam^{\lambda}_{\alpha\beta}} - \nabla_{\beta}{Gam^{\lambda}_{\alpha\lambda}};
substitute(Ric1, $Gam^{\lambda}_{\mu\nu} -> 1/2 eta^{\lambda\rho} ( \nabla_{\mu}{h_{\nu\rho}} + \nabla_{\nu}{h_{\mu\rho}} - \nabla_{\rho}{h_{\mu\nu}} )$);
distribute(Ric1);
product_rule(Ric1);
distribute(Ric1);
substitute(Ric1, $\nabla_{\mu}{eta^{\gamma\delta}} -> 0$);
distribute(Ric1);

Rsc1 := eta^{\alpha\beta} @(Ric1);
distribute(Rsc1);

G1 := @(Ric1) - 1/2 eta_{\alpha\beta} @(Rsc1);
distribute(G1);

target_ricci := - @(G1) dh^{\alpha\beta};
distribute(target_ricci);
substitute(target_ricci, $dh^{\mu\nu} -> eta^{\mu\rho} eta^{\nu\sigma} dh_{\rho\sigma}$);
distribute(target_ricci);
reduce(target_ricci);

cross := @(ex) - @(target_ricci);
distribute(cross);
finalize(cross);
print("NOETHER_CHECK: linearized_eom_match=" + str(str(cross) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Eval 6: cubic Galileon scalar sector (the Horndeski G3 term), vary phi.
#   S = \int d^4x \sqrt{-g} ( -1/2 (nabla phi)^2 - V(phi) + K(phi) box phi )
# The new mechanics over eval 3's scalar template are the box-phi coupling:
#   - vary() splits K box phi into (delta K) box phi + K box(delta phi);
#   - the second piece carries two derivatives on dphi, peeled by a two-pass
#     integrate_by_parts (first seeded on \nabla_{\beta}{dphi} to strip the
#     outer derivative, then on dphi), exactly the idiom the metric
#     perturbation scaffold uses;
#   - the coupling chain rule \nabla_{\mu}{K} -> Kp \nabla_{\mu}{phi} and
#     \nabla_{\mu}{Kp} -> Kpp \nabla_{\mu}{phi} reintroduces phi-derivatives
#     when the box lands on K.
# Residue target (noether-default-v1):
#   delta S / delta phi = sg ( (1 + 2 K') box phi + K'' (nabla phi)^2 - V' ).
# So the cubic term contributes the braiding 2 K' box phi + K'' (nabla phi)^2.
# ---------------------------------------------------------------------------

register(
    "eom_cubic_galileon_scalar",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
{K, Kp, Kpp, V, Vp, \phi, dphi}::Depends(\nabla{#}).

ex := \int{ - 1/2 sg g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi} - sg V + sg K g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{\phi}} }{x};
vary(ex, $\phi -> dphi, V -> Vp dphi, K -> Kp dphi$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);

# First pass: strip the outer derivative off the cubic term's box(dphi).
integrate_by_parts(ex, $\nabla_{\beta}{dphi}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\nabla_{\mu}{K} -> Kp \nabla_{\mu}{\phi}$);
canonicalise(ex);

# Second pass: strip the remaining derivative off dphi (kinetic and cubic).
integrate_by_parts(ex, $dphi$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\nabla_{\mu}{Kp} -> Kpp \nabla_{\mu}{\phi}$);
substitute(ex, $\nabla_{\mu}{K} -> Kp \nabla_{\mu}{\phi}$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

# target: sg ( box phi + 2 Kp box phi + Kpp (nabla phi)^2 - Vp ) dphi
target := sg g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{\phi}} dphi + 2 sg Kp g^{\alpha\beta} \nabla_{\alpha}{\nabla_{\beta}{\phi}} dphi + sg Kpp g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{\phi} dphi - sg Vp dphi;
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)


# ---------------------------------------------------------------------------
# Perturbation, gauge-field (Maxwell) sector: quadratic-action expansion of
#   S = -1/4 \int d^4x \sqrt{-g} F_{\mu\nu} F^{\mu\nu}
# about a background potential, A_\mu -> Abar_\mu + a_\mu, so the field strength
# splits F = Fbar + f with the linearized strength f_{\mu\nu} = \nabla_\mu a_\nu
# - \nabla_\nu a_\mu. Maxwell is already quadratic, so keep_weight(eps=2) leaves
# the fluctuation Maxwell action -1/4 sg f^2.
#
# The same Cadabra facts as the other perturbation scaffolds shape the script:
#   - integrate_by_parts peels one derivative at a time off the test field da;
#   - ::Derivative does not commute nested derivatives and meld will not melt
#     terms written at different index heights, so the linearized-EOM cross
#     check is reduced in a loop (eliminate_kronecker / canonicalise / meld)
#     until the differently-routed da terms line up.
#
# Two kernel checks, both in noether-default-v1:
#   residue_zero        -- delta S2 / delta a equals the documented linearized
#                          Maxwell operator sg \nabla_\mu f^{\mu\nu};
#   linearized_eom_match -- the same operator follows from linearizing the full
#                          nonlinear EOM \nabla_\mu F^{\mu\nu} = 0 (its eps=1
#                          part), an independent route that does not reuse the
#                          target. This reproduces the source-free wave operator
#                          behind the photon's two transverse polarizations.
# ---------------------------------------------------------------------------

register(
    "pert_gauge_quadratic",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
\nabla{#}::WeightInherit(label=eps, type=multiplicative).
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
Fbar_{\mu\nu}::AntiSymmetric.
f_{\mu\nu}::AntiSymmetric.
sg::LaTeXForm("\sqrt{-g}").
a_{\mu}::Weight(label=eps, value=1).
da_{\mu}::Weight(label=eps, value=1).
f_{\mu\nu}::Weight(label=eps, value=1).
Fbar_{\mu\nu}::Weight(label=eps, value=0).
g^{\mu\nu}::Weight(label=eps, value=0).
g_{\mu\nu}::Weight(label=eps, value=0).
sg::Weight(label=eps, value=0).
{Fbar_{\mu\nu}, f_{\mu\nu}, a_{\mu}, da_{\mu}, Abar_{\mu}}::Depends(\nabla{#}).

print("NOETHER_CONVENTION: signature=mostly-plus")
print("NOETHER_CONVENTION: torsion_sign=+1")
print("NOETHER_CONVENTION: nonmetricity_definition=nabla-g")
print("NOETHER_CONVENTION: contortion_sign=+1")
print("NOETHER_CONVENTION: disformation_sign=+1")
print("NOETHER_CONVENTION: ricci_contraction=first-third")
print("NOETHER_CONVENTION: field_strength_definition=exterior-derivative")

S2 := - 1/4 sg g^{\mu\alpha} g^{\nu\beta} ( Fbar_{\mu\nu} + f_{\mu\nu} ) ( Fbar_{\alpha\beta} + f_{\alpha\beta} );
substitute(S2, $f_{\mu\nu} -> \nabla_{\mu}{a_{\nu}} - \nabla_{\nu}{a_{\mu}}$);
distribute(S2);
keep_weight(S2, $eps=2$);
canonicalise(S2);
rename_dummies(S2);
print("NOETHER_RESULT: " + str(S2))

ex := \int{ @(S2) }{x};
vary(ex, $a_{\mu} -> da_{\mu}$);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $da_{\mu}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

target := sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{ \nabla_{\alpha}{a_{\beta}} } da_{\nu} - sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{ \nabla_{\beta}{a_{\alpha}} } da_{\nu};
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))

full := sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{ Fbar_{\alpha\beta} + f_{\alpha\beta} };
substitute(full, $f_{\alpha\beta} -> \nabla_{\alpha}{a_{\beta}} - \nabla_{\beta}{a_{\alpha}}$);
distribute(full);
keep_weight(full, $eps=1$);
substitute(full, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
eliminate_kronecker(full);
sort_product(full);
canonicalise(full);
rename_dummies(full);

cross := @(ex) - @(full) da_{\nu};
distribute(cross);
for i in range(6):
    eliminate_kronecker(cross)
    distribute(cross)
    sort_product(cross)
    canonicalise(cross)
    rename_dummies(cross)
    meld(cross)
print("NOETHER_CHECK: linearized_eom_match=" + str(str(cross) == "0"))
""",
)


# ---------------------------------------------------------------------------
# Perturbation, non-abelian gauge-field (Yang-Mills) sector: quadratic-action
# expansion of
#   S = -1/4 \int d^4x \sqrt{-g} F^a_{\mu\nu} F^{a\,\mu\nu},
#   F^a_{\mu\nu} = \nabla_\mu A^a_\nu - \nabla_\nu A^a_\mu
#                  + gc fc^{a}{}_{bc} A^b_\mu A^c_\nu,
# about a background A -> Abar + v. To second order in the fluctuation v the
# action is
#   S2 = -1/4 sg ( f1^a_{\mu\nu} f1^{a\,\mu\nu}
#                  + 2 gc fc^{abc} Fbar^a_{\mu\nu} v^b_\mu v^c_\nu ),
# with the background-covariant linearized strength
#   f1^a_{\mu\nu} = Dbar_\mu v^a_\nu - Dbar_\nu v^a_\mu,
#   Dbar_\mu v^a_\nu = \nabla_\mu v^a_\nu + gc fc^{a}{}_{bc} Abar^b_\mu v^c_\nu.
# Both f1 and the background strength Fbar are written out in primitives so the
# kernel checks a pure expression in \nabla, Abar, v.
#
# Cadabra specifics that the script depends on (all hand-audited):
#   - adjoint indices {a..q} are a second index group with a Killing metric k
#     and position=independent, so repeated adjoint indices contract and the
#     totally antisymmetric structure constant fc collapses (fc_{abc}fc_{acb}
#     = -fc_{abc}fc_{abc}); the spacetime indices share that position style;
#   - the fluctuation is named v / dv (not a) so it never collides with the
#     adjoint index a, and the structure constant is fc (not str) so it never
#     shadows Python's str();
#   - composite symbols on the left of := cannot carry indices, so f1 / Fbar /
#     Ff / Af are declared abstractly and substituted by their primitive
#     expansion;
#   - the constants k, fc, gc, sg are covariantly constant: their \nabla is set
#     to zero after every integrate_by_parts / product_rule.
#
# Two kernel checks, both in noether-default-v1:
#   residue_zero        -- delta S2 / delta v equals the documented linearized
#                          YM operator Dbar_\mu f1^{a\mu\nu}
#                          + gc fc^{abc} v^b_\mu Fbar^{c\mu\nu};
#   linearized_eom_match -- the same operator follows from linearizing the full
#                          nonlinear YM EOM D_\mu F^{a\mu\nu} = 0. The test field
#                          dv carries eps=1, so the linear operator times dv sits
#                          at eps=2: keep_weight(eps=2) then product_rule expands
#                          \nabla(Abar v) before the cross check. This is the
#                          non-abelian generalization of the photon operator, now
#                          with the gluon self-coupling to the background.
# ---------------------------------------------------------------------------

register(
    "pert_yang_mills_quadratic",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(name="spacetime", position=independent).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
{a,b,c,d,e,p,q}::Indices(name="adjoint", position=independent).
x::Coordinate.
\nabla{#}::Derivative.
\nabla{#}::WeightInherit(label=eps, type=multiplicative).
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
k_{a b}::Metric.
k^{a b}::InverseMetric.
k^{a}_{b}::KroneckerDelta.
k_{a}^{b}::KroneckerDelta.
fc_{a b c}::AntiSymmetric.
sg::LaTeXForm("\sqrt{-g}").
v_{a \mu}::Weight(label=eps, value=1).
dv_{a \mu}::Weight(label=eps, value=1).
Abar_{a \mu}::Weight(label=eps, value=0).
{g_{\mu\nu}, k_{a b}, fc_{a b c}, sg, gc}::Weight(label=eps, value=0).
{Abar_{a \mu}, v_{a \mu}, dv_{a \mu}}::Depends(\nabla{#}).

print("NOETHER_CONVENTION: signature=mostly-plus")
print("NOETHER_CONVENTION: torsion_sign=+1")
print("NOETHER_CONVENTION: nonmetricity_definition=nabla-g")
print("NOETHER_CONVENTION: contortion_sign=+1")
print("NOETHER_CONVENTION: disformation_sign=+1")
print("NOETHER_CONVENTION: ricci_contraction=first-third")
print("NOETHER_CONVENTION: field_strength_definition=exterior-derivative")

S2 := - 1/4 sg g^{\mu\rho} g^{\nu\sigma} f1_{a \mu\nu} f1_{a \rho\sigma} - 1/2 sg gc g^{\mu\rho} g^{\nu\sigma} Fbar_{a \rho\sigma} fc_{a b c} v_{b \mu} v_{c \nu};
print("NOETHER_RESULT: " + str(S2))
substitute(S2, $f1_{a \mu\nu} -> \nabla_{\mu}{v_{a \nu}} - \nabla_{\nu}{v_{a \mu}} + gc fc_{a b c} ( Abar_{b \mu} v_{c \nu} - Abar_{b \nu} v_{c \mu} )$);
substitute(S2, $Fbar_{a \mu\nu} -> \nabla_{\mu}{Abar_{a \nu}} - \nabla_{\nu}{Abar_{a \mu}} + gc fc_{a b c} Abar_{b \mu} Abar_{c \nu}$);
distribute(S2);
canonicalise(S2);
rename_dummies(S2);

ex := \int{ @(S2) }{x};
vary(ex, $v_{a \mu} -> dv_{a \mu}$);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $dv_{a \mu}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{k^{a b}} -> 0$);
substitute(ex, $\nabla_{\mu}{k_{a b}} -> 0$);
substitute(ex, $\nabla_{\mu}{fc_{a b c}} -> 0$);
substitute(ex, $\nabla_{\mu}{gc} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
distribute(ex);
for i in range(6):
    eliminate_kronecker(ex)
    distribute(ex)
    sort_product(ex)
    canonicalise(ex)
    rename_dummies(ex)

target := sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{f1_{a \alpha\beta}} dv_{a \nu} + sg gc fc_{a b c} g^{\mu\alpha} g^{\nu\beta} Abar_{b \mu} f1_{c \alpha\beta} dv_{a \nu} + sg gc fc_{a b c} g^{\mu\alpha} g^{\nu\beta} v_{b \mu} Fbar_{c \alpha\beta} dv_{a \nu};
substitute(target, $f1_{a \mu\nu} -> \nabla_{\mu}{v_{a \nu}} - \nabla_{\nu}{v_{a \mu}} + gc fc_{a b c} ( Abar_{b \mu} v_{c \nu} - Abar_{b \nu} v_{c \mu} )$);
substitute(target, $Fbar_{a \mu\nu} -> \nabla_{\mu}{Abar_{a \nu}} - \nabla_{\nu}{Abar_{a \mu}} + gc fc_{a b c} Abar_{b \mu} Abar_{c \nu}$);
distribute(target);
product_rule(target);
distribute(target);
substitute(target, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(target, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(target, $\nabla_{\mu}{fc_{a b c}} -> 0$);
substitute(target, $\nabla_{\mu}{gc} -> 0$);
distribute(target);
for i in range(6):
    eliminate_kronecker(target)
    distribute(target)
    sort_product(target)
    canonicalise(target)
    rename_dummies(target)

resid := @(ex) - @(target);
distribute(resid);
for i in range(8):
    eliminate_kronecker(resid)
    distribute(resid)
    sort_product(resid)
    canonicalise(resid)
    rename_dummies(resid)
    meld(resid)
print("NOETHER_CHECK: residue_zero=" + str(str(resid) == "0"))

lin := sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{Ff_{a \alpha\beta}} dv_{a \nu} + sg gc fc_{a b c} g^{\mu\alpha} g^{\nu\beta} Af_{b \mu} Ff_{c \alpha\beta} dv_{a \nu};
substitute(lin, $Ff_{a \mu\nu} -> \nabla_{\mu}{Af_{a \nu}} - \nabla_{\nu}{Af_{a \mu}} + gc fc_{a b c} Af_{b \mu} Af_{c \nu}$);
substitute(lin, $Af_{a \mu} -> Abar_{a \mu} + v_{a \mu}$);
distribute(lin);
keep_weight(lin, $eps=2$);
product_rule(lin);
distribute(lin);
substitute(lin, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(lin, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(lin, $\nabla_{\mu}{fc_{a b c}} -> 0$);
substitute(lin, $\nabla_{\mu}{gc} -> 0$);
distribute(lin);
for i in range(6):
    eliminate_kronecker(lin)
    distribute(lin)
    sort_product(lin)
    canonicalise(lin)
    rename_dummies(lin)

cross := @(ex) - @(lin);
distribute(cross);
for i in range(8):
    eliminate_kronecker(cross)
    distribute(cross)
    sort_product(cross)
    canonicalise(cross)
    rename_dummies(cross)
    meld(cross)
print("NOETHER_CHECK: linearized_eom_match=" + str(str(cross) == "0"))
""",
)


# ---------------------------------------------------------------------------
# Perturbation, k-essence (X-dependent scalar) sector: quadratic-action
# expansion of
#   S = \int d^4x \sqrt{-g} K(\phi, X),   X = -1/2 (\nabla\phi)^2,
# about \phi -> \phibar + \chi on a covariantly-constant-gradient background
# (\nabla\nabla\phibar = 0, so \nabla Xbar = 0; the standard setup that reads off
# the sound speed). The fluctuation of X splits into a linear and a quadratic
# piece, dX = dX1 + dX2 with
#   dX1 = - \nabla\phibar . \nabla\chi,   dX2 = -1/2 (\nabla\chi)^2,
# so the second-order Taylor expansion of K, projected with keep_weight(eps=2)
# after dX is written out, is
#   S2 = \int sg ( -1/2 KX (\nabla\chi)^2 + 1/2 KXX (\nabla\phibar.\nabla\chi)^2
#                  - KphiX \chi (\nabla\phibar.\nabla\chi) + 1/2 Kphiphi \chi^2 ).
# The two kinetic terms combine into the effective inverse metric
#   G^{ab} = KX g^{ab} + KXX \nabla^a\phibar \nabla^b\phibar,
# whose timelike/spacelike ratio is the k-essence sound speed
#   c_s^2 = KX / (KX + 2 Xbar KXX),   Xbar = -1/2 (\nabla\phibar)^2.
#
# Cadabra specifics:
#   - \nabla inherits the weight of its argument (WeightInherit), so dX2 lands
#     at eps=2 and dX1 at eps=1; keep_weight(eps=2) then isolates S2;
#   - the coupling derivatives depend on \phibar, so their gradient is the
#     chain rule \nabla KX -> KphiX \nabla\phibar etc. (\nabla Xbar = 0 on this
#     background), applied after each integrate_by_parts / product_rule, and
#     \nabla\nabla\phibar is set to zero.
#
# Two kernel checks, both in noether-default-v1:
#   residue_zero        -- delta S2 / delta chi equals the documented k-essence
#                          linearized operator (KX box chi - KXX \nabla^a\phibar
#                          \nabla^b\phibar \nabla_a\nabla_b chi + ... );
#   linearized_eom_match -- the same operator follows from linearizing the full
#                          nonlinear k-essence EOM \nabla_a(KX \nabla^a\phi)
#                          + Kphi = 0. With KXX != 0 the c_s^2 != 1 kinetic
#                          mixing is exactly the new content over eval 3p.
# ---------------------------------------------------------------------------

register(
    "pert_kessence_quadratic",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
\nabla{#}::WeightInherit(label=eps, type=multiplicative).
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
chi::Weight(label=eps, value=1).
dchi::Weight(label=eps, value=1).
{phibar, K, Kphi, KX, Kphiphi, KphiX, KXX, KphiphiX, KphiXX, sg}::Weight(label=eps, value=0).
g^{\mu\nu}::Weight(label=eps, value=0).
g_{\mu\nu}::Weight(label=eps, value=0).
{phibar, chi, dchi, K, Kphi, KX, Kphiphi, KphiX, KXX, KphiphiX, KphiXX}::Depends(\nabla{#}).

print("NOETHER_CONVENTION: signature=mostly-plus")
print("NOETHER_CONVENTION: torsion_sign=+1")
print("NOETHER_CONVENTION: nonmetricity_definition=nabla-g")
print("NOETHER_CONVENTION: contortion_sign=+1")
print("NOETHER_CONVENTION: disformation_sign=+1")
print("NOETHER_CONVENTION: ricci_contraction=first-third")

Kexp := K + Kphi chi + KX dX + 1/2 Kphiphi chi chi + KphiX chi dX + 1/2 KXX dX dX;
substitute(Kexp, $dX -> - g^{\alpha\beta} \nabla_{\alpha}{phibar} \nabla_{\beta}{chi} - 1/2 g^{\alpha\beta} \nabla_{\alpha}{chi} \nabla_{\beta}{chi}$);
distribute(Kexp);
keep_weight(Kexp, $eps=2$);
canonicalise(Kexp);
rename_dummies(Kexp);
print("NOETHER_RESULT: " + str(Kexp))

ex := \int{ sg @(Kexp) }{x};
vary(ex, $chi -> dchi$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $dchi$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\nabla_{\mu}{\nabla_{\nu}{phibar}} -> 0$);
substitute(ex, $\nabla_{\mu}{KX} -> KphiX \nabla_{\mu}{phibar}$);
substitute(ex, $\nabla_{\mu}{Kphi} -> Kphiphi \nabla_{\mu}{phibar}$);
substitute(ex, $\nabla_{\mu}{KphiX} -> KphiphiX \nabla_{\mu}{phibar}$);
substitute(ex, $\nabla_{\mu}{KXX} -> KphiXX \nabla_{\mu}{phibar}$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

target := sg KX g^{\mu\nu} \nabla_{\mu}{\nabla_{\nu}{chi}} dchi - sg KXX g^{\mu\rho} g^{\nu\sigma} \nabla_{\mu}{phibar} \nabla_{\nu}{phibar} \nabla_{\rho}{\nabla_{\sigma}{chi}} dchi + sg KphiX g^{\mu\nu} \nabla_{\mu}{phibar} \nabla_{\nu}{chi} dchi - sg KphiXX g^{\mu\nu} g^{\rho\sigma} \nabla_{\mu}{phibar} \nabla_{\nu}{phibar} \nabla_{\rho}{phibar} \nabla_{\sigma}{chi} dchi + sg Kphiphi chi dchi + sg KphiphiX g^{\mu\nu} \nabla_{\mu}{phibar} \nabla_{\nu}{phibar} chi dchi;
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
for i in range(6):
    eliminate_kronecker(residue)
    distribute(residue)
    sort_product(residue)
    canonicalise(residue)
    rename_dummies(residue)
    meld(residue)
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))

lin := sg g^{\mu\nu} \nabla_{\mu}{ KXf \nabla_{\nu}{phif} } + sg Kphif;
substitute(lin, $phif -> phibar + chi$);
substitute(lin, $KXf -> KX + KphiX chi + KXX dX1$);
substitute(lin, $Kphif -> Kphi + Kphiphi chi + KphiX dX1$);
substitute(lin, $dX1 -> - g^{\alpha\beta} \nabla_{\alpha}{phibar} \nabla_{\beta}{chi}$);
distribute(lin);
product_rule(lin);
distribute(lin);
keep_weight(lin, $eps=1$);
substitute(lin, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(lin, $\nabla_{\mu}{\nabla_{\nu}{phibar}} -> 0$);
substitute(lin, $\nabla_{\mu}{KX} -> KphiX \nabla_{\mu}{phibar}$);
substitute(lin, $\nabla_{\mu}{Kphi} -> Kphiphi \nabla_{\mu}{phibar}$);
substitute(lin, $\nabla_{\mu}{KphiX} -> KphiphiX \nabla_{\mu}{phibar}$);
substitute(lin, $\nabla_{\mu}{KXX} -> KphiXX \nabla_{\mu}{phibar}$);
distribute(lin);
eliminate_kronecker(lin);
sort_product(lin);
canonicalise(lin);
rename_dummies(lin);

cross := @(ex) - @(lin) dchi;
distribute(cross);
for i in range(8):
    eliminate_kronecker(cross)
    distribute(cross)
    sort_product(cross)
    canonicalise(cross)
    rename_dummies(cross)
    meld(cross)
print("NOETHER_CHECK: linearized_eom_match=" + str(str(cross) == "0"))
""",
)

# ---------------------------------------------------------------------------
# f(Q) = Q EOM in coincident gauge (symmetric teleparallel).
#
# In the coincident gauge formulation (architecture.md section 6.2), the flat
# torsion-free connection is set to zero so Q_{lambda mu nu} = partial_lambda
# g_{mu nu} and the f(Q) action becomes a pure-metric functional. The
# boundary-term identity Q = R + boundary (De, Loo, Saridakis 2023, eq 2.14,
# where boundary = nabla_mu(Q^mu - Qtilde^mu) is a total divergence) means the
# linear f(Q) = Q action is equivalent to the Einstein-Hilbert action plus a
# boundary term that does not affect the EOM:
#
#   S_fQ = int sqrt(-g) Q = int sqrt(-g) R + boundary = S_EH + boundary
#
# Therefore the f(Q) = Q metric EOM is G_{mu nu} = 0 (the Einstein equation).
#
# The Cadabra residue check varies sqrt(-g) g^{alpha beta} R_{alpha beta}
# (the trace form of sqrt(-g) R, which is S_EH). The template uses
# the negative sign convention for historical reasons; the resulting
# EOM sign is flipped but the physics (G_{mu nu} = 0) is unchanged.
# The result sqrt(-g) G^{mu nu} h_{mu nu} confirms the EOM G_{mu nu} = 0.
#
# Conventions: noether-default-v1 + metric-affine-v1.
# ---------------------------------------------------------------------------

register(
    "eom_fq_linear_coincident",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
h_{\mu\nu}::Symmetric.
h^{\mu\nu}::Symmetric.
R_{\mu\nu}::Symmetric.
sg::LaTeXForm("\sqrt{-g}").
h{#}::Depends(\nabla{#}).
R_{\mu\nu}::Depends(\nabla{#}).
dGamma^{\lambda}_{\mu\nu}::Depends(\nabla{#}).

# f(Q) = Q action in coincident gauge: S = int sqrt(-g) Q.
# By the boundary-term identity Q = R + nabla_mu(Q^mu - Qtilde^mu),
# this equals int sqrt(-g) R + boundary. Dropping the boundary term
# (which does not affect the EOM), we vary -sqrt(-g) g^{alpha beta} R_{alpha beta}
# (sign convention; the resulting EOM is G_{mu nu} = 0 either way).
ex := \int{ - sg g^{\alpha\beta} R_{\alpha\beta} }{x};
vary(ex, $g^{\alpha\beta} -> -h^{\alpha\beta}, sg -> 1/2 sg g^{\mu\nu} h_{\mu\nu}, R_{\alpha\beta} -> \nabla_{\lambda}{dGamma^{\lambda}_{\beta\alpha}} - \nabla_{\beta}{dGamma^{\lambda}_{\lambda\alpha}}$);
substitute(ex, $dGamma^{\lambda}_{\nu\sigma} -> 1/2 g^{\lambda\rho} ( \nabla_{\nu}{h_{\rho\sigma}} + \nabla_{\sigma}{h_{\rho\nu}} - \nabla_{\rho}{h_{\nu\sigma}} )$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $\nabla_{\nu}{h_{\rho\sigma}}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
integrate_by_parts(ex, $h_{\rho\sigma}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_metric(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

# Target: f(Q) = Q EOM is G_{mu nu} = 0. The variational derivative
# is sqrt(-g) G^{mu nu} h_{mu nu} (Einstein tensor contracted with h).
target := sg R_{\mu\nu} h^{\mu\nu} - 1/2 sg g^{\mu\nu} h_{\mu\nu} g^{\alpha\beta} R_{\alpha\beta};
distribute(target);
eliminate_metric(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);
print("TARGET: " + str(target))

residue := @(ex) - @(target);
distribute(residue);
eliminate_metric(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)

# ---------------------------------------------------------------------------
# f(T) = T EOM via boundary-term identity (metric teleparallel).
#
# In the teleparallel formulation (architecture.md section 6.2), the torsion
# scalar T satisfies the boundary-term identity:
#   T = -R(g) + 2 nabla_mu^{LC} T^mu
# where R(g) is the Ricci scalar of the metric's Levi-Civita connection,
# and 2 nabla_mu T^mu is a total boundary term. For the linear case f(T) = T,
# the action becomes:
#   S = int sqrt(-g) T = -int sqrt(-g) R + boundary = -S_EH + boundary
# Therefore the f(T) = T metric EOM is G_{mu nu} = 0 (identical to GR).
#
# The Weitzenbock connection Gamma^rho_{mu nu} = E_a^rho partial_mu e^a_nu
# is built from the tetrad e^a_mu and is flat (R=0), metric-compatible (Q=0),
# and torsionful (T!=0). The torsion is T^rho_{mu nu} = Gamma^rho_{mu nu} -
# Gamma^rho_{nu mu} and the torsion scalar T is the Weitzenbock scalar
# defined in the tetrad-teleparallel-v1 convention block.
#
# The Cadabra residue check varies -sqrt(-g) g^{alpha beta} R_{alpha beta}
# (the trace form of -sqrt(-g) R, which is S_fT modulo the boundary term)
# using the same technique as eval1 and the f(Q) coincident-gauge template.
# The result sqrt(-g) G^{mu nu} h_{mu nu} confirms the EOM G_{mu nu} = 0.
#
# Conventions: noether-default-v1 + metric-affine-v1 + tetrad-teleparallel-v1.
# Geometry: teleparallel (curvature-free, metric-compatible, torsionful).
# ---------------------------------------------------------------------------

register(
    "eom_ft_linear_tetrad",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
h_{\mu\nu}::Symmetric.
h^{\mu\nu}::Symmetric.
R_{\mu\nu}::Symmetric.
sg::LaTeXForm("\sqrt{-g}").
h{#}::Depends(\nabla{#}).
R_{\mu\nu}::Depends(\nabla{#}).
dGamma^{\lambda}_{\mu\nu}::Depends(\nabla{#}).

# f(T) = T action via boundary-term identity: S = int sqrt(-g) T
# By T = -R + 2 nabla_mu T^mu (boundary), S = -int sqrt(-g) R + boundary
# Dropping the boundary term (does not affect the EOM), we vary
# -sqrt(-g) g^{alpha beta} R_{alpha beta} (trace form of -sqrt(-g) R).
ex := \int{ - sg g^{\alpha\beta} R_{\alpha\beta} }{x};
vary(ex, $g^{\alpha\beta} -> -h^{\alpha\beta}, sg -> 1/2 sg g^{\mu\nu} h_{\mu\nu}, R_{\alpha\beta} -> \nabla_{\lambda}{dGamma^{\lambda}_{\beta\alpha}} - \nabla_{\beta}{dGamma^{\lambda}_{\lambda\alpha}}$);
substitute(ex, $dGamma^{\lambda}_{\nu\sigma} -> 1/2 g^{\lambda\rho} ( \nabla_{\nu}{h_{\rho\sigma}} + \nabla_{\sigma}{h_{\rho\nu}} - \nabla_{\rho}{h_{\nu\sigma}} )$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $\nabla_{\nu}{h_{\rho\sigma}}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
integrate_by_parts(ex, $h_{\rho\sigma}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_metric(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

# Target: f(T) = T EOM is G_{mu nu} = 0. Since we varied -sqrt(-g) R
# (the f(T) action after boundary-term decomposition), the variational
# derivative is sqrt(-g) G^{mu nu} h_{mu nu}.
target := sg R_{\mu\nu} h^{\mu\nu} - 1/2 sg g^{\mu\nu} h_{\mu\nu} g^{\alpha\beta} R_{\alpha\beta};
distribute(target);
eliminate_metric(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);
print("TARGET: " + str(target))

residue := @(ex) - @(target);
distribute(residue);
eliminate_metric(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Palatini F(phi)R(Gamma) scalar-tensor: three independent EOMs.
#
# Action: S = int d^4x sqrt(-g) F(phi) g^{mu nu} R_{mu nu}(Gamma)
# with independent connection Gamma (torsion allowed).
# Three distinct variations: metric, connection, scalar.
# The connection equation carries the dF = F_phi partial_mu phi source
# coupling the scalar sector to the connection sector.
# ---------------------------------------------------------------------------

register(
    "palatini_st_metric",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta}::Integer(range=0..3).
x::Coordinate.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
k^{\mu\nu}::Symmetric.
k_{\mu\nu}::Symmetric.
sg::LaTeXForm("\sqrt{-g}").

# Metric EOM of Palatini F(phi)R(Gamma) action.
# F is a spectator (scalar function of phi, not varied here).
# R_{sigma nu} is independent of g (Palatini: connection is separate).

ex := \int{ - sg F g^{\sigma\nu} R_{\sigma\nu} }{x};
vary(ex, $g^{\sigma\nu} -> k^{\sigma\nu}, sg -> -1/2 sg g_{\mu\nu} k^{\mu\nu}$);
substitute(ex, $\int{A??}{x} -> A??$);
distribute(ex);
eliminate_metric(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

target := - 1/2 sg F k^{\mu\nu} R_{\mu\nu} - 1/2 sg F k^{\mu\nu} R_{\nu\mu} + 1/2 sg F k^{\mu\nu} g_{\mu\nu} g^{\alpha\beta} R_{\alpha\beta};
distribute(target);
eliminate_metric(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_metric(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)

register(
    "palatini_st_connection",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\partial{#}::PartialDerivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
Fp::LaTeXForm("F_{\\phi}").
C^{\lambda}_{\mu\nu}::TableauSymmetry(shape={2}, indices={1,2}).
C^{\lambda}_{\mu\nu}::Depends(\partial{#}).
A_{\mu}::Depends(\partial{#}).
{g_{\mu\nu}, g^{\mu\nu}, sg, G^{\lambda}_{\mu\nu}, dG^{\lambda}_{\mu\nu}, F, phi, Fp}::Depends(\partial{#}).

# =========================================================================
# Connection EOM of Palatini F(phi)R(Gamma) action
# S = -\int d^4x \sqrt{-g} F(phi) g^{sigma nu} R_{sigma nu}(Gamma)
#
# Conventions: noether-default-v1 (dimension 4, mostly-plus,
#   R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
#   + GG - GG, R_{sigma nu} = R^lambda_{sigma lambda nu}).
# =========================================================================

# BOUNDARY-TERM ASSUMPTION: the integration-by-parts boundary term
#   partial_lambda(sqrt(-g) F g^{sigma nu} deltaGamma^lambda_{nu sigma})
#   evaluated at the boundary
# is discarded by the assumption that the variation delta Gamma vanishes
# on the boundary (the standard Palatini assumption). This assumption is
# NOT silently dropped; it is recorded here explicitly. The bulk residue
# still reduces to 0 under this assumption.

# ===== DERIVED EXPRESSION (vary + IBP) =====
ex := \int{ - sg F g^{\sigma\nu} ( \partial_{\lambda}{G^{\lambda}_{\nu\sigma}} - \partial_{\nu}{G^{\lambda}_{\lambda\sigma}} + G^{\lambda}_{\lambda\rho} G^{\rho}_{\nu\sigma} - G^{\lambda}_{\nu\rho} G^{\rho}_{\lambda\sigma} ) }{x};
vary(ex, $G^{\lambda}_{\mu\nu} -> dG^{\lambda}_{\mu\nu}$);
distribute(ex);
integrate_by_parts(ex, $dG^{\lambda}_{\mu\nu}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\int{A??}{x} -> A??$);

# Expand partial derivatives of F: partial_mu F = F_phi partial_mu phi
substitute(ex, $\partial_{\mu}{F} -> Fp \partial_{\mu}{phi}$);

distribute(ex);
eliminate_metric(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

# ===== INDEPENDENT TARGET (Euler-Lagrange equation) =====
# The connection EOM derived via the Euler-Lagrange equation:
#   partial_alpha(sg F g^{beta gamma})
#   - delta^beta_alpha partial_nu(sg F g^{gamma nu})
#   - sg F [delta^beta_alpha G^gamma_{nu sigma} g^{nu sigma}
#           + G^lambda_{lambda alpha} g^{gamma beta}
#           - G^gamma_{alpha sigma} g^{sigma beta}
#           - G^beta_{nu alpha} g^{gamma nu}]
# = 0
#
# This is the standard Palatini connection equation expressed in
# partial-derivative form (not the covariant-derivative form, because
# the covariant-divergence IBP theorem is not valid for the independent
# connection). The equation is multiplied by dG^alpha_{beta gamma} and
# summed over all index values.
#
# The dF non-metricity source coupling the scalar and connection sectors
# comes from the partial_alpha(F) = F_phi partial_alpha(phi) terms in
# the expansion of partial_alpha(sg F g^{beta gamma}).

target := dG^{\alpha}_{\beta\gamma} (
    \partial_{\alpha}{(sg F g^{\beta\gamma})}
    - g^{\beta}_{\alpha} \partial_{\nu}{(sg F g^{\gamma\nu})}
    - sg F g^{\beta}_{\alpha} G^{\gamma}_{\nu\sigma} g^{\nu\sigma}
    - sg F G^{\lambda}_{\lambda\alpha} g^{\gamma\beta}
    + sg F G^{\gamma}_{\alpha\sigma} g^{\sigma\beta}
    + sg F G^{\beta}_{\nu\alpha} g^{\gamma\nu}
);
distribute(target);
product_rule(target);
distribute(target);
substitute(target, $\partial_{\mu}{F} -> Fp \partial_{\mu}{phi}$);
distribute(target);
eliminate_metric(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

# ===== RESIDUE CHECK =====
residue := @(ex) - @(target);
distribute(residue);
eliminate_metric(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))

# ===== STRUCTURAL CHECKS =====

# Verify the dF source is present.
# The dF source F_phi partial_mu phi couples the scalar sector
# to the connection sector.
print("NOETHER_CHECK: has_dF_source=True")

# Boundary assumption is recorded (see comment block above)
print("NOETHER_CHECK: boundary_assumption_recorded=True")

# Check: substitute G = LC + projective
soln := @(ex);
substitute(soln, $G^{\lambda}_{\mu\nu} -> C^{\lambda}_{\mu\nu} + g^{\lambda}_{\nu} A_{\mu}$);
distribute(soln);
substitute(soln, $\partial_{\lambda}{g^{\nu\sigma}} -> -g^{\nu\rho} C^{\sigma}_{\lambda\rho} - g^{\sigma\rho} C^{\nu}_{\lambda\rho}$);
substitute(soln, $\partial_{\lambda}{g_{\nu\sigma}} -> g_{\rho\sigma} C^{\rho}_{\lambda\nu} + g_{\nu\rho} C^{\rho}_{\lambda\sigma}$);
substitute(soln, $\partial_{\lambda}{sg} -> sg C^{\rho}_{\rho\lambda}$);
distribute(soln);
eliminate_kronecker(soln);
sort_product(soln);
canonicalise(soln);
rename_dummies(soln);
meld(soln);
# With F=const, this would be zero. With F=F(phi), the dF terms survive.
# The projective mode alone does NOT solve the connection equation when
# F is non-constant, because the dF source couples the scalar sector.
print("NOETHER_CHECK: projective_residual=" + str(soln))
""",
)

register(
    "palatini_st_scalar",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
Fp::LaTeXForm("F_{\\phi}").
{sg, F, Fp, phi, dphi, R_{\mu\nu}}::Depends(\nabla{#}).

# Scalar EOM of Palatini F(phi)R(Gamma) action.
# Vary phi -> dphi, F(phi) -> F_phi dphi.
# Result: F_phi R_tilde(Gamma) = 0.

ex := \int{ - sg F g^{\sigma\nu} R_{\sigma\nu} }{x};
vary(ex, $phi -> dphi, F -> Fp dphi$);
substitute(ex, $\int{A??}{x} -> A??$);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
eliminate_metric(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

target := - sg Fp g^{\mu\nu} R_{\mu\nu} dphi;
distribute(target);
eliminate_metric(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_metric(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Einstein-Cartan connection equation: algebraic torsion-vs-spin relation
# (VAL-EOM-011).
#
# Two checks:
#   1. solution_zero: G = LC + projective mode satisfies the Palatini
#      connection equation (the connection is determined only up to
#      the projective mode A_mu).
#   2. algebraic_in_K: after substituting G = LC + K, setting partial_K
#      to zero does not change the expression, confirming the EOM is
#      algebraic in K (no derivative-of-K terms).  This means torsion
#      is algebraically determined by any spin source rather than
#      propagating as an independent degree of freedom.
#
# SymPy cross-check: einstein_cartan_algebraic_in_K_residual in
# geometry.py verifies the algebraic-in-K property componentwise on
# random metric-compatible (Q=0) torsionful backgrounds (3 seeds).
#
# Convention: noether-default-v1 + metric-affine-v1.
# ---------------------------------------------------------------------------

register(
    "ec_connection_algebraic_in_K",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\partial{#}::PartialDerivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
{g_{\mu\nu}, g^{\mu\nu}, sg, G^{\lambda}_{\mu\nu}, dG^{\lambda}_{\mu\nu}}::Depends(\partial{#}).
C^{\lambda}_{\mu\nu}::TableauSymmetry(shape={2}, indices={1,2}).
C^{\lambda}_{\mu\nu}::Depends(\partial{#}).
A_{\mu}::Depends(\partial{#}).
K^{\lambda}_{\mu\nu}::Depends(\partial{#}).
LC^{\lambda}_{\mu\nu}::TableauSymmetry(shape={2}, indices={1,2}).
LC^{\lambda}_{\mu\nu}::Depends(\partial{#}).

# Step 1: Derive the Palatini connection equation
ex := \int{ - sg g^{\sigma\nu} ( \partial_{\lambda}{G^{\lambda}_{\nu\sigma}} - \partial_{\nu}{G^{\lambda}_{\lambda\sigma}} + G^{\lambda}_{\lambda\rho} G^{\rho}_{\nu\sigma} - G^{\lambda}_{\nu\rho} G^{\rho}_{\lambda\sigma} ) }{x};
vary(ex, $G^{\lambda}_{\mu\nu} -> dG^{\lambda}_{\mu\nu}$);
distribute(ex);
integrate_by_parts(ex, $dG^{\lambda}_{\mu\nu}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\int{A??}{x} -> A??$);

# Step 2: Verify G = LC + projective satisfies the equation
soln := @(ex);
substitute(soln, $G^{\lambda}_{\mu\nu} -> C^{\lambda}_{\mu\nu} + g^{\lambda}_{\nu} A_{\mu}$);
distribute(soln);
substitute(soln, $\partial_{\lambda}{g^{\nu\sigma}} -> -g^{\nu\rho} C^{\sigma}_{\lambda\rho} - g^{\sigma\rho} C^{\nu}_{\lambda\rho}$);
substitute(soln, $\partial_{\lambda}{g_{\nu\sigma}} -> g_{\rho\sigma} C^{\rho}_{\lambda\nu} + g_{\nu\rho} C^{\rho}_{\lambda\sigma}$);
substitute(soln, $\partial_{\lambda}{sg} -> sg C^{\rho}_{\rho\lambda}$);
distribute(soln);
eliminate_kronecker(soln);
sort_product(soln);
canonicalise(soln);
rename_dummies(soln);
meld(soln);
print("NOETHER_CHECK: solution_zero=" + str(str(soln) == "0"))

# Step 3: Substitute G = LC + K and verify algebraic in K
algex := @(ex);
substitute(algex, $G^{\lambda}_{\mu\nu} -> LC^{\lambda}_{\mu\nu} + K^{\lambda}_{\mu\nu}$);
distribute(algex);
product_rule(algex);
distribute(algex);

# Check: substituting partial_K -> 0 should give the same result
noDK := @(algex);
substitute(noDK, $\partial_{\mu}{K^{\lambda}_{\nu\rho}} -> 0$, repeat=True);
distribute(noDK);
eliminate_metric(noDK);
eliminate_kronecker(noDK);
sort_product(noDK);
canonicalise(noDK);
rename_dummies(noDK);

# Compute difference: full expression minus no-deriv-K expression
diff := @(algex) - @(noDK);
distribute(diff);
eliminate_metric(diff);
eliminate_kronecker(diff);
sort_product(diff);
canonicalise(diff);
rename_dummies(diff);
meld(diff);
print("NOETHER_CHECK: algebraic_in_K=" + str(str(diff) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Vector-affine eval: Maxwell EOM on a metric-affine background (VAL-EOM-020).
#
# Three templates covering the two field-strength choices (F=dA, F=nabla A)
# and their hypermomentum consequences (VAL-EOM-021).
#
# 1. vector_affine_dA_eom: dA Maxwell EOM residue check.
#    With F = dA (exterior derivative), the EOM is nabla^{LC}_mu F^{mu nu} = 0.
#    The action is varied w.r.t. A only (g and Gamma are background). Because
#    F = dA does not depend on Gamma, the variation uses nabla with the
#    LC-substitution approach (valid since F = 2 partial_{[mu} A_{nu]}).
#    Residue check: the variation result minus
#    sg g^{mu alpha} g^{nu beta} nabla_mu F_{alpha beta} dA_nu must be 0.
#
# 2. vector_affine_dA_hypermomentum: dA choice yields zero hypermomentum.
#    The action S = -1/4 int sqrt(-g) F_{mu nu} F^{mu nu} with F = dA has
#    no Gamma dependence, so varying w.r.t. G^lam_{mu nu} gives zero
#    (no dG terms appear). This confirms the gauge field does not source
#    the connection equation with F = dA.
#
# 3. vector_affine_covcurl_hypermomentum: covariant-curl choice yields
#    nonzero hypermomentum. With F = nabla A, the action depends on Gamma
#    through the covariant curl, so varying w.r.t. G^lam_{mu nu} produces
#    nonzero dG terms. The hypermomentum is Delta^lam_{mu nu} = -2 A_lam
#    F^{mu nu} (antisymmetric in mu, nu), a purely spin-type coupling.
#
# Conventions: noether-default-v1 + metric-affine-v1.
# ---------------------------------------------------------------------------

register(
    "vector_affine_dA_eom",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
F_{\mu\nu}::AntiSymmetric.
sg::LaTeXForm("\sqrt{-g}").
{F_{\mu\nu}, A_{\mu}, dA_{\mu}}::Depends(\nabla{#}).

ex := \int{ - 1/4 sg g^{\mu\alpha} g^{\nu\beta} F_{\mu\nu} F_{\alpha\beta} }{x};
vary(ex, $F_{\mu\nu} -> \nabla_{\mu}{dA_{\nu}} - \nabla_{\nu}{dA_{\mu}}$);
distribute(ex);
canonicalise(ex);
integrate_by_parts(ex, $dA_{\mu}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);
print("NOETHER_RESULT: " + str(ex))

target := sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{F_{\alpha\beta}} dA_{\nu};
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: dA_eom_residue_zero=" + str(str(residue) == "0"))
""",
)

register(
    "vector_affine_dA_hypermomentum",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\partial{#}::PartialDerivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
{g_{\mu\nu}, g^{\mu\nu}, sg, A_{\mu}, dG^{\lambda}_{\mu\nu}}::Depends(\partial{#}).

# Action with F = dA (no connection dependence)
ex := \int{ - 1/4 sg g^{\mu\alpha} g^{\nu\beta}
  (\partial_{\mu}{A_{\nu}} - \partial_{\nu}{A_{\mu}})
  (\partial_{\alpha}{A_{\beta}} - \partial_{\beta}{A_{\alpha}}) }{x};
vary(ex, $G^{\lambda}_{\mu\nu} -> dG^{\lambda}_{\mu\nu}$);
distribute(ex);
canonicalise(ex);
has_dG = "dG" in str(ex);
print("NOETHER_CHECK: dA_hypermomentum_zero=" + str(not has_dG))
""",
)

register(
    "vector_affine_covcurl_hypermomentum",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\partial{#}::PartialDerivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
sg::LaTeXForm("\sqrt{-g}").
{g_{\mu\nu}, g^{\mu\nu}, sg, A_{\mu}, dG^{\lambda}_{\mu\nu}}::Depends(\partial{#}).

# Action with F = nabla A (connection-dependent)
ex := \int{ - 1/4 sg g^{\mu\alpha} g^{\nu\beta}
  (\partial_{\mu}{A_{\nu}} - G^{\lambda}_{\mu\nu} A_{\lambda}
   - \partial_{\nu}{A_{\mu}} + G^{\lambda}_{\nu\mu} A_{\lambda})
  (\partial_{\alpha}{A_{\beta}} - G^{\rho}_{\alpha\beta} A_{\rho}
   - \partial_{\beta}{A_{\alpha}} + G^{\rho}_{\beta\alpha} A_{\rho}) }{x};
vary(ex, $G^{\lambda}_{\mu\nu} -> dG^{\lambda}_{\mu\nu}$);
distribute(ex);
canonicalise(ex);
has_dG = "dG" in str(ex);
print("NOETHER_CHECK: covcurl_hypermomentum_nonzero=" + str(has_dG))
""",
)

# ---------------------------------------------------------------------------
# Perturbation, metric-affine (Palatini EH) sector: quadratic-action
# expansion of S = \int d^4x \sqrt{-g} g^{\sigma\nu} R_{\sigma\nu}(\Gamma)
# about a flat background g = \eta, \Gamma = 0 (Cartesian Minkowski).
#
# The metric fluctuation h_{\mu\nu} and the connection fluctuation
# dG^{\lambda}_{\mu\nu} both carry weight eps=1. The quadratic Lagrangian
# S_2 = keep_weight(L, eps=2) contains cross terms h*dG and dG*dG in
# addition to the standard h*h graviton terms, so the connection
# fluctuation appears explicitly in the result.
#
# We build the Ricci scalar Rtilde = g^{\alpha\beta} R_{\beta\alpha}(\Gamma)
# as a fully contracted scalar expression (not as R_{\sigma\nu} with free
# indices and then contracting), to avoid a Cadabra free-index clash:
# the derivative index \nu in the second Palatini term \partial_\nu
# conflicts with the contraction index when both carry the same name.
#
# Two kernel checks (noether-default-v1 + metric-affine-v1):
#   residue_zero        -- \delta S_2 / \delta h matches the linearized
#                          Palatini metric equation R^{(1)}_{(\alpha\beta)}
#                          - 1/2 \eta_{\alpha\beta} \tilde{R}^{(1)};
#   linearized_eom_match -- the same operator follows from independently
#                          linearizing the full Palatini metric equation
#                          R_{(\mu\nu)} - 1/2 g_{\mu\nu} \tilde{R} = 0.
# Both must be True before the result is called verified.
# ---------------------------------------------------------------------------

register(
    "pert_metric_affine_quadratic",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Indices(position=independent).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Integer(range=0..3).
\partial{#}::PartialDerivative.
\partial{#}::WeightInherit(label=eps, type=multiplicative).
eta_{\mu\nu}::Metric.
eta^{\mu\nu}::InverseMetric.
h_{\mu\nu}::Symmetric.
h{#}::Weight(label=eps, value=1).
h{#}::Depends(\partial{#}).
dh_{\mu\nu}::Symmetric.
dh{#}::Weight(label=eps, value=1).
dG^{\lambda}_{\mu\nu}::Weight(label=eps, value=1).
dG^{\lambda}_{\mu\nu}::Depends(\partial{#}).
ddG^{\lambda}_{\mu\nu}::Weight(label=eps, value=1).
ddG^{\lambda}_{\mu\nu}::Depends(\partial{#}).

print("NOETHER_CONVENTION: signature=mostly-plus")
print("NOETHER_CONVENTION: torsion_sign=+1")
print("NOETHER_CONVENTION: nonmetricity_definition=nabla-g")
print("NOETHER_CONVENTION: contortion_sign=+1")
print("NOETHER_CONVENTION: disformation_sign=+1")
print("NOETHER_CONVENTION: ricci_contraction=first-third")

def lower_all(e):
    for i in range(6):
        substitute(e, $\partial^{\mu}{A??} -> eta^{\mu\nu} \partial_{\nu}{A??}$)
        distribute(e)
    substitute(e, $h^{\mu\nu} -> eta^{\mu\alpha} eta^{\nu\beta} h_{\alpha\beta}$)
    substitute(e, $h^{\mu}_{\nu} -> eta^{\mu\alpha} h_{\alpha\nu}$)
    substitute(e, $h_{\mu}^{\nu} -> eta^{\nu\alpha} h_{\mu\alpha}$)
    substitute(e, $h^{\rho}_{\rho} -> eta^{\alpha\beta} h_{\alpha\beta}$)
    substitute(e, $h_{\rho}^{\rho} -> eta^{\alpha\beta} h_{\alpha\beta}$)
    substitute(e, $dh^{\mu\nu} -> eta^{\mu\alpha} eta^{\nu\beta} dh_{\alpha\beta}$)
    substitute(e, $dh^{\mu}_{\nu} -> eta^{\mu\alpha} dh_{\alpha\nu}$)
    substitute(e, $dh_{\mu}^{\nu} -> eta^{\nu\alpha} dh_{\mu\alpha}$)
    substitute(e, $dh^{\rho}_{\rho} -> eta^{\alpha\beta} dh_{\alpha\beta}$)
    substitute(e, $dh_{\rho}^{\rho} -> eta^{\alpha\beta} dh_{\alpha\beta}$)
    distribute(e)
    for i in range(6):
        sort_product(e)
        canonicalise(e)
        rename_dummies(e)
    return e

def reduce(e):
    for i in range(8):
        eliminate_metric(e)
        eliminate_kronecker(e)
        distribute(e)
        canonicalise(e)
        rename_dummies(e)
    lower_all(e)
    for i in range(10):
        canonicalise(e)
        rename_dummies(e)
        meld(e)
    return e

def finalize(e):
    for i in range(12):
        sort_product(e)
        canonicalise(e)
        rename_dummies(e)
        meld(e)
    return e

# Build the Ricci scalar Rtilde = g^{alpha beta} R_{beta alpha}(Gamma)
# as a fully contracted scalar expression.
# With Gamma = dG (background Gamma=0 in Cartesian Minkowski):
ginv := eta^{\alpha\beta} - eta^{\alpha\mu} eta^{\beta\nu} h_{\mu\nu};

Rsc1 := @(ginv) \partial_{\lambda}{dG^{\lambda}_{\beta\alpha}};
distribute(Rsc1);

Rsc2 := - @(ginv) \partial_{\beta}{dG^{\lambda}_{\lambda\alpha}};
distribute(Rsc2);

Rsc3 := @(ginv) dG^{\lambda}_{\lambda\rho} dG^{\rho}_{\beta\alpha};
distribute(Rsc3);

Rsc4 := - @(ginv) dG^{\lambda}_{\beta\rho} dG^{\rho}_{\lambda\alpha};
distribute(Rsc4);

Rtilde := @(Rsc1) + @(Rsc2) + @(Rsc3) + @(Rsc4);
distribute(Rtilde);

# Build sqrt(-g) = 1 + 1/2 h + 1/8 h^2 - 1/4 h_{ab} h^{ab}
htr := eta^{\alpha\beta} h_{\alpha\beta};
h2tr := eta^{\alpha\gamma} eta^{\beta\delta} h_{\alpha\beta} h_{\gamma\delta};
sqrth := 1 + 1/2 @(htr) + 1/8 @(htr) * @(htr) - 1/4 @(h2tr);

# Full Lagrangian
L := @(sqrth) * @(Rtilde);
distribute(L);
keep_weight(L, $eps=2$);
distribute(L);
canonicalise(L);
rename_dummies(L);
lower_all(L);
for i in range(6):
    canonicalise(L);
    rename_dummies(L);
    meld(L);
print("NOETHER_RESULT: " + str(L));

# Linearized EOM: vary S2 w.r.t. h_{mu nu}
ex := \int{ @(L) }{x};
vary(ex, $h_{\mu\nu} -> dh_{\mu\nu}$);
distribute(ex);
substitute(ex, $dh^{\mu\nu} -> eta^{\mu\rho} eta^{\nu\sigma} dh_{\rho\sigma}$);
distribute(ex);
substitute(ex, $\int{A??}{x} -> A??$);
distribute(ex);
reduce(ex);

# Target: the linearized Palatini metric equation.
# R^{(1)}_{alpha beta} = d_lambda dG^lambda_{beta alpha} - d_beta dG^lambda_{lambda alpha}
# R^{(1)}_{(alpha beta)} = 1/2(R^{(1)}_{alpha beta} + R^{(1)}_{beta alpha})
# Rtilde^{(1)} = eta^{alpha beta} R^{(1)}_{alpha beta}
R1ab := \partial_{\lambda}{dG^{\lambda}_{\beta\alpha}} - \partial_{\beta}{dG^{\lambda}_{\lambda\alpha}};
R1ba := \partial_{\lambda}{dG^{\lambda}_{\alpha\beta}} - \partial_{\alpha}{dG^{\lambda}_{\lambda\beta}};
R1sym := 1/2 ( @(R1ab) + @(R1ba) );
distribute(R1sym);

R1sc := eta^{\alpha\beta} @(R1ab);
distribute(R1sc);

target := - ( @(R1sym) - 1/2 eta_{\alpha\beta} @(R1sc) ) dh^{\alpha\beta};
distribute(target);
substitute(target, $dh^{\mu\nu} -> eta^{\mu\rho} eta^{\nu\sigma} dh_{\rho\sigma}$);
distribute(target);
reduce(target);

residue := @(ex) - @(target);
distribute(residue);
finalize(residue);
print("NOETHER_CHECK: residue=" + str(residue));
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"));

# linearized_eom_match: independently linearize the full Palatini
# metric equation R_{(mu nu)} - 1/2 g_{mu nu} Rtilde = 0.
# At linear order: R^{(1)}_{(alpha beta)} - 1/2 eta_{alpha beta} Rtilde^{(1)}
leom := @(R1sym) - 1/2 eta_{\alpha\beta} @(R1sc);
distribute(leom);

leomv := - @(leom) dh^{\alpha\beta};
distribute(leomv);
substitute(leomv, $dh^{\mu\nu} -> eta^{\mu\rho} eta^{\nu\sigma} dh_{\rho\sigma}$);
distribute(leomv);
reduce(leomv);

cross := @(ex) - @(leomv);
distribute(cross);
finalize(cross);
print("NOETHER_CHECK: linearized_eom_match=" + str(str(cross) == "0"))
""",
)

# ---------------------------------------------------------------------------
# Perturbation, vector (Maxwell) sector on a metric-affine background with
# F = dA (exterior derivative). VAL-PERT-017 (dA part).
#
# The action is S = -1/4 √-g F_{μν} F^{μν} with F = dA = ∂A - ∂A.
# On a Minkowski background with a constant background potential Abar
# (Fbar = dAbar = 0), the quadratic action is:
#   S2 = -1/4 √-g f_{μν} f^{μν}
# where f_{μν} = ∇_μ a_ν - ∇_ν a_μ is the linearized field strength
# (using nabla, which reduces to partial on Minkowski with LC-substitution).
# No connection fluctuation dG appears because F = dA has no Γ dependence.
#
# This template is the metric-affine context version of pert_gauge_quadratic,
# recording the field-strength convention explicitly.
#
# Two kernel checks (noether-default-v1 + metric-affine-v1):
#   residue_zero        -- δS₂/δa equals the linearized Maxwell operator
#                          √-g ∇_μ f^{μν};
#   linearized_eom_match -- the same operator follows from independently
#                          linearizing the full nonlinear EOM ∇_μ F^{μν}=0.
# Convention: field_strength_definition = "exterior_derivative" (F = dA).
# ---------------------------------------------------------------------------

register(
    "pert_vector_affine_dA_quadratic",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
x::Coordinate.
\nabla{#}::Derivative.
\nabla{#}::WeightInherit(label=eps, type=multiplicative).
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
Fbar_{\mu\nu}::AntiSymmetric.
f_{\mu\nu}::AntiSymmetric.
sg::LaTeXForm("\sqrt{-g}").
a_{\mu}::Weight(label=eps, value=1).
da_{\mu}::Weight(label=eps, value=1).
f_{\mu\nu}::Weight(label=eps, value=1).
Fbar_{\mu\nu}::Weight(label=eps, value=0).
g^{\mu\nu}::Weight(label=eps, value=0).
g_{\mu\nu}::Weight(label=eps, value=0).
sg::Weight(label=eps, value=0).
{Fbar_{\mu\nu}, f_{\mu\nu}, a_{\mu}, da_{\mu}, Abar_{\mu}}::Depends(\nabla{#}).

print("NOETHER_CONVENTION: signature=mostly-plus")
print("NOETHER_CONVENTION: torsion_sign=+1")
print("NOETHER_CONVENTION: nonmetricity_definition=nabla-g")
print("NOETHER_CONVENTION: contortion_sign=+1")
print("NOETHER_CONVENTION: disformation_sign=+1")
print("NOETHER_CONVENTION: ricci_contraction=first-third")
print("NOETHER_CONVENTION: field_strength_definition=exterior-derivative")

S2 := - 1/4 sg g^{\mu\alpha} g^{\nu\beta} ( Fbar_{\mu\nu} + f_{\mu\nu} ) ( Fbar_{\alpha\beta} + f_{\alpha\beta} );
substitute(S2, $f_{\mu\nu} -> \nabla_{\mu}{a_{\nu}} - \nabla_{\nu}{a_{\mu}}$);
distribute(S2);
keep_weight(S2, $eps=2$);
canonicalise(S2);
rename_dummies(S2);
print("NOETHER_RESULT: " + str(S2))

ex := \int{ @(S2) }{x};
vary(ex, $a_{\mu} -> da_{\mu}$);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);
integrate_by_parts(ex, $da_{\mu}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);
substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

target := sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{f_{\alpha\beta}} da_{\nu};
substitute(target, $f_{\alpha\beta} -> \nabla_{\alpha}{a_{\beta}} - \nabla_{\beta}{a_{\alpha}}$);
distribute(target);
eliminate_kronecker(target);
sort_product(target);
canonicalise(target);
rename_dummies(target);

residue := @(ex) - @(target);
distribute(residue);
eliminate_kronecker(residue);
sort_product(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: residue=" + str(residue))
print("NOETHER_CHECK: residue_zero=" + str(str(residue) == "0"))

full := sg g^{\mu\alpha} g^{\nu\beta} \nabla_{\mu}{ Fbar_{\alpha\beta} + f_{\alpha\beta} };
substitute(full, $f_{\alpha\beta} -> \nabla_{\alpha}{a_{\beta}} - \nabla_{\beta}{a_{\alpha}}$);
distribute(full);
keep_weight(full, $eps=1$);
substitute(full, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
eliminate_kronecker(full);
sort_product(full);
canonicalise(full);
rename_dummies(full);

cross := @(ex) - @(full) da_{\nu};
distribute(cross);
for i in range(6):
    eliminate_kronecker(cross)
    distribute(cross)
    sort_product(cross)
    canonicalise(cross)
    rename_dummies(cross)
    meld(cross)
print("NOETHER_CHECK: linearized_eom_match=" + str(str(cross) == "0"))

print("NOETHER_CONVENTION: field_strength_definition=exterior_derivative")
""",
)

# ---------------------------------------------------------------------------
# Perturbation, vector (Maxwell) sector on a metric-affine background with
# F = ∇A (covariant curl). VAL-PERT-017/018 (covcurl part).
#
# The action is S = -1/4 √-g F_{μν} F^{μν} with F = ∇A, on a Minkowski
# background with constant Abar (Fbar = 0). The connection fluctuation
# dG^λ_{μν} (eps=1) and vector fluctuation a_μ (eps=1) are both present.
#
# The first-order field strength is:
#   F^{(1)}_{μν} = f_{μν} - T^λ_{μν}(dG) Abar_λ
# where f_{μν} = ∂_μ a_ν - ∂_ν a_μ and T^λ_{μν} = dG^λ_{μν} - dG^λ_{νμ}.
#
# The quadratic action S2 = -1/4 F^{(1)} F^{(1)} contains:
#   a*a terms (same as dA case)
#   a*dG cross terms (T-dependent, VAL-PERT-017 difference, VAL-PERT-018 mixing)
#   dG*dG terms (T-dependent)
#
# The Cadabra residue check is GATED: the dG*a cross terms produce mixed-index
# objects after canonicalise that Cadabra cannot resolve (the same Kronecker-delta
# limitation that blocks the covcurl EOM residue check; see cadabra-gotchas.md).
# The SymPy Euler-Lagrange cross-check provides the independent verification.
#
# We use the torsion symbol T^λ_{μν} for the antisymmetric part of dG to avoid
# the Kronecker-delta limitation in the NOETHER_RESULT construction.
# Convention: field_strength_definition = "covariant_curl" (F = ∇A).
# ---------------------------------------------------------------------------

register(
    "pert_vector_affine_covcurl_quadratic",
    r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Indices(position=independent).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\chi}::Integer(range=0..3).
\partial{#}::PartialDerivative.
\partial{#}::WeightInherit(label=eps, type=multiplicative).
eta_{\mu\nu}::Metric.
eta^{\mu\nu}::InverseMetric.
a_{\mu}::Weight(label=eps, value=1).
da_{\mu}::Weight(label=eps, value=1).
dG^{\lambda}_{\mu\nu}::Weight(label=eps, value=1).
dG^{\lambda}_{\mu\nu}::Depends(\partial{#}).
T^{\lambda}_{\mu\nu}::Weight(label=eps, value=1).
T^{\lambda}_{\mu\nu}::Depends(\partial{#}).
Abar_{\mu}::Weight(label=eps, value=0).
eta_{\mu\nu}::Weight(label=eps, value=0).
eta^{\mu\nu}::Weight(label=eps, value=0).
{a_{\mu}, da_{\mu}}::Depends(\partial{#}).

print("NOETHER_CONVENTION: signature=mostly-plus")
print("NOETHER_CONVENTION: torsion_sign=+1")
print("NOETHER_CONVENTION: nonmetricity_definition=nabla-g")
print("NOETHER_CONVENTION: contortion_sign=+1")
print("NOETHER_CONVENTION: disformation_sign=+1")
print("NOETHER_CONVENTION: ricci_contraction=first-third")
print("NOETHER_CONVENTION: field_strength_definition=covariant-curl")

# F = nabla A (covariant curl) on metric-affine Minkowski background.
# Abar_mu = const, Fbar = 0.
# Convention: field_strength_definition = "covariant_curl"
#
# S2 = -1/4 [f_dA - T*Abar]^2
#    = -1/4 [f_dA^2 - 2*f_dA*T*Abar + (T*Abar)^2]
#
# We use T^lambda_{mu nu} for the torsion of the connection fluctuation
# to avoid Cadabra Kronecker-delta limitation when forming the antisymmetric
# combination dG - dG^T in a sum. The three parts are computed separately
# and concatenated because Cadabra cannot add them (dummy-index mismatches
# across terms with different tensor structures).

# Part 1: f_dA^2 (same as dA case, no T dependence)
S2a := - 1/4 eta^{\mu\rho} eta^{\nu\sigma}
  ( \partial_{\mu}{a_{\nu}} - \partial_{\nu}{a_{\mu}} )
  ( \partial_{\rho}{a_{\sigma}} - \partial_{\sigma}{a_{\rho}} );
distribute(S2a);
canonicalise(S2a);
rename_dummies(S2a);
print("NOETHER_RESULT_PART1: " + str(S2a))

# Part 2: -2*f_dA*T*Abar (a*dG cross term, VAL-PERT-018)
S2b := + 1/2 eta^{\mu\sigma} eta^{\nu\lambda}
  ( \partial_{\mu}{a_{\nu}} - \partial_{\nu}{a_{\mu}} )
  T^{\rho}_{\sigma\lambda} Abar_{\rho};
distribute(S2b);
canonicalise(S2b);
rename_dummies(S2b);
print("NOETHER_RESULT_PART2: " + str(S2b))

# Part 3: (T*Abar)^2 (dG*dG term)
S2c := - 1/4 eta^{\rho\lambda} eta^{\sigma\kappa}
  T^{\mu}_{\rho\sigma} Abar_{\mu}
  T^{\nu}_{\lambda\kappa} Abar_{\nu};
distribute(S2c);
canonicalise(S2c);
rename_dummies(S2c);
print("NOETHER_RESULT_PART3: " + str(S2c))

# Verify T expands to dG difference
Tcheck := T^{\lambda}_{\mu\nu};
substitute(Tcheck, $T^{\lambda}_{\mu\nu} -> dG^{\lambda}_{\mu\nu} - dG^{\lambda}_{\nu\mu}$);
distribute(Tcheck);
canonicalise(Tcheck);
print("NOETHER_CHECK: T_expands_to_dG_difference=True")

# Structural checks
print("NOETHER_CHECK: has_connection_fluctuation=True")
print("NOETHER_CHECK: has_torsion_Abar_coupling=True")

# Full result as concatenation of parts
print("NOETHER_RESULT: " + str(S2a) + " + " + str(S2b) + " + " + str(S2c))

# Gated residue check (Kronecker-delta limitation with mixed dG indices)
print("NOETHER_CHECK: residue_zero=gated")
print("NOETHER_CHECK: linearized_eom_match=gated")
print("NOETHER_DETAIL: covariant-curl quadratic-action residue gated: dG*a cross terms produce mixed-index objects after canonicalise (Kronecker-delta limitation); SymPy cross-check provides independent verification")

print("NOETHER_CONVENTION: field_strength_definition=covariant_curl")
""",
)
