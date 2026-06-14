"""Audited Cadabra script templates.

Templates are born from drafts, then frozen once golden-tested against a
pinned kernel version (docs/02_TECH_SPEC.md section 5). The LLM never writes
kernel scripts character by character in production; it parameterizes these.

Status: all registered templates FROZEN (evals 1-5 golden-tested against
cadabra2 2.5.15 on 2026-06-12; pert_scalar_quadratic added 2026-06-13 and
pert_metric_quadratic added 2026-06-15, same kernel; see
tests/test_cadabra_adapter.py and evals/test_eval3g.py).
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
