"""Audited Cadabra script templates.

Templates are born from drafts, then frozen once golden-tested against a
pinned kernel version (docs/02_TECH_SPEC.md section 5). The LLM never writes
kernel scripts character by character in production; it parameterizes these.

Status: all registered templates FROZEN (golden-tested against cadabra2
2.5.15 on 2026-06-12; see tests/test_cadabra_adapter.py).
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
