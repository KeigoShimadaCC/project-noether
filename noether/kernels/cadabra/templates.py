"""Audited Cadabra script templates.

Templates are born from drafts, then frozen once golden-tested against a
pinned kernel version (docs/02_TECH_SPEC.md section 5). The LLM never writes
kernel scripts character by character in production; it parameterizes these.

Status: eval1_eh_trace FROZEN (golden-tested against cadabra2 2.5.15 on
2026-06-12; see tests/test_cadabra_adapter.py::TestGoldenEval1).
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
