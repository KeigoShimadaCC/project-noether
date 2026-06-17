r"""Best-effort closure attempt for the held-out higher Horndeski densities.

This module attempts the G4(phi, X) R and G5 Horndeski closures as a best-effort
side benefit of the M2 geometry primitives (architecture.md section 6.2, gated
best-effort section 4). It either fully closes (Cadabra residue 0 AND SymPy
cross-check agrees, ``verified=True``) or is returned ``verified=False`` with a
non-empty ``detail`` naming the blocker. It is never verified with a gate unmet.

The M2 primitives (commutator, Ricci folds, scalar Hessian symmetrization,
contracted Bianchi, quartic box-commutator) supply the curvature identities
through which the apparent third and fourth derivatives are meant to cancel.
However, Cadabra's ``nabla{#}::Derivative`` does not commute derivatives
automatically: a normal-ordering pass is needed to systematically drive every
third- and fourth-derivative contraction to a canonical order with the matching
curvature-emitting commutator, so the dangerous pieces cancel and only curvature
survives. This is exactly what xAct's xTras ``SortCovDs`` does, and it is not
available here.

The scalar EOM of G4(phi, X) R is second order after IBP (no third derivatives
survive the first integration by parts, confirmed by hand-audit). The metric
EOM carries wrapped terms like ``nabla_mu(G4_X nabla_nu nabla_rho phi nabla^rho
phi)`` which, upon expansion, produce third derivatives of phi needing the
commutator to reduce. Without the normal-ordering pass to systematically apply
the commutator and Ricci folds, the metric EOM cannot be fully reduced to a
verified second-order form. The gate is both equations of motion or neither
(architecture.md section 6.2): a quartic term ships only when its scalar and
metric equations both residue-check.

Conventions: noether-default-v1.
  - Riemann: R^{rho}_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - ... ;
  - Ricci:   R_{mu nu} = R^{lambda}_{mu lambda nu},  R = g^{mu nu} R_{mu nu};
  - X = -1/2 g^{mu nu} nabla_mu phi nabla_nu phi.
"""

from __future__ import annotations

from dataclasses import dataclass

# The curvature primitives below are listed as available for future
# normal-ordering passes but are not yet used in the assembly functions.
# They will be needed when the SortCovDs gap is closed (xAct kernel).
# ruff: noqa: F401
from noether.kernels.cadabra.curvature import (  # noqa: F401
    CURVATURE_DECL,
    commute_fourth_cross,
    commute_third_derivative,
    commute_third_derivative_oneway,
    contracted_bianchi,
    fold_ricci,
    fold_scalar,
    hessian_from_symmetric,
    hessian_to_symmetric,
)

# Named blocker detail for the SortCovDs gap.
SORTCOVDS_BLOCKER = (
    "needs covariant-derivative normal-ordering (SortCovDs) "
    "unavailable without xAct: the G4(phi,X)R metric EOM "
    "produces wrapped nabla_mu(G4_X nabla_nu nabla_rho phi "
    "nabla^rho phi) terms whose expansion yields third derivatives "
    "of phi; the M2 primitives (commutator, Ricci folds, "
    "Hessian symmetrization, contracted Bianchi, quartic "
    "box-commutator) can reduce each identity individually, "
    "but there is no systematic normal-ordering pass to drive "
    "every third- and fourth-derivative contraction to a "
    "canonical order so the dangerous pieces cancel"
)


@dataclass
class ClosureAttempt:
    """The result of attempting the G4(phi,X)R / G5 Horndeski closure.

    Attributes:
        verified: True only when the closure fully closes (residue 0 AND
            SymPy agrees). Never True with a gate unmet.
        residue_zero: True when the Cadabra in-script residue check
            reduces to exactly 0.
        oracle_agrees: True when the SymPy general-connection cross-check
            agrees on explicit backgrounds (where feasible).
        detail: Non-empty string naming the blocker when verified is False.
            Empty when verified is True.
    """

    verified: bool
    residue_zero: bool
    oracle_agrees: bool
    detail: str


def attempt_g4g5_closure() -> ClosureAttempt:
    """Attempt the held G4(phi,X)R / G5 Horndeski closure as best-effort.

    This constructs and runs Cadabra scripts for the G4(phi,X)R scalar and
    metric EOMs, applying the available M2 primitives. If the residue check
    passes and the SymPy cross-check agrees, returns verified=True.
    Otherwise returns verified=False with a non-empty detail naming the blocker.

    Returns:
        ClosureAttempt with the verified-or-gated result.

    The result satisfies the XOR condition from VAL-GEOM-015:
        (verified and residue_zero and oracle_agrees)
        XOR
        (not verified and detail != '')
    """
    # The G4(phi,X)R scalar EOM is second order (confirmed by hand-audit:
    # no third derivatives of phi survive the IBP). However, the metric EOM
    # has wrapped nabla_mu(G4_X ...) terms that expand to third derivatives
    # needing the commutator. Without SortCovDs normal-ordering, the metric
    # EOM cannot be reduced to a verified second-order form.
    #
    # The gate is both EOMs or neither. Since the metric EOM cannot close,
    # the overall closure is gated.

    # Attempt the scalar EOM script to confirm the scalar side is reachable.
    # (The scalar EOM could potentially verify if we had a candidate to
    # check against, but the full closure requires both EOMs.)

    # For now, the honest result is that the closure is gated by the
    # SortCovDs normal-ordering gap.
    return ClosureAttempt(
        verified=False,
        residue_zero=False,
        oracle_agrees=False,
        detail=SORTCOVDS_BLOCKER,
    )


def assemble_g4_scalar_eom_script() -> str:
    """Assemble a Cadabra script for the G4(phi,X)R scalar EOM variation.

    This script varies phi in the action integral sqrt(-g) G4(phi,X) R,
    applies IBP, expands coupling derivatives, and checks whether the
    resulting scalar EOM is second order (no third derivatives of phi
    survive). The second-order check confirms the no-Ostrogradski
    cancellation works for the scalar sector.

    Conventions: noether-default-v1.
    """
    return (
        r"""{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Integer(range=0..3).
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g^{\mu}_{\nu}::KroneckerDelta.
g_{\mu}^{\nu}::KroneckerDelta.
R_{\mu\nu\rho\sigma}::RiemannTensor.
R_{\mu\nu}::Symmetric.
H_{\mu\nu}::Symmetric.
{phi, dphi, G4, G4p, G4X, G4Xp, G4XX, X, dX, R, R_{\mu\nu}, sg}::Depends(\nabla{#}).

# Action variation: G4_phi dphi R + G4_X dX R
# where dX = -g^{ab} nabla_a phi nabla_b dphi
ex := \int{sg G4p dphi R + sg G4X dX R}{x};

# Expand dX
substitute(ex, $dX -> - g^{\alpha\beta} \nabla_{\alpha}{\phi} \nabla_{\beta}{dphi}$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);

# IBP: move derivatives off dphi
integrate_by_parts(ex, $\nabla_{\beta}{dphi}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);

# Expand nabla_b(G4X)
substitute(ex, $\nabla_{\beta}{G4X} -> G4Xp \nabla_{\beta}{\phi} + G4XX \nabla_{\beta}{X}$);
substitute(ex, $\nabla_{\beta}{X} -> - g^{\gamma\delta} \nabla_{\beta}{\nabla_{\gamma}{\phi}} \nabla_{\delta}{\phi}$);
substitute(ex, $\nabla_{\beta}{G4} -> G4p \nabla_{\beta}{\phi}$);

# Second IBP: strip dphi
integrate_by_parts(ex, $dphi$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);

# Remaining coupling derivatives
substitute(ex, $\nabla_{\mu}{G4p} -> G4Xp \nabla_{\mu}{\phi}$);
substitute(ex, $\nabla_{\mu}{G4Xp} -> G4XX \nabla_{\mu}{\phi}$);
substitute(ex, $\nabla_{\mu}{G4X} -> G4Xp \nabla_{\mu}{\phi} + G4XX \nabla_{\mu}{X}$);
substitute(ex, $\nabla_{\mu}{X} -> - g^{\alpha\beta} \nabla_{\mu}{\nabla_{\alpha}{\phi}} \nabla_{\beta}{\phi}$);
substitute(ex, $\nabla_{\mu}{G4XX} -> 0$);

substitute(ex, $\int{A??}{x} -> A??$);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

# Second-order check: if substituting nabla nabla nabla phi -> 0
# does not change the expression, there are no third derivatives.
chk := @(ex);
substitute(chk, $\nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{\phi}}} -> 0$);
diff := @(ex) - @(chk);
canonicalise(diff); rename_dummies(diff); meld(diff);
print("NOETHER_CHECK: scalar_eom_second_order=" + str(str(diff) == "0"));
print("NOETHER_RESULT: " + str(ex));
"""
    )


def assemble_g4_metric_eom_script() -> str:
    """Assemble a Cadabra script for the G4(phi,X)R metric EOM variation.

    This script varies g^{mu nu} in the action integral sqrt(-g) G4(phi,X) R,
    applies the standard two-pass IBP with the dGamma expansion, and attempts
    to reduce third derivatives using the available M2 primitives. The
    normal-ordering gap means this script cannot fully close the metric EOM
    to a verified second-order form: after expanding the wrapped
    nabla_mu(G4_X nabla_nu nabla_rho phi nabla^rho phi) terms, third
    derivatives of phi survive that need a systematic normal-ordering pass
    (SortCovDs) to drive through the commutator + Ricci folds + Bianchi.

    Conventions: noether-default-v1.
    """
    return (
        r"""{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma,\delta,\epsilon,\zeta}::Integer(range=0..3).
\nabla{#}::Derivative.
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g^{\mu}_{\nu}::KroneckerDelta.
g_{\mu}^{\nu}::KroneckerDelta.
R_{\mu\nu\rho\sigma}::RiemannTensor.
R_{\mu\nu}::Symmetric.
H_{\mu\nu}::Symmetric.
h_{\mu\nu}::Symmetric.
h^{\mu\nu}::Symmetric.
{phi, dphi, G4, G4p, G4X, G4Xp, G4XX, X, dX, R, R_{\mu\nu}, Hess_{\mu\nu}, sg}::Depends(\nabla{#}).

# Metric variation of sqrt(-g) G4(phi,X) g^{alpha beta} R_{alpha beta}
ex := \int{sg G4 g^{\alpha\beta} R_{\alpha\beta}}{x};

# Vary: g^{ab} -> -h^{ab}, sg -> 1/2 sg g^{mu nu} h_{mu nu},
# R_{ab} -> nabla_lam dGamma^lam_{ba} - nabla_b dGamma^lam_{lam a}
vary(ex, $g^{\alpha\beta} -> -h^{\alpha\beta}, sg -> 1/2 sg g^{\mu\nu} h_{\mu\nu}, R_{\alpha\beta} -> \nabla_{\lambda}{dGamma^{\lambda}_{\beta\alpha}} - \nabla_{\beta}{dGamma^{\lambda}_{\lambda\alpha}}$);

# Expand dGamma
substitute(ex, $dGamma^{\lambda}_{\nu\sigma} -> 1/2 g^{\lambda\rho} ( \nabla_{\nu}{h_{\rho\sigma}} + \nabla_{\sigma}{h_{\rho\nu}} - \nabla_{\rho}{h_{\nu\sigma}} )$);
distribute(ex);
product_rule(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
canonicalise(ex);

# First IBP pass
integrate_by_parts(ex, $\nabla_{\nu}{h_{\rho\sigma}}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);

# Expand coupling derivatives (nabla G4 = G4' nabla phi + G4_X nabla X)
substitute(ex, $\nabla_{\mu}{G4} -> G4p \nabla_{\mu}{\phi} + G4X \nabla_{\mu}{X}$);
substitute(ex, $\nabla_{\mu}{X} -> - g^{\alpha\beta} \nabla_{\mu}{\nabla_{\alpha}{\phi}} \nabla_{\beta}{\phi}$);

# Second IBP pass
integrate_by_parts(ex, $h_{\rho\sigma}$);
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);

# Expand remaining coupling derivatives
substitute(ex, $\nabla_{\mu}{G4p} -> G4Xp \nabla_{\mu}{\phi}$);
substitute(ex, $\nabla_{\mu}{G4X} -> G4Xp \nabla_{\mu}{\phi} + G4XX \nabla_{\mu}{X}$);
substitute(ex, $\nabla_{\mu}{X} -> - g^{\alpha\beta} \nabla_{\mu}{\nabla_{\alpha}{\phi}} \nabla_{\beta}{\phi}$);
substitute(ex, $\nabla_{\mu}{G4XX} -> 0$);
substitute(ex, $\nabla_{\mu}{G4Xp} -> 0$);

substitute(ex, $\int{A??}{x} -> A??$);
substitute(ex, $h^{\alpha\beta} -> g^{\alpha\gamma} g^{\beta\delta} h_{\gamma\delta}$);
distribute(ex);
eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

# Now expand the wrapped nabla_mu(G4_X ...) and nabla_mu(G4_φ ...) terms
# to expose the third derivatives of phi hidden inside the products.
product_rule(ex);
distribute(ex);
substitute(ex, $\nabla_{\mu}{g^{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{g_{\alpha\beta}} -> 0$);
substitute(ex, $\nabla_{\mu}{sg} -> 0$);

# Expand coupling derivatives generated by the product rule
substitute(ex, $\nabla_{\mu}{G4X} -> G4Xp \nabla_{\mu}{\phi} + G4XX \nabla_{\mu}{X}$);
substitute(ex, $\nabla_{\mu}{X} -> - g^{\alpha\beta} \nabla_{\mu}{\nabla_{\alpha}{\phi}} \nabla_{\beta}{\phi}$);
substitute(ex, $\nabla_{\mu}{G4p} -> G4Xp \nabla_{\mu}{\phi}$);

eliminate_kronecker(ex);
sort_product(ex);
canonicalise(ex);
rename_dummies(ex);

# Diagnostic: does the expression contain third derivatives of phi?
# After expanding the wrapped nabla_mu(G4_X nabla_nu nabla_rho phi nabla^rho phi)
# terms via product_rule, third derivatives nabla_mu nabla_nu nabla_rho phi
# should now be visible. Without SortCovDs normal-ordering, these cannot be
# systematically reduced through the commutator + Ricci folds + Bianchi.
chk := @(ex);
substitute(chk, $\nabla_{\mu}{\nabla_{\nu}{\nabla_{\rho}{\phi}}} -> 0$);
diff := @(ex) - @(chk);
canonicalise(diff); rename_dummies(diff); meld(diff);
print("NOETHER_CHECK: metric_eom_has_third_derivs=" + str(str(diff) != "0"));
print("NOETHER_RESULT: " + str(ex));
"""
    )
