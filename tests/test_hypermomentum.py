"""Hypermomentum decomposition into spin, dilation, and shear pieces.

VAL-EOM-022: For matter that couples to the connection, the connection
variation returns a hypermomentum source decomposing into spin, dilation,
and shear pieces (not only the projective trace); verified-or-gated.

Conventions: noether-default-v1 + metric-affine-v1.

Two verification gates:
1. Cadabra residue check: the reconstruction identity (Delta = tau + dil + sigma)
   and the trace properties (spin trace = 0, shear trace = 0).
2. SymPy cross-check: the full decomposition (reconstruction, tracelessness,
   antisymmetry of spin, symmetry of shear) on explicit random backgrounds.
"""

from __future__ import annotations

import pytest
import sympy as sp

from noether.kernels.base import Capability, KernelTask
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel.geometry import (
    components,
    hypermomentum_dilation_trace,
    hypermomentum_reconstruction_residual,
    hypermomentum_shear,
    hypermomentum_shear_sym_residual,
    hypermomentum_shear_trace_residual,
    hypermomentum_spin,
    hypermomentum_spin_antisym_residual,
    hypermomentum_spin_trace_residual,
    random_diagonal_metric,
    random_hypermomentum,
)

# ---------------------------------------------------------------------------
# Cadabra residue checks (require cadabra2 installed)
# ---------------------------------------------------------------------------

HYPERMOMENTUM_DECOMP_SCRIPT = r"""
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma}::Indices(position=fixed).
{\mu,\nu,\rho,\sigma,\lambda,\kappa,\alpha,\beta,\gamma}::Integer(range=0..3).
g_{\mu\nu}::Metric.
g^{\mu\nu}::InverseMetric.
g_{\mu}^{\nu}::KroneckerDelta.
g^{\mu}_{\nu}::KroneckerDelta.
\partial{#}::PartialDerivative.
Delta^{\lambda}_{\mu\nu}::Depends(\partial{#}).

n := 4;

# Reconstruction: tau + dil + sigma = Delta
tauterm := 1/2 Delta^{\lambda}_{\mu\nu} - 1/2 g^{\lambda\rho} g_{\mu\sigma} Delta^{\sigma}_{\rho\nu};
sigmatrm := 1/2 Delta^{\lambda}_{\mu\nu} + 1/2 g^{\lambda\rho} g_{\mu\sigma} Delta^{\sigma}_{\rho\nu} - (1/n) g^{\lambda}_{\mu} Delta^{\kappa}_{\kappa\nu};
dilatr := (1/n) g^{\lambda}_{\mu} Delta^{\kappa}_{\kappa\nu};

recon := @(tauterm) + @(sigmatrm) + @(dilatr);
distribute(recon);
eliminate_metric(recon);
eliminate_kronecker(recon);
canonicalise(recon);
rename_dummies(recon);
meld(recon);

residue := @(recon) - Delta^{\lambda}_{\mu\nu};
distribute(residue);
eliminate_metric(residue);
eliminate_kronecker(residue);
canonicalise(residue);
rename_dummies(residue);
meld(residue);
print("NOETHER_CHECK: reconstruction_zero=" + str(str(residue) == "0"))

# Spin trace: tau^{lam}_{lam nu} = (1/2)(Delta^{lam}_{lam nu} - Delta^{kap}_{kap nu}) = 0
spntr := (1/2) Delta^{\lambda}_{\lambda\nu} - (1/2) Delta^{\kappa}_{\kappa\nu};
distribute(spntr);
canonicalise(spntr);
rename_dummies(spntr);
meld(spntr);
print("NOETHER_CHECK: spin_trace_zero=" + str(str(spntr) == "0"))

# Shear trace: sigma^{lam}_{lam nu} = Delta^{kap}_{kap nu} - Delta^{rho}_{rho nu} = 0
shtr := (1/2) Delta^{\lambda}_{\lambda\nu} + (1/2) Delta^{\kappa}_{\kappa\nu} - Delta^{\rho}_{\rho\nu};
distribute(shtr);
canonicalise(shtr);
rename_dummies(shtr);
meld(shtr);
print("NOETHER_CHECK: shear_trace_zero=" + str(str(shtr) == "0"))
"""


def _run_hm_script():
    """Run the hypermomentum decomposition script and return the ComputedResult."""
    return CadabraAdapter().run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="hypermomentum decomposition check",
            payload={"script": HYPERMOMENTUM_DECOMP_SCRIPT},
        )
    )


@pytest.mark.kernel_cadabra
@pytest.mark.skipif(not CadabraAdapter().available(), reason="cadabra2 not installed")
class TestHypermomentumDecomposition:
    """Hypermomentum decomposition: spin / dilation / shear."""

    def test_reconstruction_zero(self):
        """Delta = tau + dilation + sigma (algebraic reconstruction identity)."""
        result = _run_hm_script()
        checks = result.value.get("checks", {})
        assert checks.get("reconstruction_zero") == "True", (
            f"Hypermomentum reconstruction failed: {checks}"
        )

    def test_spin_trace_zero(self):
        """Spin part is traceless: tau^{lambda}_{lambda nu} = 0."""
        result = _run_hm_script()
        checks = result.value.get("checks", {})
        assert checks.get("spin_trace_zero") == "True", (
            f"Spin trace check failed: {checks}"
        )

    def test_shear_trace_zero(self):
        """Shear part is traceless: sigma^{lambda}_{lambda nu} = 0."""
        result = _run_hm_script()
        checks = result.value.get("checks", {})
        assert checks.get("shear_trace_zero") == "True", (
            f"Shear trace check failed: {checks}"
        )


# ---------------------------------------------------------------------------
# SymPy cross-check: verify the decomposition on explicit backgrounds
# ---------------------------------------------------------------------------

class TestHypermomentumDecompositionSymPy:
    """SymPy cross-check for hypermomentum decomposition properties."""

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_reconstruction_on_random_delta(self, seed):
        """Delta = tau + dilation + sigma on a random hypermomentum tensor."""
        Delta = random_hypermomentum(seed, dim=3)
        g = sp.ImmutableMatrix(sp.eye(3))
        g_inv = sp.ImmutableMatrix(sp.eye(3))

        residual = hypermomentum_reconstruction_residual(Delta, g, g_inv)
        for c in components(residual):
            assert sp.simplify(c) == 0, (
                f"Reconstruction residual nonzero at seed {seed}: {c}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_spin_trace_on_random_delta(self, seed):
        """Spin part is traceless: tau^{lambda}_{lambda nu} = 0."""
        Delta = random_hypermomentum(seed, dim=3)
        g = sp.ImmutableMatrix(sp.eye(3))
        g_inv = sp.ImmutableMatrix(sp.eye(3))

        residual = hypermomentum_spin_trace_residual(Delta, g, g_inv)
        for c in components(residual):
            assert sp.simplify(c) == 0, (
                f"Spin trace residual nonzero at seed {seed}: {c}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_shear_trace_on_random_delta(self, seed):
        """Shear part is traceless: sigma^{lambda}_{lambda nu} = 0."""
        Delta = random_hypermomentum(seed, dim=3)
        g = sp.ImmutableMatrix(sp.eye(3))
        g_inv = sp.ImmutableMatrix(sp.eye(3))

        residual = hypermomentum_shear_trace_residual(Delta, g, g_inv)
        for c in components(residual):
            assert sp.simplify(c) == 0, (
                f"Shear trace residual nonzero at seed {seed}: {c}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_spin_antisymmetry_on_random_delta(self, seed):
        """Spin part is antisymmetric in the first pair (requires metric)."""
        Delta = random_hypermomentum(seed, dim=3)
        g = sp.ImmutableMatrix(sp.eye(3))
        g_inv = sp.ImmutableMatrix(sp.eye(3))

        residual = hypermomentum_spin_antisym_residual(Delta, g, g_inv)
        for c in components(residual):
            assert sp.simplify(c) == 0, (
                f"Spin antisymmetry residual nonzero at seed {seed}: {c}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_shear_symmetry_on_random_delta(self, seed):
        """Shear part is symmetric in the first pair (requires metric)."""
        Delta = random_hypermomentum(seed, dim=3)
        g = sp.ImmutableMatrix(sp.eye(3))
        g_inv = sp.ImmutableMatrix(sp.eye(3))

        residual = hypermomentum_shear_sym_residual(Delta, g, g_inv)
        for c in components(residual):
            assert sp.simplify(c) == 0, (
                f"Shear symmetry residual nonzero at seed {seed}: {c}"
            )

    @pytest.mark.parametrize("seed", [7, 19, 37])
    def test_dilation_carries_full_trace(self, seed):
        """Dilation trace equals the full hypermomentum trace.

        Delta_nu = Delta^{lambda}_{lambda nu} and both spin and shear
        are traceless, so the dilation carries the full trace.
        """
        Delta = random_hypermomentum(seed, dim=3)
        g = sp.ImmutableMatrix(sp.eye(3))
        g_inv = sp.ImmutableMatrix(sp.eye(3))

        trace = hypermomentum_dilation_trace(Delta)
        tau = hypermomentum_spin(Delta, g, g_inv)
        sigma = hypermomentum_shear(Delta, g, g_inv)

        n = 3
        # Check spin is traceless
        for nu in range(n):
            spin_tr = sum(tau[lam, lam, nu] for lam in range(n))
            assert sp.simplify(spin_tr) == 0, (
                f"Spin has nonzero trace at nu={nu}, seed {seed}"
            )

        # Check shear is traceless
        for nu in range(n):
            shear_tr = sum(sigma[lam, lam, nu] for lam in range(n))
            assert sp.simplify(shear_tr) == 0, (
                f"Shear has nonzero trace at nu={nu}, seed {seed}"
            )

        # Check dilation equals the full trace
        for nu in range(n):
            full_tr = sum(Delta[lam, lam, nu] for lam in range(n))
            assert sp.simplify(trace[nu] - full_tr) == 0, (
                f"Dilation trace mismatch at nu={nu}, seed {seed}"
            )

    @pytest.mark.parametrize("seed", [7, 19])
    def test_decomposition_on_curved_metric(self, seed):
        """Decomposition on a curved metric (not just identity)."""
        geom = random_diagonal_metric(seed, dim=3)
        Delta = random_hypermomentum(seed + 50, dim=3)

        g_inv = sp.ImmutableDenseNDimArray(geom.g_inv)
        g = sp.ImmutableDenseNDimArray(geom.g)

        # Reconstruction
        res_recon = hypermomentum_reconstruction_residual(Delta, g, g_inv)
        for c in components(res_recon):
            assert sp.simplify(c) == 0, f"Reconstruction failed on curved metric seed {seed}"

        # Spin trace
        res_spin_tr = hypermomentum_spin_trace_residual(Delta, g, g_inv)
        for c in components(res_spin_tr):
            assert sp.simplify(c) == 0, f"Spin trace failed on curved metric seed {seed}"

        # Shear trace
        res_shear_tr = hypermomentum_shear_trace_residual(Delta, g, g_inv)
        for c in components(res_shear_tr):
            assert sp.simplify(c) == 0, f"Shear trace failed on curved metric seed {seed}"

        # Spin antisymmetry
        res_spin_as = hypermomentum_spin_antisym_residual(Delta, g, g_inv)
        for c in components(res_spin_as):
            assert sp.simplify(c) == 0, f"Spin antisym failed on curved metric seed {seed}"

        # Shear symmetry
        res_shear_sym = hypermomentum_shear_sym_residual(Delta, g, g_inv)
        for c in components(res_shear_sym):
            assert sp.simplify(c) == 0, f"Shear sym failed on curved metric seed {seed}"
