"""Component geometry: known results pin the convention signs.

These are the tests that keep noether-default-v1 honest. If a sign convention
in the geometry code drifts, the 2-sphere result changes and this fails.
"""

import sympy as sp

from noether.kernels.sympy_kernel.geometry import (
    ComponentGeometry,
    components,
    random_diagonal_metric,
    two_sphere,
)


class TestKnownGeometries:
    def test_flat_space_is_flat(self):
        t, x, y, z = sp.symbols("t x y z")
        geom = ComponentGeometry([t, x, y, z], sp.diag(-1, 1, 1, 1))
        assert all(c == 0 for c in components(geom.riemann))
        assert geom.ricci_scalar == 0

    def test_two_sphere_ricci_scalar(self):
        # Round sphere of radius a: R = +2/a^2 under noether-default-v1.
        geom = two_sphere()
        a = [s for s in geom.g.free_symbols if str(s) == "a"][0]
        assert sp.simplify(geom.ricci_scalar - 2 / a**2) == 0

    def test_two_sphere_einstein_vanishes(self):
        # G_{mu nu} == 0 identically in 2 dimensions.
        geom = two_sphere()
        assert all(sp.simplify(c) == 0 for c in components(geom.einstein))


class TestRandomBackground:
    def test_seeded_metric_is_curved_and_deterministic(self):
        g1 = random_diagonal_metric(seed=7)
        g2 = random_diagonal_metric(seed=7)
        assert g1.g == g2.g
        assert any(c != 0 for c in components(g1.riemann))

    def test_contracted_bianchi_on_random_background(self):
        # nabla^mu G_{mu nu} == 0: kernel-computed identity, the heart of V2.
        geom = random_diagonal_metric(seed=7)
        grad = geom.covariant_derivative(geom.einstein, ["down", "down"])
        grad_up = geom.raise_first_index(grad, 0)
        div = sp.tensorcontraction(grad_up, (0, 1))
        assert all(sp.simplify(c) == 0 for c in components(div))

    def test_ricci_alone_is_not_divergence_free(self):
        # Falsifier sanity: nabla^mu R_{mu nu} = (1/2) nabla_nu R != 0 generally.
        # If this ever passes as zero, the spot check has lost its teeth.
        geom = random_diagonal_metric(seed=7)
        grad = geom.covariant_derivative(geom.ricci, ["down", "down"])
        grad_up = geom.raise_first_index(grad, 0)
        div = sp.tensorcontraction(grad_up, (0, 1))
        assert any(sp.simplify(c) != 0 for c in components(div))
