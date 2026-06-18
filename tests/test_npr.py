"""NPR layer: AST serialization, schema round-trips, and structural validation."""

import pytest
from pydantic import TypeAdapter

from noether.npr.ast import Deriv, Expr, Sym, add, cov, down, num, prod, tensor, up
from noether.npr.conventions import NOETHER_DEFAULT_V1, Conventions
from noether.npr.latex import render
from noether.npr.parse import parse_lagrangian
from noether.npr.schema import NPR, Action, ConnectionSpec, Geometry, ObjectDecl, Task
from noether.npr.validate import ValidationError, free_indices, validate_expression

EXPR = TypeAdapter(Expr)


def eh_trace_lagrangian():
    return prod(tensor("g", up("mu"), up("nu")), tensor("G", down("mu"), down("nu")))


class TestAst:
    def test_json_round_trip(self):
        expr = add(
            tensor("R", down("mu"), down("nu"), connection="Gamma"),
            prod(num(-1, 2), tensor("g", down("mu"), down("nu")), tensor("R", connection="Gamma")),
        )
        dumped = EXPR.dump_python(expr)
        loaded = EXPR.validate_python(dumped)
        assert EXPR.dump_python(loaded) == dumped

    def test_cov_builder_defaults_to_metric_connection(self):
        assert cov(down("mu"), tensor("F", up("mu"), up("nu"))) == Deriv(
            op="covariant",
            index=down("mu"),
            expr=tensor("F", up("mu"), up("nu")),
            connection="metric",
        )


class TestValidation:
    def test_contraction_balances(self):
        assert free_indices(eh_trace_lagrangian()) == {}

    def test_free_indices_survive(self):
        expr = tensor("R", down("mu"), down("nu"))
        assert set(free_indices(expr)) == {("mu", "down"), ("nu", "down")}

    def test_same_variance_repeat_rejected(self):
        bad = prod(tensor("v", down("mu")), tensor("w", down("mu")))
        with pytest.raises(ValidationError, match="same variance"):
            free_indices(bad)

    def test_triple_index_rejected(self):
        bad = prod(
            tensor("g", up("mu"), up("mu")),
            tensor("v", down("mu")),
        )
        with pytest.raises(ValidationError):
            free_indices(bad)

    def test_sum_mismatch_rejected(self):
        bad = add(tensor("v", down("mu")), tensor("w", down("nu")))
        with pytest.raises(ValidationError, match="mismatched free indices"):
            free_indices(bad)

    def test_derivative_contracts_across_boundary(self):
        # nabla_mu F^{mu nu}: free index is nu (up).
        expr = cov(down("mu"), tensor("F", up("mu"), up("nu")))
        assert set(free_indices(expr)) == {("nu", "up")}

    def test_expected_free_enforced(self):
        expr = tensor("G", down("mu"), down("nu"))
        validate_expression(expr, [down("mu"), down("nu")])
        with pytest.raises(ValidationError):
            validate_expression(expr, [down("mu")])

    def test_balanced_metric_affine_expression_validates_without_metric_compatibility(self):
        expr = prod(
            tensor("R", down("mu"), down("nu"), connection="Gamma"),
            tensor("T", up("mu"), down("rho"), down("sigma")),
            tensor("Q", up("nu"), up("rho"), up("sigma")),
        )

        validate_expression(expr, metric_compatible=False)

    def test_index_unbalance_still_raises_without_metric_compatibility(self):
        expr = prod(
            tensor("R", down("mu"), down("nu"), connection="Gamma"),
            tensor("T", up("mu"), down("rho"), down("sigma")),
            tensor("Q", up("nu"), up("rho"), up("rho")),
        )

        with pytest.raises(ValidationError):
            validate_expression(expr, metric_compatible=False)


class TestLatex:
    def test_known_rendering(self):
        expr = tensor("G", down("mu"), down("nu"))
        assert render(expr) == r"G_{\mu \nu}"

    def test_mixed_variance_groups(self):
        expr = tensor("R", up("rho"), down("sigma"), down("mu"), down("nu"))
        assert render(expr) == r"R^{\rho}{}_{\sigma \mu \nu}"

    def test_rational_coefficient(self):
        expr = prod(num(-1, 2), tensor("g", down("mu"), down("nu")), tensor("R"))
        assert render(expr) == r"-\tfrac{1}{2} \, g_{\mu \nu} \, R"

    def test_determinism(self):
        expr = add(
            tensor("R", down("mu"), down("nu")),
            prod(num(-1, 2), tensor("g", down("mu"), down("nu")), tensor("R")),
        )
        assert render(expr) == render(expr.model_copy(deep=True))

    def test_nonmetric_derivative_round_trip(self):
        expr = Deriv(
            op="covariant",
            index=down("mu"),
            expr=Sym(name="phi"),
            connection="Gamma",
        )
        assert parse_lagrangian(render(expr)) == expr


class TestSchema:
    def test_metric_affine_convention_fields_validate_with_alternative_ricci_contraction(self):
        conventions = Conventions(
            id="metric-affine-test",
            dimension=4,
            signature="mostly-plus",
            riemann_sign="+1",
            torsion_sign="+1",
            nonmetricity_definition="nabla-g",
            contortion_sign="+1",
            disformation_sign="+1",
            ricci_contraction="first-fourth",
            field_strength_definition="exterior-derivative",
            symmetrization_weight="1/n!",
        )
        assert conventions.ricci_contraction == "first-fourth"

    def test_metric_affine_npr_json_round_trip_preserves_geometry_and_conventions(self):
        npr = NPR(
            conventions=Conventions(
                id="metric-affine-test",
                dimension="D",
                signature="mostly-minus",
                riemann_sign="-1",
                torsion_sign="-1",
                nonmetricity_definition="minus-nabla-g",
                contortion_sign="+1",
                disformation_sign="-1",
                ricci_contraction="first-fourth",
                field_strength_definition="covariant-curl",
                symmetrization_weight="1",
            ),
            geometry=Geometry(
                metric_name="g",
                connection_name="Gamma",
                connection=ConnectionSpec(
                    type="independent",
                    torsion=True,
                    nonmetricity=True,
                    metric_compatible=False,
                    family="metric-affine",
                ),
            ),
            objects=[
                ObjectDecl(name="g", kind="metric", role="dynamical", symmetry="symmetric", rank=2),
                ObjectDecl(name="Gamma", kind="connection", role="dynamical", rank=3),
            ],
            action=Action(
                measure_tex=r"d^Dx \sqrt{-g}",
                lagrangian=tensor("R", connection="Gamma"),
                lagrangian_tex=r"R(\Gamma)",
            ),
            task=Task(type="vary", with_respect_to=["g", "Gamma"]),
        )

        round_tripped = NPR.model_validate_json(npr.model_dump_json())

        assert round_tripped == npr
        assert round_tripped.geometry == npr.geometry
        assert round_tripped.conventions == npr.conventions
        assert round_tripped.geometry.connection_name == "Gamma"
        assert round_tripped.objects == npr.objects
        assert round_tripped.action.lagrangian == npr.action.lagrangian

    def test_default_metric_affine_slots_are_present(self):
        assert NOETHER_DEFAULT_V1.torsion_sign == "+1"
        assert NOETHER_DEFAULT_V1.nonmetricity_definition == "nabla-g"
        assert NOETHER_DEFAULT_V1.contortion_sign == "+1"
        assert NOETHER_DEFAULT_V1.disformation_sign == "+1"
        assert NOETHER_DEFAULT_V1.ricci_contraction == "first-third"
