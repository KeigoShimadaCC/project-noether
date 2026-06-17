"""NPR layer: AST serialization, structural validation, LaTeX determinism."""

import pytest
from pydantic import TypeAdapter

from noether.npr.ast import Deriv, Expr, Sym, add, cov, down, num, prod, tensor, up
from noether.npr.latex import render
from noether.npr.parse import parse_lagrangian
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
