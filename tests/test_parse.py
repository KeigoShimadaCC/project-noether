"""LaTeX action parser (noether.npr.parse).

The corpus is exactly the five acceptance actions in docs/04_EVALS.md, taken
from each eval's own `lagrangian_tex` so the parser is tested against the same
strings a user would paste. The parser is purely syntactic; physics meaning is
asserted nowhere here (that is ingest's ambiguity ledger, tested separately).
"""

import pytest

from evals import (
    eval1_eh_trace,
    eval2_palatini,
    eval3_scalar_tensor,
    eval4_maxwell,
    eval5_gauss_bonnet,
)
from noether.npr.ast import Func, Pow, Sym, add, cov, down, num, prod, tensor, up
from noether.npr.latex import render
from noether.npr.parse import ParseError, parse_action, parse_lagrangian
from noether.npr.validate import validate_expression

ALL_EVALS = [
    eval1_eh_trace,
    eval2_palatini,
    eval3_scalar_tensor,
    eval4_maxwell,
    eval5_gauss_bonnet,
]


def _tex(mod) -> str:
    return mod.build_npr().action.lagrangian_tex


class TestRoundTrip:
    @pytest.mark.parametrize("mod", ALL_EVALS, ids=lambda m: m.__name__.split(".")[-1])
    def test_parses_and_validates(self, mod):
        expr = parse_lagrangian(_tex(mod))
        validate_expression(expr)  # scalar density: no free indices

    @pytest.mark.parametrize("mod", ALL_EVALS, ids=lambda m: m.__name__.split(".")[-1])
    def test_render_reparse_is_stable(self, mod):
        expr = parse_lagrangian(_tex(mod))
        assert parse_lagrangian(render(expr)) == expr


class TestExactAgainstStoredNpr:
    """Where the eval's stored Lagrangian uses the same dummy-index names as
    its tex (evals 1, 2, 4), the parse must reproduce it exactly."""

    def test_eval1_einstein_trace(self):
        assert (
            parse_lagrangian(_tex(eval1_eh_trace)) == eval1_eh_trace.build_npr().action.lagrangian
        )

    def test_eval2_palatini_drops_connection_annotation(self):
        result = parse_action(_tex(eval2_palatini))
        assert result.connection_annotated is True
        assert result.annotated_tensors == ["R"]
        assert result.expr == eval2_palatini.build_npr().action.lagrangian

    def test_eval4_maxwell(self):
        assert parse_lagrangian(_tex(eval4_maxwell)) == eval4_maxwell.build_npr().action.lagrangian


class TestExpectedTrees:
    """Functions and curvature-scalar powers: faithful syntactic structure."""

    def test_eval3_functions_and_kinetic_term(self):
        phi = Sym(name="phi")
        expected = add(
            prod(Func(name="F", args=[phi]), tensor("R")),
            prod(num(-1, 2), cov(down("mu"), phi), cov(up("mu"), phi)),
            prod(num(-1, 1), Func(name="V", args=[phi])),
        )
        assert parse_lagrangian(_tex(eval3_scalar_tensor)) == expected

    def test_eval5_gauss_bonnet_scalar(self):
        expected = add(
            Pow(base=tensor("R"), exp=2),
            prod(num(-4), tensor("R", down("mu"), down("nu")), tensor("R", up("mu"), up("nu"))),
            prod(
                tensor("R", down("mu"), down("nu"), down("rho"), down("sigma")),
                tensor("R", up("mu"), up("nu"), up("rho"), up("sigma")),
            ),
        )
        assert parse_lagrangian(_tex(eval5_gauss_bonnet)) == expected


class TestPrimitives:
    def test_tfrac_single_token_arguments(self):
        # \tfrac14 is TeX shorthand for \tfrac{1}{4}.
        assert parse_lagrangian(r"\tfrac14") == num(1, 4)

    def test_leading_minus_folds_into_rational(self):
        assert parse_lagrangian(r"-\tfrac14 F_{\mu\nu} F^{\mu\nu}").factors[0] == num(-1, 4)

    def test_power_vs_index_superscript(self):
        assert parse_lagrangian(r"R^2") == Pow(base=tensor("R"), exp=2)
        assert parse_lagrangian(r"R^{\mu\nu}") == tensor("R", up("mu"), up("nu"))

    def test_function_call(self):
        assert parse_lagrangian(r"F(\phi)") == Func(name="F", args=[Sym(name="phi")])

    def test_geometric_scalar_vs_plain_symbol(self):
        assert parse_lagrangian(r"R") == tensor("R")  # curvature scalar
        assert parse_lagrangian(r"V") == Sym(name="V")  # generic scalar

    def test_nonnumeric_division_rejected(self):
        with pytest.raises(ParseError):
            parse_lagrangian(r"\phi / R")

    def test_unexpected_character_rejected(self):
        with pytest.raises(ParseError):
            parse_lagrangian("a & b")
