"""Typed expression AST with abstract indices.

The Lagrangian and every kernel result live as trees of these nodes, never as
strings (docs/02_TECH_SPEC.md section 4). LaTeX is a rendering, not a source
of truth.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Variance = Literal["up", "down"]


class Index(BaseModel):
    name: str
    variance: Variance

    def flipped(self) -> "Index":
        return Index(name=self.name, variance="up" if self.variance == "down" else "down")


class Num(BaseModel):
    """Rational number p/q."""

    node: Literal["num"] = "num"
    p: int
    q: int = 1


class Sym(BaseModel):
    """Named scalar symbol: a constant or a scalar field, per its ObjectDecl."""

    node: Literal["sym"] = "sym"
    name: str


class Func(BaseModel):
    """Function of scalar arguments, e.g. F(phi) or K(phi, X)."""

    node: Literal["func"] = "func"
    name: str
    args: list["Expr"]


class Tensor(BaseModel):
    """Abstract-index tensor instance, e.g. R_{mu nu} or g^{mu nu}."""

    node: Literal["tensor"] = "tensor"
    name: str
    indices: list[Index] = []


class Deriv(BaseModel):
    """Single derivative; nest for higher derivatives."""

    node: Literal["deriv"] = "deriv"
    op: Literal["covariant", "partial"]
    index: Index
    expr: "Expr"


class Pow(BaseModel):
    node: Literal["pow"] = "pow"
    base: "Expr"
    exp: int


class Prod(BaseModel):
    node: Literal["prod"] = "prod"
    factors: list["Expr"]


class Sum(BaseModel):
    node: Literal["sum"] = "sum"
    terms: list["Expr"]


Expr = Annotated[
    Num | Sym | Func | Tensor | Deriv | Pow | Prod | Sum,
    Field(discriminator="node"),
]

for _model in (Func, Tensor, Deriv, Pow, Prod, Sum):
    _model.model_rebuild()


# -- convenience constructors (keep eval/test code readable) -----------------


def up(name: str) -> Index:
    return Index(name=name, variance="up")


def down(name: str) -> Index:
    return Index(name=name, variance="down")


def num(p: int, q: int = 1) -> Num:
    return Num(p=p, q=q)


def tensor(name: str, *indices: Index) -> Tensor:
    return Tensor(name=name, indices=list(indices))


def cov(index: Index, expr: Expr) -> Deriv:
    return Deriv(op="covariant", index=index, expr=expr)


def prod(*factors: Expr) -> Prod:
    return Prod(factors=list(factors))


def add(*terms: Expr) -> Sum:
    return Sum(terms=list(terms))
