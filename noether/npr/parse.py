"""LaTeX action parsing: a deterministic, purely syntactic LaTeX -> NPR Expr.

This is the INGEST front door's syntax half (docs/02_TECH_SPEC.md section 3.1,
`parse_latex`). It turns the scalar Lagrangian density a physicist writes into
the typed NPR expression tree (`noether.npr.ast`). It is deliberately and only
syntactic: it never decides field roles, symmetries, gauges, or conventions.
Every physics choice is left to elicitation (see noether.orchestrator.ingest,
which builds the ambiguity ledger on top of this parser).

Supported grammar (the subset the five acceptance actions in docs/04_EVALS.md
live in):

  expr     := term (('+' | '-') term)*
  term     := ('-' | '+')? factor (factor | '\\cdot' factor | '*' factor)*
              with numeric '/' folded into rationals
  factor   := atom ('^' group)?            # '^' + number is a power
  atom     := number | '\\frac'/'\\tfrac' group group
            | '(' expr ')'
            | deriv | box
            | name index_group*            # tensor / scalar / function call

Lexical conventions (documented, not physics inference):
  - A name in GEOMETRIC_NAMES (R, G, C, W) with no indices is still a rank-0
    tensor node (e.g. the Ricci scalar R), not a generic scalar symbol.
  - Any name carrying index groups is a Tensor.
  - A bare name with a parenthesised argument list and no indices is a Func
    (e.g. F(\\phi)); a name with neither indices nor args is a scalar Sym.
  - A trailing "(\\Gamma)" on an indexed curvature tensor is read as a
    connection annotation: it is dropped from the expression and recorded on
    the ParseResult so ingest can raise the independent-connection question.

Anything outside this grammar raises ParseError rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noether.npr.ast import (
    Deriv,
    Expr,
    Func,
    Index,
    Num,
    Pow,
    Prod,
    Sum,
    Sym,
    Tensor,
    down,
    up,
)

GEOMETRIC_NAMES = {"R", "G", "C", "W"}

# Backslash commands that are pure spacing and carry no content.
_SPACING_WORDS = {"quad", "qquad", "left", "right"}


class ParseError(ValueError):
    """Raised when an action string falls outside the supported grammar."""


# -- tokenizer ---------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    kind: str  # 'num' | 'name' | 'cmd' | 'punct'
    value: str


def tokenize(s: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "\\":
            j = i + 1
            if j < n and s[j].isalpha():
                k = j
                while k < n and s[k].isalpha():
                    k += 1
                word = s[j:k]
                i = k
                if word in _SPACING_WORDS:
                    continue
                tokens.append(Token("cmd", word))
                continue
            # backslash + non-letter: spacing macro (\, \; \! \: \ ) -> skip
            i = j + 1 if j < n else j
            continue
        if c.isdigit():
            # One token per digit: an unbraced TeX argument (\tfrac12, x^2)
            # grabs a single character, while multi-digit literals are
            # re-merged by the number atom parser.
            tokens.append(Token("num", c))
            i += 1
            continue
        if c.isalpha():
            tokens.append(Token("name", c))
            i += 1
            continue
        if c in "{}()^_+-/*,=":
            tokens.append(Token("punct", c))
            i += 1
            continue
        raise ParseError(f"unexpected character {c!r} at position {i} in {s!r}")
    return tokens


# -- parse result ------------------------------------------------------------


@dataclass
class ParseResult:
    expr: Expr
    connection_annotated: bool = False
    annotated_tensors: list[str] = field(default_factory=list)


# -- parser ------------------------------------------------------------------

_GREEK_INDEX_HINTS = {
    "mu",
    "nu",
    "rho",
    "sigma",
    "lambda",
    "kappa",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "tau",
    "chi",
    "xi",
    "eta",
}


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.toks = tokens
        self.pos = 0
        self.connection_annotated = False
        self.annotated_tensors: list[str] = []
        self._box_counter = 0

    # -- token helpers --
    def _peek(self, offset: int = 0) -> Token | None:
        idx = self.pos + offset
        return self.toks[idx] if 0 <= idx < len(self.toks) else None

    def _next(self) -> Token:
        tok = self._peek()
        if tok is None:
            raise ParseError("unexpected end of input")
        self.pos += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> Token:
        tok = self._next()
        if tok.kind != kind or (value is not None and tok.value != value):
            want = value if value is not None else kind
            raise ParseError(f"expected {want!r}, got {tok.value!r}")
        return tok

    def _at_punct(self, value: str) -> bool:
        tok = self._peek()
        return tok is not None and tok.kind == "punct" and tok.value == value

    # -- groups (braced or single-token argument, TeX-style) --
    def _read_group_tokens(self) -> list[Token]:
        """Read a '{...}' group or a single following token (TeX arg grabbing)."""
        if self._at_punct("{"):
            self._next()
            depth = 1
            collected: list[Token] = []
            while depth:
                tok = self._peek()
                if tok is None:
                    raise ParseError("unbalanced '{' in group")
                if tok.kind == "punct" and tok.value == "{":
                    depth += 1
                elif tok.kind == "punct" and tok.value == "}":
                    depth -= 1
                    if depth == 0:
                        self._next()
                        break
                collected.append(tok)
                self._next()
            return collected
        return [self._next()]

    @staticmethod
    def _group_is_indexlike(group: list[Token]) -> bool:
        if not group:
            return False
        for tok in group:
            if tok.kind == "name" and tok.value.isalpha():
                continue
            if tok.kind == "cmd":
                continue
            return False
        return True

    @staticmethod
    def _group_indices(group: list[Token], variance: str) -> list[Index]:
        out: list[Index] = []
        for tok in group:
            out.append(Index(name=tok.value, variance=variance))
        return out

    # -- entry --
    def parse(self) -> Expr:
        expr = self._parse_sum()
        if self._peek() is not None:
            tok = self._peek()
            assert tok is not None
            raise ParseError(f"trailing tokens starting at {tok.value!r}")
        return expr

    def _parse_sum(self) -> Expr:
        terms = [self._parse_term()]
        while True:
            tok = self._peek()
            if tok is not None and tok.kind == "punct" and tok.value in "+-":
                self._next()
                term = self._parse_term()
                if tok.value == "-":
                    term = _negate(term)
                terms.append(term)
            else:
                break
        return terms[0] if len(terms) == 1 else Sum(terms=terms)

    def _parse_term(self) -> Expr:
        leading_neg = False
        if self._at_punct("-"):
            self._next()
            leading_neg = True
        elif self._at_punct("+"):
            self._next()

        factors = [self._parse_factor()]
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "punct" and tok.value in "+-)}=,":
                break
            if tok.kind == "punct" and tok.value == "*":
                self._next()
                factors.append(self._parse_factor())
                continue
            if tok.kind == "cmd" and tok.value == "cdot":
                self._next()
                factors.append(self._parse_factor())
                continue
            if tok.kind == "punct" and tok.value == "/":
                self._next()
                divisor = self._parse_factor()
                factors[-1] = _divide(factors[-1], divisor)
                continue
            factors.append(self._parse_factor())

        term = factors[0] if len(factors) == 1 else Prod(factors=factors)
        if leading_neg:
            term = _negate(term)
        return term

    def _parse_factor(self) -> Expr:
        atom = self._parse_atom()
        # A trailing '^' + numeric group is a power (index groups were already
        # consumed by the atom parser).
        if self._at_punct("^"):
            save = self.pos
            self._next()
            group = self._read_group_tokens()
            if group and all(t.kind == "num" for t in group):
                return Pow(base=atom, exp=int("".join(t.value for t in group)))
            self.pos = save
        return atom

    def _parse_atom(self) -> Expr:
        tok = self._peek()
        if tok is None:
            raise ParseError("unexpected end of input while parsing atom")

        if tok.kind == "num":
            digits = [self._next().value]
            while (nxt := self._peek()) is not None and nxt.kind == "num":
                digits.append(self._next().value)
            return Num(p=int("".join(digits)))

        if tok.kind == "punct" and tok.value == "(":
            self._next()
            inner = self._parse_sum()
            self._expect("punct", ")")
            return inner

        if tok.kind == "cmd":
            if tok.value in ("frac", "tfrac", "dfrac"):
                self._next()
                numer = _as_number(self._parse_group_expr(), "fraction numerator")
                denom = _as_number(self._parse_group_expr(), "fraction denominator")
                return _rational(numer, denom)
            if tok.value in ("nabla", "partial"):
                return self._parse_deriv(tok.value)
            if tok.value == "Box":
                self._next()
                return self._make_box(self._parse_factor())
            # a greek/command name acting as a symbol or tensor base
            self._next()
            return self._parse_named(tok.value)

        if tok.kind == "name":
            self._next()
            return self._parse_named(tok.value)

        raise ParseError(f"unexpected token {tok.value!r} while parsing atom")

    def _parse_group_expr(self) -> Expr:
        group = self._read_group_tokens()
        return _Parser(group).parse()

    def _parse_deriv(self, op_word: str) -> Expr:
        self._next()  # consume the command token
        tok = self._peek()
        if tok is None or tok.kind != "punct" or tok.value not in "_^":
            raise ParseError(f"\\{op_word} must be followed by an index")
        variance = "down" if tok.value == "_" else "up"
        self._next()
        group = self._read_group_tokens()
        idxs = self._group_indices(group, variance)
        if len(idxs) != 1:
            raise ParseError(f"\\{op_word} takes exactly one index, got {len(idxs)}")
        op = "covariant" if op_word == "nabla" else "partial"
        operand = self._parse_factor()
        return Deriv(op=op, index=idxs[0], expr=operand)

    def _parse_named(self, name: str) -> Expr:
        indices: list[Index] = []
        while True:
            tok = self._peek()
            if tok is None or tok.kind != "punct" or tok.value not in "_^":
                break
            variance = "down" if tok.value == "_" else "up"
            save = self.pos
            self._next()
            group = self._read_group_tokens()
            if not self._group_is_indexlike(group):
                # e.g. '^2' is a power, not an index group: rewind for parse_factor.
                self.pos = save
                break
            indices.extend(self._group_indices(group, variance))

        if indices:
            self._maybe_connection_annotation(name)
            return Tensor(name=name, indices=indices)

        # No indices: function call, geometric scalar, or plain symbol.
        if self._at_punct("("):
            args = self._parse_arg_list()
            return Func(name=name, args=args)
        if name in GEOMETRIC_NAMES:
            return Tensor(name=name, indices=[])
        return Sym(name=name)

    def _parse_arg_list(self) -> list[Expr]:
        self._expect("punct", "(")
        args = [self._parse_sum()]
        while self._at_punct(","):
            self._next()
            args.append(self._parse_sum())
        self._expect("punct", ")")
        return args

    def _maybe_connection_annotation(self, name: str) -> None:
        # Recognise a literal '(\Gamma)' suffix on a curvature tensor.
        t0, t1, t2 = self._peek(0), self._peek(1), self._peek(2)
        if (
            t0 is not None
            and t0.kind == "punct"
            and t0.value == "("
            and t1 is not None
            and t1.kind == "cmd"
            and t1.value == "Gamma"
            and t2 is not None
            and t2.kind == "punct"
            and t2.value == ")"
        ):
            self._next()
            self._next()
            self._next()
            self.connection_annotated = True
            self.annotated_tensors.append(name)

    def _make_box(self, operand: Expr) -> Expr:
        self._box_counter += 1
        dummy = f"box{self._box_counter}"
        return Deriv(
            op="covariant",
            index=up(dummy),
            expr=Deriv(op="covariant", index=down(dummy), expr=operand),
        )


# -- numeric helpers ---------------------------------------------------------


def _negate(expr: Expr) -> Expr:
    if isinstance(expr, Num):
        return Num(p=-expr.p, q=expr.q)
    if isinstance(expr, Prod) and expr.factors and isinstance(expr.factors[0], Num):
        head = expr.factors[0]
        return Prod(factors=[Num(p=-head.p, q=head.q), *expr.factors[1:]])
    return Prod(factors=[Num(p=-1), expr])


def _as_number(expr: Expr, where: str) -> Num:
    if isinstance(expr, Num):
        return expr
    raise ParseError(f"{where} must be a number, got {expr!r}")


def _rational(numer: Num, denom: Num) -> Num:
    from math import gcd

    p = numer.p * denom.q
    q = numer.q * denom.p
    if q == 0:
        raise ParseError("division by zero in fraction")
    if q < 0:
        p, q = -p, -q
    g = gcd(abs(p), q) or 1
    return Num(p=p // g, q=q // g)


def _divide(left: Expr, right: Expr) -> Expr:
    if isinstance(left, Num) and isinstance(right, Num):
        return _rational(left, right)
    raise ParseError("only numeric '/' division is supported")


# -- public API --------------------------------------------------------------


def parse_lagrangian(tex: str) -> Expr:
    """Parse a scalar Lagrangian-density LaTeX string into an NPR Expr."""
    return parse_action(tex).expr


def parse_action(tex: str) -> ParseResult:
    """Parse a Lagrangian density, returning the Expr plus syntactic metadata
    (connection annotations) that ingest needs for the ambiguity ledger."""
    parser = _Parser(tokenize(tex))
    expr = parser.parse()
    return ParseResult(
        expr=expr,
        connection_annotated=parser.connection_annotated,
        annotated_tensors=list(parser.annotated_tensors),
    )
